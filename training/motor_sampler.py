"""Fresh motor traces with latent key errors, missed waypoints, and rounded turns.

This independent GPU sampler follows the public training error model. It never
reads evaluation recordings, identities, seeds, or target-word ranges.
"""
import math

import torch

from retrieval_model import LAYOUT


class MotorSampler:
    def __init__(self, words, device):
        self.device = device
        coordinates = [LAYOUT[chr(i + 97)] for i in range(26)]
        self.layout = torch.tensor(coordinates, device=device)
        sequences, lengths, eligible = [], [], []
        for i, word in enumerate(words):
            sequence = [ord(c) - 97 for j, c in enumerate(word) if j == 0 or c != word[j - 1]]
            if 4 <= len(word) <= 12 and len(sequence) >= 2:
                eligible.append(i)
            sequence = sequence[:12]
            lengths.append(len(sequence))
            sequences.append(sequence + [sequence[-1]] * (12 - len(sequence)))
        self.keys = torch.tensor(sequences, device=device)
        self.lengths = torch.tensor(lengths, device=device)
        self.eligible = torch.tensor(eligible, device=device)

    def labels(self, count):
        return self.eligible[torch.randint(len(self.eligible), (count,), device=self.device)]

    @torch.no_grad()
    def sample(self, labels, strength=None, return_metadata=False):
        batch, device = len(labels), self.device
        def signed(*shape):
            return torch.rand(*shape, device=device) * 2 - 1
        if strength is None:
            choice = torch.rand(batch, device=device)
            strength = torch.where(choice < 0.1, 0.0, torch.where(choice < 0.3, 0.55, 1.0))
        elif isinstance(strength, (int, float)):
            strength = torch.full((batch,), float(strength), device=device)
        magnitude = strength[:, None, None]
        ids = self.keys[labels]
        lengths = self.lengths[labels]
        key_error = signed(batch, 26, 2) * torch.tensor([0.38, 0.28], device=device)
        row_shift = signed(batch, 3) * 0.4
        rows = self.layout[ids, 1].long()
        offsets = key_error.gather(1, ids[:, :, None].expand(-1, -1, 2))
        offsets[:, :, 0] += row_shift.gather(1, rows)
        keys = self.layout[ids] + offsets * magnitude
        drop = (torch.rand(batch, device=device) < strength * 0.35) & (lengths > 4)
        skipped = 1 + (torch.rand(batch, device=device) * (lengths - 2)).long()
        indices = torch.arange(12, device=device)[None].expand(batch, -1)
        compressed = (indices + ((indices >= skipped[:, None]) & drop[:, None])).clamp_max(11)
        keys = keys.gather(1, compressed[:, :, None].expand(-1, -1, 2))
        lengths = lengths - drop.long()
        batch_ids = torch.arange(batch, device=device)
        endpoint_error = signed(batch, 2, 2) * torch.tensor([0.3, 0.25], device=device) * magnitude
        keys[:, 0] += endpoint_error[:, 0]
        keys[batch_ids, lengths - 1] += endpoint_error[:, 1]
        # Pad with the actual endpoint, so unused corners add no path length.
        last = keys[batch_ids, lengths - 1]
        keys = torch.where((indices < lengths[:, None])[:, :, None], keys, last[:, None])
        amount = (0.1 + torch.rand(batch, 1, 1, device=device) * 0.35) * magnitude
        current = keys[:, 1:-1]
        entry = current + amount * (keys[:, :-2] - current)
        exit = current + amount * (0.65 + 0.7 * torch.rand(batch, 10, 1, device=device)) * (keys[:, 2:] - current)
        start = torch.cat([keys[:, :1], exit[:, :-1]], dim=1)
        t_line = torch.arange(1, 9, device=device)[None, None, :, None] / 8
        line = start[:, :, None] + t_line * (entry - start)[:, :, None]
        t_curve = torch.arange(1, 11, device=device)[None, None, :, None] / 10
        bezier = (1 - t_curve).square() * entry[:, :, None] + 2 * (1 - t_curve) * t_curve * current[:, :, None] + t_curve.square() * exit[:, :, None]
        corners = torch.cat([line, bezier], dim=2).reshape(batch, 180, 2)
        dense = torch.cat([keys[:, :1], corners, last[:, None].expand(-1, 12, -1)], dim=1)
        last_exit = exit[batch_ids, (lengths - 3).clamp_min(0)]
        last_exit = torch.where((lengths > 2)[:, None], last_exit, keys[:, 0])
        final_t = torch.arange(1, 13, device=device)[None, :, None] / 12
        final_line = last_exit[:, None] + final_t * (last - last_exit)[:, None]
        final_indices = 1 + (lengths - 2)[:, None] * 18 + torch.arange(12, device=device)[None]
        dense.scatter_(1, final_indices[:, :, None].expand(-1, -1, 2), final_line)
        dense_lengths = 1 + (lengths - 2) * 18 + 12
        dense = torch.where((torch.arange(193, device=device)[None] < dense_lengths[:, None])[:, :, None], dense, last[:, None])
        # Simulate sparse, uneven acquisition before arc-length resampling.
        counts = torch.randint(45, 185, (batch,), device=device)
        t = (torch.arange(184, device=device)[None] / (counts - 1)[:, None]).clamp_max(1)
        t = t.pow((signed(batch, 1) * 0.6).exp())
        position = t * (dense_lengths - 1)[:, None]
        lo = position.long().clamp_max(191)
        weight = (position - lo).unsqueeze(2)
        a = dense.gather(1, lo[:, :, None].expand(-1, -1, 2))
        b = dense.gather(1, (lo + 1)[:, :, None].expand(-1, -1, 2))
        path = a + weight * (b - a)
        phase = torch.rand(batch, 1, device=device) * math.tau
        wave = (math.tau * t + phase).sin()
        drift = signed(batch, 1, 2) * torch.tensor([0.45, 0.4], device=device)
        slant = signed(batch, 1) * 0.25
        x = path[:, :, 0] + strength[:, None] * (slant * path[:, :, 1] + drift[:, :, 0] * t + 0.08 * wave)
        y = path[:, :, 1] * (signed(batch, 1) * 0.65 * strength[:, None]).exp() + strength[:, None] * (drift[:, :, 1] * t + 0.06 * wave)
        angle = (signed(batch, 1) * math.radians(28) + signed(batch, 1) * math.radians(4)) * strength[:, None]
        path = torch.stack([angle.cos() * x - angle.sin() * y, angle.sin() * x + angle.cos() * y], dim=2)
        path += signed(batch, 184, 2) * magnitude * 0.025
        endpoint = path[batch_ids, counts - 1]
        path = torch.where((torch.arange(184, device=device)[None] < counts[:, None])[:, :, None], path, endpoint[:, None])
        distances = (path[:, 1:] - path[:, :-1]).norm(dim=2)
        cumulative = torch.nn.functional.pad(distances.cumsum(1), (1, 0))
        targets = cumulative[:, -1:] * torch.linspace(0, 1, 64, device=device)
        hi = torch.searchsorted(cumulative.contiguous(), targets.contiguous(), right=True).clamp(1, 183)
        lo = hi - 1
        weight = ((targets - cumulative.gather(1, lo)) / (cumulative.gather(1, hi) - cumulative.gather(1, lo)).clamp_min(1e-8))[:, :, None]
        a = path.gather(1, lo[:, :, None].expand(-1, -1, 2))
        b = path.gather(1, hi[:, :, None].expand(-1, -1, 2))
        path = a + weight * (b - a)
        path -= path.mean(1, keepdim=True)
        normalized = path / path.square().sum(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)[:, :, None]
        if return_metadata:
            retained = ids.gather(1, compressed) + 1
            retained = retained.masked_fill(indices >= lengths[:, None], 0)
            return normalized, {"keys": retained, "lengths": lengths, "dropped": drop}
        return normalized
