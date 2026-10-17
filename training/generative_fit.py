"""Fit latent motor errors to dictionary hypotheses using geometry alone.

Each candidate is rendered with shared per-letter offsets, row shifts, rounded
corners, and a global pose. Missing interior keys are explicit latent hypotheses.
Neither target labels nor writer identities enter the fitting procedure.
"""
import math

import torch
from torch.nn import functional as F

from retrieval_model import LAYOUT


def resample(path, count=64):
    distances = (path[:, 1:] - path[:, :-1]).square().sum(2).clamp_min(1e-12).sqrt()
    cumulative = F.pad(distances.cumsum(1), (1, 0))
    targets = cumulative[:, -1:] * torch.linspace(0, 1, count, device=path.device)
    hi = torch.searchsorted(cumulative.detach().contiguous(), targets.detach().contiguous(), right=True).clamp(1, path.shape[1] - 1)
    lo = hi - 1
    fraction = (targets - cumulative.gather(1, lo)) / (cumulative.gather(1, hi) - cumulative.gather(1, lo)).clamp_min(1e-8)
    a = path.gather(1, lo[:, :, None].expand(-1, -1, 2))
    b = path.gather(1, hi[:, :, None].expand(-1, -1, 2))
    path = a + fraction[:, :, None] * (b - a)
    path = path - path.mean(1, keepdim=True)
    return path / path.square().sum(2).mean(1, keepdim=True).clamp_min(1e-8).sqrt()[:, :, None]


class MotorFitter:
    def __init__(self, words, device="mps"):
        self.device = device
        self.words = words
        self.layout = torch.tensor([LAYOUT[chr(i + 97)] for i in range(26)], device=device)
        sequences = [[ord(c) - 97 for j, c in enumerate(word) if j == 0 or c != word[j - 1]] for word in words]
        width = max(max(map(len, sequences)), 3)
        self.keys = torch.tensor([row + [row[-1]] * (width - len(row)) for row in sequences], device=device)
        self.lengths = torch.tensor(list(map(len, sequences)), device=device)

    def render(self, keys, lengths, parameters):
        """Render continuous geometry; parameters contain no observed metadata."""
        count, width = keys.shape
        p = parameters.tanh()
        offsets = p[:, :52].reshape(count, 26, 2) * torch.tensor([0.38, 0.28], device=self.device)
        rows = self.layout[keys, 1].long()
        offset = offsets.gather(1, keys[:, :, None].expand(-1, -1, 2))
        shift = p[:, 52:55].gather(1, rows) * 0.4
        offset = offset + torch.stack([shift, torch.zeros_like(shift)], dim=2)
        points = self.layout[keys] + offset
        positions = torch.arange(width, device=self.device)[None]
        first = points[:, 0] + p[:, 55:57] * torch.tensor([0.3, 0.25], device=self.device)
        last = points[torch.arange(count, device=self.device), lengths - 1] + p[:, 57:59] * torch.tensor([0.3, 0.25], device=self.device)
        points = torch.where((positions == 0)[:, :, None], first[:, None], points)
        points = torch.where((positions >= lengths[:, None] - 1)[:, :, None], last[:, None], points)
        amount = 0.275 + p[:, 59:60, None] * 0.175
        asymmetry = 1 + p[:, 70:70 + width - 2, None] * 0.35
        current = points[:, 1:-1]
        entry = current + amount * (points[:, :-2] - current)
        exit = current + amount * asymmetry * (points[:, 2:] - current)
        starts = torch.cat([points[:, :1], exit[:, :-1]], dim=1)
        t = torch.arange(1, 9, device=self.device)[None, None, :, None] / 8
        lines = starts[:, :, None] + t * (entry - starts)[:, :, None]
        t = torch.arange(1, 11, device=self.device)[None, None, :, None] / 10
        curves = (1 - t).square() * entry[:, :, None] + 2 * (1 - t) * t * current[:, :, None] + t.square() * exit[:, :, None]
        corners = torch.cat([lines, curves], dim=2).reshape(count, (width - 2) * 18, 2)
        dense = torch.cat([points[:, :1], corners, last[:, None].expand(-1, 12, -1)], dim=1)
        batch = torch.arange(count, device=self.device)
        last_exit = exit[batch, (lengths - 3).clamp_min(0)]
        last_exit = torch.where((lengths > 2)[:, None], last_exit, first)
        t = torch.arange(1, 13, device=self.device)[None, :, None] / 12
        final = last_exit[:, None] + t * (last - last_exit)[:, None]
        final_indices = 1 + (lengths - 2).clamp_min(0)[:, None] * 18 + torch.arange(12, device=self.device)[None]
        dense = dense.scatter(1, final_indices[:, :, None].expand(-1, -1, 2), final)
        used = 1 + (lengths - 2).clamp_min(0) * 18 + 12
        dense = torch.where((torch.arange(dense.shape[1], device=self.device)[None] < used[:, None])[:, :, None], dense, last[:, None])
        # Drift and hand tremor progress with pen travel, before the global pose.
        time = torch.arange(dense.shape[1], device=self.device)[None] / (used - 1)[:, None]
        time = time.clamp_max(1)
        phase = p[:, 67:68] * math.pi
        wave = (time * math.tau + phase).sin()
        x = dense[:, :, 0] + p[:, 61:62] * 0.25 * dense[:, :, 1] + p[:, 64:65] * 0.45 * time + 0.08 * p[:, 68:69] * wave
        y = dense[:, :, 1] * (p[:, 62:63] * 0.65).exp() + p[:, 65:66] * 0.4 * time + 0.06 * p[:, 69:70] * wave
        angle = p[:, 63:64] * math.radians(32)
        transformed = torch.stack([angle.cos() * x - angle.sin() * y, angle.sin() * x + angle.cos() * y], dim=2)
        return resample(transformed)

    def hypotheses(self, candidates):
        ids = candidates.reshape(-1)
        keys, lengths = self.keys[ids], self.lengths[ids]
        width = keys.shape[1]
        # Include the unchanged route and every possible interior omission.
        position = torch.arange(width, device=self.device)
        skipped = torch.arange(width - 1, device=self.device)
        index = position[None] + ((position[None] >= skipped[:, None]) & (skipped[:, None] > 0))
        index = index.clamp_max(width - 1)
        expanded = keys[:, None].expand(-1, width - 1, -1).gather(2, index[None].expand(len(keys), -1, -1))
        valid = (skipped[None] == 0) | ((lengths[:, None] > 4) & (skipped[None] < lengths[:, None] - 1))
        sizes = lengths[:, None] - (skipped[None] > 0).long()
        return expanded.reshape(-1, width), sizes.reshape(-1), valid, width - 1

    def fit(self, query, candidates, steps=80, learning_rate=0.08, regularization=0.0002, initializer=None):
        """Return fit errors and parameter penalties for exact candidate lists."""
        keys, lengths, valid, variants = self.hypotheses(candidates)
        keep = valid.reshape(-1)
        keys, lengths = keys[keep], lengths[keep]
        repeated = query[:, None, None].expand(-1, candidates.shape[1], variants, -1, -1).reshape(-1, 64, 2)[keep]
        if initializer is None:
            parameters = torch.zeros(len(keys), 70 + keys.shape[1] - 2, device=self.device, requires_grad=True)
        else:
            with torch.no_grad():
                suggested = initializer(repeated, keys, lengths, self)
            parameters = suggested.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([parameters], lr=learning_rate)
        best = torch.full((len(keys),), torch.inf, device=self.device)
        best_error, best_penalty = best.clone(), best.clone()
        for step in range(max(steps, 1)):
            path = self.render(keys, lengths, parameters)
            error = (path - repeated).square().mean((1, 2))
            penalty = parameters.tanh().square().mean(1)
            objective = error + regularization * penalty
            with torch.no_grad():
                improved = objective < best
                best_error = torch.where(improved, error, best_error)
                best_penalty = torch.where(improved, penalty, best_penalty)
                best = torch.minimum(best, objective)
            if steps > 0:
                optimizer.zero_grad(set_to_none=True)
                objective.sum().backward()
                optimizer.step()
        shape = (*candidates.shape, variants)
        errors = torch.full((keep.numel(),), torch.inf, device=self.device)
        penalties = torch.zeros_like(errors)
        errors[keep], penalties[keep] = best_error, best_penalty
        return errors.reshape(shape), penalties.reshape(shape)
