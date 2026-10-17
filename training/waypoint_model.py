"""Recover latent corners for every possible visited-key count from stroke features."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class WaypointDecoder(nn.Module):
    def __init__(self, raw_channels=8, width=192, layers=3):
        super().__init__()
        self.memory = nn.Conv1d(256+raw_channels, width, 1)
        self.memory_position = nn.Parameter(torch.randn(1, 64, width)*0.01)
        self.slots = nn.Embedding(32, width)
        self.counts = nn.Embedding(33, width)
        self.base_projection = nn.Linear(3, width)
        block = nn.TransformerDecoderLayer(width, 6, width*4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(block, layers, norm=nn.LayerNorm(width))
        self.location = nn.Linear(width, 2)
        self.log_scale = nn.Linear(width, 2)
        self.count_head = nn.Sequential(nn.Linear(width*2, width), nn.GELU(), nn.Linear(width, 31))
        nn.init.zeros_(self.location.weight); nn.init.zeros_(self.location.bias)
        nn.init.zeros_(self.log_scale.weight); nn.init.constant_(self.log_scale.bias, math.log(0.3))

    def encode_memory(self, sequence, raw, embedding):
        sequence = F.interpolate(sequence, size=64, mode='linear', align_corners=True)
        raw = F.interpolate(raw, size=64, mode='linear', align_corners=True)
        global_view = embedding[:, :, None].expand(-1, -1, 64)
        return self.memory(torch.cat([sequence, raw, global_view], dim=1)).transpose(1, 2) + self.memory_position

    def count_logits(self, memory):
        return self.count_head(torch.cat([memory.mean(1), memory.amax(1)], dim=1))

    def conditioned(self, memory, points, hypotheses):
        """Predict corners for supplied count hypotheses; training chooses its loss target."""
        if hypotheses.shape != (len(memory),) or torch.any((hypotheses < 2) | (hypotheses > 32)):
            raise ValueError('each count hypothesis must be between two and thirty-two')
        slots = torch.arange(32, device=memory.device)
        valid = slots[None] < hypotheses[:, None]
        fraction = (slots[None]/(hypotheses[:, None]-1)).clamp_max(1)
        position = fraction*63
        lo = position.long().clamp_max(62)
        weight = (position-lo)[..., None]
        first = points.gather(1, lo[..., None].expand(-1,-1,2))
        last = points.gather(1, (lo+1)[..., None].expand(-1,-1,2))
        base = first + weight*(last-first)
        query = (self.slots(slots)[None] + self.counts(hypotheses)[:, None]
                 + self.base_projection(torch.cat([base, fraction[...,None]], dim=2)))
        hidden = self.decoder(query, memory, tgt_key_padding_mask=~valid)
        return base+self.location(hidden), self.log_scale(hidden).clamp(-5, 1), valid

    @torch.no_grad()
    def infer(self, memory, points, chunk=4):
        """Return every count hypothesis without receiving a true count or word."""
        means, scales = [], []
        batch = len(memory)
        counts = torch.arange(2, 33, device=memory.device)
        for part in counts.split(chunk):
            count = len(part)
            hypotheses = part[None].expand(batch, -1).reshape(-1)
            repeated_memory = memory[:,None].expand(-1,count,-1,-1).reshape(-1,64,memory.shape[-1])
            repeated_points = points[:,None].expand(-1,count,-1,-1).reshape(-1,64,2)
            mean, scale, _ = self.conditioned(repeated_memory, repeated_points, hypotheses)
            means.append(mean.reshape(batch,count,32,2)); scales.append(scale.reshape(batch,count,32,2))
        return {'means': torch.cat(means,1), 'log_scales': torch.cat(scales,1),
                'count_log_probabilities': self.count_logits(memory).log_softmax(1),
                'valid': torch.arange(32,device=memory.device)[None] < counts[:,None]}


def supervised_loss(model, memory, points, targets, counts):
    means, log_scales, valid = model.conditioned(memory, points, counts)
    gaussian = 0.5*((targets-means)*(-log_scales).exp()).square()+log_scales
    per_query = (gaussian*valid[...,None]).sum((1,2))/(2*counts)
    regression = per_query.mean()
    count_loss = F.cross_entropy(model.count_logits(memory), counts-2)
    return regression+0.5*count_loss, regression, count_loss
