"""Learn global route corrections around a frozen convolutional encoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from large_encoder import LargeGestureEncoder


class GeometryAttention(nn.Module):
    """Compare all stroke positions with a learned geometric attention bias."""

    def __init__(self, width=192, heads=6):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.geometry = nn.Linear(3, heads, bias=False)
        self.output = nn.Linear(width, width)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(),
            nn.Dropout(0.05), nn.Linear(width * 4, width))
        nn.init.zeros_(self.geometry.weight)

    def forward(self, tokens, geometry):
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm(tokens)).reshape(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.geometry(geometry).permute(0, 3, 1, 2)
        weights = ((query @ key.transpose(-1, -2)) / math.sqrt(width // self.heads) + bias).softmax(-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.output(attended)
        return tokens + self.feedforward(tokens)


class AttentionGestureEncoder(nn.Module):
    """Preserve a frozen baseline and learn geometry-only retrieval corrections."""

    input_channels = 16

    def __init__(self):
        super().__init__()
        self.base = LargeGestureEncoder(8).requires_grad_(False).eval()
        self.input = nn.Linear(16, 192)
        self.position = nn.Parameter(torch.randn(1, 64, 192) * 0.01)
        self.blocks = nn.ModuleList([GeometryAttention() for _ in range(4)])
        self.head = nn.Sequential(nn.LayerNorm(192), nn.Flatten(), nn.Linear(64 * 192, 256),
                                  nn.GELU(), nn.Linear(256, 128))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, inputs):
        raw = inputs.transpose(1, 2)
        iso = raw[:, :, :2]
        squared_distance = (iso[:, :, None] - iso[:, None, :]).square().sum(-1)
        tangent = F.normalize(raw[:, :, 6:8], dim=-1, eps=1e-6)
        direction = tangent @ tangent.transpose(1, 2)
        progress = torch.linspace(0, 1, raw.shape[1], device=raw.device)
        separation = (progress[:, None] - progress[None, :]).abs().expand_as(direction)
        geometry = torch.stack([squared_distance, direction, separation], dim=-1)
        tokens = self.input(raw) + self.position
        for block in self.blocks:
            tokens = block(tokens, geometry)
        with torch.no_grad():
            baseline = self.base(inputs[:, :8])
        return F.normalize(baseline + self.head(tokens), dim=1)
