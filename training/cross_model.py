"""Rank candidate letter routes by attending to the observed stroke sequence."""
import torch
from torch import nn
from torch.nn import functional as F


class RouteCrossRanker(nn.Module):
    def __init__(self, raw_channels=8, width=192, layers=3):
        super().__init__()
        self.memory = nn.Conv1d(128 + raw_channels, width, 1)
        self.memory_position = nn.Parameter(torch.randn(1, 32, width) * 0.01)
        self.letters = nn.Embedding(27, width, padding_idx=0)
        self.positions = nn.Embedding(64, width)
        block = nn.TransformerDecoderLayer(width, 6, 512, dropout=0.1, batch_first=True, norm_first=True)
        self.cross = nn.TransformerDecoder(block, layers, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(nn.Linear(width * 2 + 129, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sequence, raw, embedding, candidates, similarity, keys, lengths):
        batch, count = candidates.shape
        symbols, sizes = keys[candidates], lengths[candidates]
        width = symbols.shape[2]
        positions = torch.arange(width, device=sequence.device)
        padding = positions[None] >= sizes.reshape(-1, 1)
        route = self.letters(symbols.reshape(-1, width)) + self.positions(positions)[None]
        raw = F.interpolate(raw, size=32, mode="linear", align_corners=True)
        memory = self.memory(torch.cat([sequence, raw], dim=1)).transpose(1, 2) + self.memory_position
        memory = memory[:, None].expand(-1, count, -1, -1).reshape(-1, 32, memory.shape[2])
        attended = self.cross(route, memory, tgt_key_padding_mask=padding)
        pooled = attended.masked_fill(padding[:, :, None], 0).sum(1) / sizes.reshape(-1, 1)
        endpoints = attended[:, 0] + attended[torch.arange(len(attended), device=sequence.device), sizes.reshape(-1) - 1]
        global_view = embedding[:, None].expand(-1, count, -1).reshape(-1, 128)
        joint = torch.cat([pooled, endpoints, global_view, similarity.reshape(-1, 1)], dim=1)
        return 32 * similarity + self.head(joint).reshape(batch, count)
