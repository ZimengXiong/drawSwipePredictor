"""Rank candidate letter routes by attending to the observed stroke sequence."""
import torch
from torch import nn
from torch.nn import functional as F

from retrieval_model import LAYOUT


class RouteCrossRanker(nn.Module):
    def __init__(self, raw_channels=8, width=192, layers=3, route_geometry=False):
        super().__init__()
        self.route_geometry = route_geometry
        if route_geometry:
            self.register_buffer("key_coordinates", torch.tensor([[0.0, 0.0]] + [LAYOUT[chr(i + 97)] for i in range(26)]))
            self.route_projection = nn.Linear(8, width)
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
        if self.route_geometry:
            coordinates = self.key_coordinates[symbols.reshape(-1, width)]
            present = (~padding)[:, :, None]
            coordinates = coordinates - (coordinates * present).sum(1, keepdim=True) / sizes.reshape(-1, 1, 1)
            scale = (coordinates.square().sum(2) * (~padding)).sum(1) / sizes.reshape(-1)
            coordinates = coordinates / scale.clamp_min(1e-6).sqrt()[:, None, None]
            endpoint = coordinates.gather(1, (sizes.reshape(-1, 1, 1) - 1).expand(-1, -1, 2))
            coordinates = torch.where(padding[:, :, None], endpoint, coordinates)
            previous = torch.cat([coordinates[:, :1], coordinates[:, :-1]], dim=1)
            following = torch.cat([coordinates[:, 1:], coordinates[:, -1:]], dim=1)
            progress = positions[None].expand(len(coordinates), -1) / (sizes.reshape(-1, 1) - 1).clamp_min(1)
            route_features = torch.cat([coordinates, coordinates - previous, following - coordinates,
                progress[:, :, None], sizes.reshape(-1, 1, 1).float().log().expand(-1, width, -1) / 4], dim=2)
            route = route + self.route_projection(route_features)
        raw = F.interpolate(raw, size=32, mode="linear", align_corners=True)
        memory = self.memory(torch.cat([sequence, raw], dim=1)).transpose(1, 2) + self.memory_position
        memory = memory[:, None].expand(-1, count, -1, -1).reshape(-1, 32, memory.shape[2])
        attended = self.cross(route, memory, tgt_key_padding_mask=padding)
        pooled = attended.masked_fill(padding[:, :, None], 0).sum(1) / sizes.reshape(-1, 1)
        endpoints = attended[:, 0] + attended[torch.arange(len(attended), device=sequence.device), sizes.reshape(-1) - 1]
        global_view = embedding[:, None].expand(-1, count, -1).reshape(-1, 128)
        joint = torch.cat([pooled, endpoints, global_view, similarity.reshape(-1, 1)], dim=1)
        return 32 * similarity + self.head(joint).reshape(batch, count)
