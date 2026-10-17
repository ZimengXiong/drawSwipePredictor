"""A deeper route encoder that preserves local detail before global pooling."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualRouteBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(128, 192, 5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(192), nn.ReLU(),
            nn.Conv1d(192, 128, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(128))
        # Start each block close to the identity while the new head adapts.
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, x):
        return F.relu(x + self.layers(x))


class LargeGestureEncoder(nn.Module):
    """Use six residual context blocks and retain 32 steps through the trunk."""
    def __init__(self, input_channels=8):
        super().__init__()
        self.input_channels = input_channels
        self.stages = nn.Sequential(
            nn.Conv1d(input_channels, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU(),
            nn.Conv1d(48, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualRouteBlock(d) for d in [1, 2, 4, 8, 1, 2]])
        self.downsample = nn.Sequential(nn.Conv1d(128, 192, 3, stride=2, padding=1), nn.BatchNorm1d(192), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(192 * 16, 768), nn.ReLU(), nn.Dropout(0.1), nn.Linear(768, 128))

    def encode_sequence(self, x):
        return self.blocks(self.stages(x))

    def embed_sequence(self, sequence):
        return F.normalize(self.head(self.downsample(sequence)), dim=1)

    def forward(self, x):
        return self.embed_sequence(self.encode_sequence(x))
