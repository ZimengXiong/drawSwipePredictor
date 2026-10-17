"""Compare a gesture and a candidate route through their full alignment matrix."""
import torch
from torch import nn
from torch.nn import functional as F


class PairRanker(nn.Module):
    """Learn local correspondences that a single retrieval vector can lose."""
    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            nn.Conv2d(8, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, query, candidates, distance):
        # Queries: B,C,32. Candidates: B,K,C,32. Each cell compares a possible
        # monotone correspondence, retaining local evidence before pooling.
        batch, count = candidates.shape[:2]
        difference = query[:, None, :6, :, None] - candidates[:, :, :6, None, :]
        position = torch.linspace(0, 1, 32, device=query.device)
        progress = (position[:, None] - position[None, :]).expand(batch, count, 1, 32, 32)
        neural = distance[:, :, None, None, None].expand(batch, count, 1, 32, 32)
        matrix = torch.cat([difference, progress, neural], dim=2).reshape(-1, 8, 32, 32)
        correction = self.head(self.stages(matrix)).reshape(batch, count)
        return -10.0 * distance + correction


def pair_features(features):
    return F.interpolate(features, size=32, mode="linear", align_corners=True)
