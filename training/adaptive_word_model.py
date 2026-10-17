"""Choose word-expert weights from observed shape and query uncertainty."""
import torch
from torch import nn

from retrieval_model import features


class AdaptiveWordGate(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.shape = nn.Sequential(nn.Conv1d(8, 48, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(48, 64, 5, stride=2, padding=2), nn.GELU(),
                                   nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128 + experts * 4, 128), nn.GELU(), nn.Linear(128, experts))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, paths, expert_scores):
        """Use no candidate identity, frequency, difficulty, or target label."""
        local = self.shape(features(paths))
        top = expert_scores.topk(2, dim=2).values
        statistics = torch.stack([top[..., 0], top[..., 0] - top[..., 1],
                                  expert_scores.mean(2), expert_scores.std(2)], dim=2).flatten(1)
        inputs = torch.cat([local.mean(2), local.amax(2), statistics], dim=1)
        weights = self.head(inputs).softmax(1)
        return (weights[..., None] * expert_scores).sum(1), weights
