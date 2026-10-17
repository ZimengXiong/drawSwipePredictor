"""Predict bounded motor parameters from an observed stroke and a route hypothesis."""
import torch
from torch import nn
from torch.nn import functional as F

from retrieval_model import features


class AmortizedMotor(nn.Module):
    def __init__(self, max_keys):
        super().__init__()
        self.max_keys = max_keys
        self.view = nn.Sequential(nn.Flatten(), nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 192), nn.ReLU())
        self.parameters_head = nn.Sequential(nn.Linear(768 + 3 * max_keys, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, 3 * max_keys + 16))
        nn.init.zeros_(self.parameters_head[-1].weight)
        nn.init.zeros_(self.parameters_head[-1].bias)

    def forward(self, paths, keys, lengths, fitter):
        """Use candidate geometry only; no observed omission flags or target IDs."""
        width = self.max_keys
        if keys.shape != (len(paths), width):
            raise ValueError("route width differs from the trained motor estimator")
        with torch.no_grad():
            initial = torch.zeros(len(paths), 70 + width - 2, device=paths.device)
            template = fitter.render(keys, lengths, initial)
            present = torch.arange(width, device=paths.device)[None] < lengths[:, None]
            vertices = fitter.layout[keys]
            vertices = vertices - (vertices * present[:, :, None]).sum(1, keepdim=True) / lengths[:, None, None]
            radius = (vertices.square().sum(2) * present).sum(1) / lengths
            vertices = vertices / radius.clamp_min(1e-6).sqrt()[:, None, None]
            vertices = vertices * present[:, :, None]
            route = torch.cat([vertices.flatten(1), present.float()], dim=1)
        observed, nominal = self.view(features(paths)), self.view(features(template))
        combined = torch.cat([observed, nominal, observed - nominal, observed * nominal, route], dim=1)
        values = self.parameters_head(combined)
        positional = values[:, :2 * width].reshape(-1, width, 2)
        # Repeated occurrences of a letter share one remembered key location.
        assignment = F.one_hot(keys, 26).transpose(1, 2).float() * present[:, None]
        offsets = torch.bmm(assignment, positional) / assignment.sum(2).clamp_min(1)[:, :, None]
        return torch.cat([offsets.flatten(1), values[:, 2 * width:2 * width + 18], values[:, 2 * width + 18:]], dim=1)


def retained_training_keys(metadata, width):
    """Prepare known visited routes for the training reconstruction loss only."""
    keys = (metadata["keys"][:, :width] - 1).clamp_min(0)
    keys = F.pad(keys, (0, width - keys.shape[1]))
    return keys, metadata["lengths"]
