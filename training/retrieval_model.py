"""A shared gesture encoder for contrastive word-route retrieval."""

from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POINTS = 64
EMBEDDING = 128
ROWS = [("qwertyuiop", 0.0, 0.0), ("asdfghjkl", 0.5, 1.0), ("zxcvbnm", 1.5, 2.0)]
LAYOUT = {c: (i + offset, y) for row, offset, y in ROWS for i, c in enumerate(row)}


def curve(points) -> np.ndarray:
    """Center and resample a finite path by arc length in physical coordinates."""
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("expected a finite moving 2D path")
    path = path - path[0]
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path = path[np.r_[True, distances > 1e-10]]
    if len(path) < 2:
        raise ValueError("stationary path")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    targets = np.linspace(0.0, cumulative[-1], POINTS)
    path = np.stack([np.interp(targets, cumulative, path[:, a]) for a in range(2)], axis=1)
    path -= path.mean(axis=0)
    path /= max(np.sqrt(np.mean(np.sum(path * path, axis=1))), 1e-10)
    return path.astype(np.float32)


def vocabulary(path: Path, limit: int):
    words, counts, curves = [], [], []
    for line in path.read_text().splitlines():
        word, count = line.split("\t")
        if len(word) < 2 or not word.isascii() or not word.islower() or not word.isalpha() or len(set(word)) < 2:
            continue
        words.append(word)
        counts.append(float(count))
        curves.append(curve([LAYOUT[c] for c in word]))
        if len(words) == limit:
            break
    return words, np.asarray(counts), np.stack(curves)


def load_recordings(path: Path, word_ids: dict[str, int]):
    """Read prepared real or synthetic JSONL; cache resampled curves by content."""
    import hashlib
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache = path.with_suffix(f".{fingerprint}.curves.npz")
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        names = saved["words"].tolist()
        valid = [i for i, word in enumerate(names) if word in word_ids]
        return saved["curves"][valid], np.asarray([word_ids[names[i]] for i in valid], dtype=np.int64)
    paths, names = [], []
    with path.open() as source:
        for line in source:
            sample = json.loads(line)
            if sample["word"] not in word_ids:
                continue
            try:
                paths.append(curve(sample["points"]))
                names.append(sample["word"])
            except ValueError:
                continue
    if not paths:
        raise ValueError(f"no valid in-vocabulary recordings in {path}")
    paths = np.stack(paths)
    np.savez_compressed(cache, curves=paths, words=np.asarray(names))
    return paths, np.asarray([word_ids[w] for w in names], dtype=np.int64)


def features(paths: torch.Tensor) -> torch.Tensor:
    """Combine shape with and without axis stretch normalization and tangents."""
    paths = paths - paths.mean(dim=1, keepdim=True)
    rms = paths.square().sum(dim=2).mean(dim=1, keepdim=True).sqrt().unsqueeze(2).clamp_min(1e-6)
    iso = paths / rms
    axis = paths / paths.square().mean(dim=1, keepdim=True).sqrt().clamp_min(rms * 0.35)
    delta_axis = torch.cat([axis[:, 1:2] - axis[:, :1], axis[:, 2:] - axis[:, :-2], axis[:, -1:] - axis[:, -2:-1]], dim=1)
    delta_iso = torch.cat([iso[:, 1:2] - iso[:, :1], iso[:, 2:] - iso[:, :-2], iso[:, -1:] - iso[:, -2:-1]], dim=1)
    return torch.cat([iso, axis, delta_axis, delta_iso], dim=2).transpose(1, 2).contiguous()


class GestureEncoder(nn.Module):
    """Encode ordered route geometry into a normalized retrieval vector."""

    def __init__(self):
        super().__init__()
        stages = []
        for source, target, stride in [(8, 48, 1), (48, 64, 2), (64, 96, 2), (96, 128, 2)]:
            stages.extend([nn.Conv1d(source, target, 5, stride=stride, padding=2), nn.BatchNorm1d(target), nn.ReLU()])
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, EMBEDDING))

    def forward(self, inputs):
        return F.normalize(self.head(self.stages(inputs)), dim=1)


@torch.no_grad()
def augment(paths: torch.Tensor, *, force_memory: bool = False) -> torch.Tensor:
    """Distort routes coherently, then remove acquisition-speed differences."""
    batch = len(paths)
    device = paths.device
    strength = torch.ones(batch, 1, 1, device=device) if force_memory else torch.randint(0, 5, (batch, 1, 1), device=device).float() / 4.0
    angle = (torch.rand(batch, 1, 1, device=device) * 2 - 1) * 0.48 * strength
    stretch = ((torch.rand(batch, 1, 1, device=device) * 2 - 1) * 0.6 * strength).exp()
    slant = (torch.rand(batch, 1, 1, device=device) * 2 - 1) * 0.2 * strength
    x, y = paths[:, :, :1], paths[:, :, 1:]
    x, y = x + slant * y, y * stretch
    result = torch.cat([angle.cos() * x - angle.sin() * y, angle.sin() * x + angle.cos() * y], dim=2)
    noise = F.interpolate(torch.randn(batch, 2, 8, device=device), size=POINTS, mode="linear", align_corners=True).transpose(1, 2)
    result += noise * strength * 0.07
    t = torch.linspace(0, 1, POINTS, device=device).view(1, -1, 1)
    result += (torch.rand(batch, 1, 2, device=device) * 2 - 1) * t * strength * 0.18
    endpoint_error = torch.randn(batch, 1, 2, device=device) * (torch.exp(-t * 16) + torch.exp(-(1 - t) * 16))
    result += endpoint_error * strength * 0.08
    smoothed = F.avg_pool1d(result.transpose(1, 2), 7, stride=1, padding=3, count_include_pad=False).transpose(1, 2)
    amount = torch.rand(batch, 1, 1, device=device) * strength * 0.65
    result = result * (1 - amount) + smoothed * amount
    starts = torch.randint(4, 45, (batch,), device=device)
    ends = starts + torch.randint(4, 13, (batch,), device=device)
    positions = torch.arange(POINTS, device=device).view(1, -1)
    bridge = ((positions - starts[:, None]) / (ends - starts)[:, None]).clamp(0, 1).unsqueeze(2)
    a = result[torch.arange(batch, device=device), starts].unsqueeze(1)
    b = result[torch.arange(batch, device=device), ends].unsqueeze(1)
    mask = ((positions >= starts[:, None]) & (positions <= ends[:, None])).unsqueeze(2)
    mask = mask & (torch.rand(batch, 1, 1, device=device) < strength * 0.12)
    result = torch.where(mask, a + bridge * (b - a), result)
    result += torch.randn_like(result) * strength * 0.008
    distances = (result[:, 1:] - result[:, :-1]).square().sum(dim=2).sqrt()
    cumulative = F.pad(distances.cumsum(dim=1), (1, 0))
    targets = cumulative[:, -1:] * torch.linspace(0, 1, POINTS, device=device)
    indices = torch.searchsorted(cumulative.contiguous(), targets.contiguous(), right=True).clamp(1, POINTS - 1)
    lo = cumulative.gather(1, indices - 1)
    hi = cumulative.gather(1, indices)
    weight = ((targets - lo) / (hi - lo).clamp_min(1e-6)).unsqueeze(2)
    left = result.gather(1, (indices - 1).unsqueeze(2).expand(-1, -1, 2))
    right = result.gather(1, indices.unsqueeze(2).expand(-1, -1, 2))
    return left + weight * (right - left)
