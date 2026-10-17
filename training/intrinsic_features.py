"""Geometry views that reduce rotation and stretch without discarding raw shape."""
import torch


def intrinsic_features(paths):
    paths = paths - paths.mean(1, keepdim=True)
    paths = paths / paths.square().sum(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)[:, :, None]
    x, y = paths[:, :, 0], paths[:, :, 1]
    floor = (x.square().mean(1) + y.square().mean(1)) * 0.025
    a = x.square().mean(1) + floor
    d = y.square().mean(1) + floor
    b = (x * y).mean(1)
    determinant_root = (a * d - b.square()).clamp_min(1e-8).sqrt()
    scale = (a + d + 2 * determinant_root).sqrt()
    denominator = ((a + determinant_root) * (d + determinant_root) - b.square()).clamp_min(1e-8)
    u = scale[:, None] * ((d + determinant_root)[:, None] * x - b[:, None] * y) / denominator[:, None]
    v = scale[:, None] * ((a + determinant_root)[:, None] * y - b[:, None] * x) / denominator[:, None]
    whitened = torch.stack([u, v], dim=2)
    chord = whitened[:, -1] - whitened[:, 0]
    fallback = whitened[:, 8] - whitened[:, 0]
    anchor = torch.where((chord.square().sum(1) > 0.0225)[:, None], chord, fallback)
    anchor = anchor / anchor.norm(dim=1, keepdim=True).clamp_min(1e-6)
    horizontal = (whitened * anchor[:, None]).sum(2)
    vertical = whitened[:, :, 1] * anchor[:, None, 0] - whitened[:, :, 0] * anchor[:, None, 1]
    frame = torch.stack([horizontal, vertical], dim=2)
    delta = torch.cat([frame[:, 1:2] - frame[:, :1], frame[:, 2:] - frame[:, :-2], frame[:, -1:] - frame[:, -2:-1]], dim=1)
    direction = delta / delta.norm(dim=2, keepdim=True).clamp_min(1e-6)
    previous = torch.cat([direction[:, :1], direction[:, :-1]], dim=1)
    turn_cos = (previous * direction).sum(2)
    turn_sin = previous[:, :, 0] * direction[:, :, 1] - previous[:, :, 1] * direction[:, :, 0]
    radius = frame.square().sum(2).sqrt()
    area = frame[:, :-1, 0] * frame[:, 1:, 1] - frame[:, :-1, 1] * frame[:, 1:, 0]
    area = torch.nn.functional.pad(area.cumsum(1), (1, 0)) / 4
    return torch.cat([frame, delta, radius[:, :, None], turn_cos[:, :, None], turn_sin[:, :, None], area[:, :, None]], dim=2).transpose(1, 2).contiguous()
