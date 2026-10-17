"""Compare recovered corners with public keyboard routes and possible omissions."""
import math

import torch

from retrieval_model import LAYOUT
from visited_route_model import route_variants


def fit_route_cost(coordinates, means, log_scales, counts, ridge=0.1):
    """Fit a regularized affine pose without changing the order of the corners.

    Coordinates and predicted corners have shape (pairs, 32, 2). The fit starts
    with an orientation-preserving similarity transform. A regularized affine
    correction permits aspect and shear errors. Every active corner contributes.
    """
    if coordinates.shape != means.shape or means.shape != log_scales.shape:
        raise ValueError('route coordinates, means, and uncertainty must match')
    if means.ndim != 3 or means.shape[1:] != (32, 2) or ridge <= 0:
        raise ValueError('expected padded thirty-two-corner routes and positive regularization')
    if counts.shape != (len(means),) or torch.any((counts < 2) | (counts > 32)):
        raise ValueError('each route must contain two to thirty-two corners')
    mask = torch.arange(32, device=means.device)[None] < counts[:, None]
    # Isotropic weights preserve the analytic similarity solution. Bound the
    # relative weights so one overconfident corner cannot determine the pose.
    variance = (2*log_scales).exp().mean(2).clamp_min(0.03**2)
    weights = variance.reciprocal()*mask
    weights = weights / (weights.sum(1, keepdim=True)/counts[:, None]).clamp_min(1e-8)
    weights = weights.clamp_max(10)*mask
    weights = weights/weights.sum(1, keepdim=True)
    x = coordinates-(coordinates*weights[..., None]).sum(1, keepdim=True)
    y = means-(means*weights[..., None]).sum(1, keepdim=True)
    denominator = (x.square().sum(2)*weights).sum(1).clamp_min(1e-8)
    real = ((x*y).sum(2)*weights).sum(1)/denominator
    imaginary = ((x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0])*weights).sum(1)/denominator
    similarity = torch.stack([real, imaginary, -imaginary, real], 1).reshape(-1, 2, 2)
    covariance = x.transpose(1, 2) @ (weights[..., None]*x)
    cross = x.transpose(1, 2) @ (weights[..., None]*y)
    regularization = ridge*denominator
    matrix = covariance+regularization[:, None, None]*torch.eye(2, device=means.device)[None]
    # An explicit two-by-two solve also works on the Metal backend.
    determinant = (matrix[:, 0, 0]*matrix[:, 1, 1]-matrix[:, 0, 1]*matrix[:, 1, 0]).clamp_min(1e-10)
    inverse = torch.stack([matrix[:, 1, 1], -matrix[:, 0, 1],
                           -matrix[:, 1, 0], matrix[:, 0, 0]], 1).reshape(-1, 2, 2)/determinant[:, None, None]
    pose = inverse @ (cross+regularization[:, None, None]*similarity)
    residual = y-x@pose
    gaussian = 0.5*residual.square()/variance[..., None]+0.5*variance.log()[..., None]
    fit = (gaussian.sum(2)*mask).sum(1)/(2*counts)
    # Penalize departures from the similarity fit and rotations outside the
    # generator's broad pose range. These penalties use no query metadata.
    correction = ridge*(pose-similarity).square().sum((1, 2))/similarity.square().sum((1, 2)).clamp_min(1e-8)
    rotation = torch.atan2(imaginary, real).abs()
    angle_penalty = ((rotation-0.65).clamp_min(0)/0.2).square()
    return fit+correction+angle_penalty


class WaypointRouteBank:
    """Marginalize the public spelling of each naturally retrieved candidate."""
    def __init__(self, words, device, omission=0.2835):
        variants = [route_variants(word, omission) for word in words]
        routes = sorted({route for rows in variants for route, _ in rows})
        lookup = {route: index for index, route in enumerate(routes)}
        width = max(map(len, variants))
        ids = torch.zeros(len(words), width, dtype=torch.long)
        priors = torch.full((len(words), width), -torch.inf)
        points = torch.zeros(len(routes), 32, 2)
        for index, route in enumerate(routes):
            if len(route) > 32:
                raise ValueError('candidate route exceeds thirty-two corners')
            points[index, :len(route)] = torch.tensor([LAYOUT[c] for c in route])
        for index, rows in enumerate(variants):
            ids[index, :len(rows)] = torch.tensor([lookup[route] for route, _ in rows])
            priors[index, :len(rows)] = torch.tensor([prior for _, prior in rows])
        self.ids, self.priors, self.coordinates = [v.to(device) for v in [ids, priors, points]]
        self.counts = torch.tensor(list(map(len, routes)), device=device)

    @torch.no_grad()
    def costs(self, prediction, candidates, log_count_prior, chunk=4096):
        if candidates.ndim != 2 or len(candidates) != len(prediction['means']) or log_count_prior.shape != (31,):
            raise ValueError('expected candidate lists and all thirty-one count hypotheses')
        ids, priors = self.ids[candidates], self.priors[candidates]
        present = priors.isfinite()
        queries = torch.arange(len(candidates), device=candidates.device)[:, None, None].expand_as(ids)
        packed = queries*len(self.counts)+ids
        unique, inverse = packed[present].unique(return_inverse=True)
        parts = []
        for part in unique.split(chunk):
            query, route = part//len(self.counts), part % len(self.counts)
            count = self.counts[route]
            means = prediction['means'][query, count-2]
            scales = prediction['log_scales'][query, count-2]
            cost = fit_route_cost(self.coordinates[route], means, scales, count)
            # Correct the count distribution learned from the training mixture.
            cost -= prediction['count_log_probabilities'][query, count-2]-log_count_prior[count-2]
            parts.append(cost)
        likelihood = torch.full_like(priors, -torch.inf)
        likelihood[present] = -torch.cat(parts)[inverse]+priors[present]
        return -likelihood.logsumexp(2)
