"""Marginalize visited routes without supplying the intended word to a query model."""
import math

import torch

from autoregressive_model import word_tokens


def route_variants(word, omission_probability=0.2835):
    """Keep repeated visits created by deletion and conserve all route probability."""
    if not 0 <= omission_probability < 1:
        raise ValueError('omission probability must be in [0, 1)')
    route = ''.join(c for i, c in enumerate(word) if i == 0 or c != word[i-1])
    if len(route) < 2:
        raise ValueError('candidate must have a moving route')
    if len(route) <= 4 or omission_probability == 0:
        return [(route, 0.0)]
    return [(route, math.log1p(-omission_probability))] + [
        (route[:i] + route[i+1:], math.log(omission_probability / (len(route) - 2)))
        for i in range(1, len(route) - 1)]


class RouteBank:
    """Score every candidate through its public spelling and possible skipped keys."""
    def __init__(self, words, device, omission_probability=0.2835):
        variants = [route_variants(word, omission_probability) for word in words]
        routes = sorted({route for rows in variants for route, _ in rows})
        ids = {route: i for i, route in enumerate(routes)}
        width = max(map(len, variants))
        self.ids = torch.zeros(len(words), width, dtype=torch.long)
        self.log_prior = torch.full((len(words), width), -torch.inf)
        for i, rows in enumerate(variants):
            self.ids[i, :len(rows)] = torch.tensor([ids[route] for route, _ in rows])
            self.log_prior[i, :len(rows)] = torch.tensor([prior for _, prior in rows])
        self.ids, self.log_prior = self.ids.to(device), self.log_prior.to(device)
        self.inputs, self.targets, self.lengths = [value.to(device) for value in word_tokens(routes)]

    @torch.no_grad()
    def costs(self, model, memory, candidates, chunk=128):
        """Use query points and candidate routes only; no target or route-count oracle."""
        if len(memory) != 1 or candidates.ndim != 1:
            raise ValueError('score one query and its natural candidate list')
        return self.costs_batch(model, memory, candidates[None], chunk)[0]

    @torch.no_grad()
    def costs_batch(self, model, memory, candidates, chunk=256):
        """Pack independent query-route pairs for efficient teacher-forced scoring."""
        if candidates.ndim != 2 or len(memory) != len(candidates):
            raise ValueError('one candidate list is required for each query')
        route_ids = self.ids[candidates]
        prior = self.log_prior[candidates]
        present = prior.isfinite()
        query_ids = torch.arange(len(memory), device=memory.device)[:, None, None].expand_as(route_ids)
        route_count = len(self.inputs)
        packed = query_ids * route_count + route_ids
        unique, inverse = packed[present].unique(return_inverse=True)
        costs = []
        for part in unique.split(chunk):
            queries, routes = part // route_count, part % route_count
            width = int(self.lengths[routes].max())
            logits = model(memory[queries], self.inputs[routes, :width])
            labels = self.targets[routes, :width]
            nll = -logits.log_softmax(-1).gather(2, labels[..., None]).squeeze(-1)
            costs.append((nll * (labels != 0)).sum(1))
        likelihoods = torch.full_like(prior, -torch.inf)
        likelihoods[present] = -torch.cat(costs)[inverse] + prior[present]
        return -likelihoods.logsumexp(2)
