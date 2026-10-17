"""Marginalize one missing interior key without changing exact-word scoring."""
import torch
from torch.nn import functional as F

NEGATIVE = -10000.0


def ctc_edit_components(log_probabilities, candidates, targets, lengths):
    """Return no-deletion and summed single-deletion CTC log likelihoods.

    The second lattice layer permits one jump past an interior key. A repeated
    retained character still requires a blank, as in ordinary CTC. No observed
    deletion flag or evaluation label enters this computation.
    """
    device = log_probabilities.device
    keys = targets.to(device)[candidates]
    sizes = lengths.to(device)[candidates]
    batch, count, width = keys.shape
    symbols = F.pad(keys, (1, 0))
    positions = torch.arange(width + 1, device=device)
    valid = positions[None, None] <= sizes[:, :, None]
    character_state = valid & (positions[None, None] > 0)
    distinct = symbols != F.pad(symbols[:, :, :-1], (1, 0))
    jump_distinct = symbols != F.pad(symbols[:, :, :-2], (2, 0))
    jump_valid = character_state & (positions[None, None] >= 3)
    shape = (batch, count, width + 1)
    blank0 = torch.full(shape, NEGATIVE, device=device)
    blank0[:, :, 0] = 0
    char0 = torch.full_like(blank0, NEGATIVE)
    blank1, char1 = torch.full_like(blank0, NEGATIVE), torch.full_like(blank0, NEGATIVE)
    for t in range(log_probabilities.shape[1]):
        emission = log_probabilities[:, t, None].expand(-1, count, -1).gather(2, symbols)
        blank_emission = log_probabilities[:, t, 0, None, None]
        next_blank0 = torch.logaddexp(blank0, char0) + blank_emission
        next_blank1 = torch.logaddexp(blank1, char1) + blank_emission
        previous_blank0 = F.pad(blank0[:, :, :-1], (1, 0), value=NEGATIVE)
        previous_blank1 = F.pad(blank1[:, :, :-1], (1, 0), value=NEGATIVE)
        previous_char0 = F.pad(char0[:, :, :-1], (1, 0), value=NEGATIVE).masked_fill(~distinct, NEGATIVE)
        previous_char1 = F.pad(char1[:, :, :-1], (1, 0), value=NEGATIVE).masked_fill(~distinct, NEGATIVE)
        normal0 = torch.logaddexp(char0, torch.logaddexp(previous_blank0, previous_char0))
        normal1 = torch.logaddexp(char1, torch.logaddexp(previous_blank1, previous_char1))
        jump_blank = F.pad(blank0[:, :, :-2], (2, 0), value=NEGATIVE)
        jump_char = F.pad(char0[:, :, :-2], (2, 0), value=NEGATIVE).masked_fill(~jump_distinct, NEGATIVE)
        jump = torch.logaddexp(jump_blank, jump_char).masked_fill(~jump_valid, NEGATIVE)
        char0 = (normal0 + emission).masked_fill(~character_state, NEGATIVE)
        char1 = (torch.logaddexp(normal1, jump) + emission).masked_fill(~character_state, NEGATIVE)
        blank0 = next_blank0.masked_fill(~valid, NEGATIVE)
        blank1 = next_blank1.masked_fill(~valid, NEGATIVE)
    ending = sizes[:, :, None]
    ordinary = torch.logaddexp(blank0, char0).gather(2, ending)[:, :, 0]
    deleted = torch.logaddexp(blank1, char1).gather(2, ending)[:, :, 0]
    return ordinary, deleted


def edit_log_likelihood(ordinary, deleted, candidate_lengths, probability):
    if probability == 0:
        return ordinary
    p = torch.where(candidate_lengths > 4, probability, 0.0).float()
    deletion_prior = p.clamp_min(1e-20).log() - (candidate_lengths - 2).clamp_min(1).float().log()
    deletion_prior = deletion_prior.masked_fill(p == 0, -torch.inf)
    return torch.logaddexp(ordinary + (1 - p).log(), deleted + deletion_prior)
