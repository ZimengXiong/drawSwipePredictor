"""Character evidence for ranking whole-word hypotheses from the same gesture."""
import torch
from torch import nn
from torch.nn import functional as F


class CharacterBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(192, 256, 3, padding=dilation, dilation=dilation), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 192, 3, padding=dilation, dilation=dilation), nn.BatchNorm1d(192))
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, x):
        return F.relu(x + self.layers(x))


class CharacterDecoder(nn.Module):
    def __init__(self, raw_channels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(256 + raw_channels, 192, 1), nn.BatchNorm1d(192), nn.ReLU())
        self.blocks = nn.Sequential(*[CharacterBlock(d) for d in [1, 2, 4, 8]])
        self.output = nn.Conv1d(192, 27, 1)

    def forward(self, sequence, raw, embedding):
        sequence = F.interpolate(sequence, size=64, mode="linear", align_corners=True)
        global_view = embedding[:, :, None].expand(-1, -1, 64)
        values = torch.cat([sequence, raw, global_view], dim=1)
        return self.output(self.blocks(self.stem(values))).transpose(1, 2)


def character_targets(words):
    collapsed = ["".join(c for i, c in enumerate(w) if i == 0 or c != w[i - 1]) for w in words]
    lengths = torch.tensor([len(w) for w in collapsed], dtype=torch.long)
    if int(lengths.max()) > 64:
        raise ValueError("character decoder requires at most 64 collapsed letters")
    targets = torch.zeros(len(words), int(lengths.max()), dtype=torch.long)
    for i, word in enumerate(collapsed):
        targets[i, :len(word)] = torch.tensor([ord(c) - 96 for c in word])
    signatures = {word: i for i, word in enumerate(sorted(set(collapsed)))}
    aliases = torch.tensor([signatures[word] for word in collapsed], dtype=torch.long)
    return targets, lengths, aliases


def candidate_ctc_cost(log_probabilities, candidates, targets, lengths):
    """Return exact CTC costs; no target is added to the candidate list."""
    batch, count = candidates.shape
    candidate_ids = candidates.cpu().reshape(-1)
    expanded = log_probabilities.transpose(0, 1).repeat_interleave(count, dim=1).cpu()
    selected_targets = targets[candidate_ids]
    selected_lengths = lengths[candidate_ids]
    input_lengths = torch.full((batch * count,), expanded.shape[0], dtype=torch.long)
    loss = F.ctc_loss(expanded, selected_targets, input_lengths, selected_lengths, reduction="none", zero_infinity=True)
    return loss.reshape(batch, count).to(log_probabilities.device)
