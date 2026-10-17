"""Score complete candidate spellings against stroke features."""

import torch
from torch import nn
from torch.nn import functional as F

PAD, START, END = 0, 27, 28


def decoder_dimensions(state):
    """Read decoder dimensions from checkpoint tensors, including legacy files."""
    indices = {int(key.split(".")[2]) for key in state if key.startswith("decoder.layers.")}
    if not indices or indices != set(range(max(indices) + 1)) or max(indices) >= 12:
        raise ValueError("decoder layers must be contiguous and contain one to twelve blocks")
    return state["symbols.weight"].shape[1], len(indices), state["positions.weight"].shape[0]


def load_decoder(state, raw_channels):
    """Load the complete decoder without silently dropping added blocks."""
    width, layers, positions = decoder_dimensions(state)
    model = SpellingDecoder(raw_channels, width, layers, positions)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def expand_decoder(model, original):
    """Add identity blocks while preserving every existing spelling score."""
    width, layers, positions = decoder_dimensions(original)
    if width != model.symbols.embedding_dim or positions != model.positions.num_embeddings or layers > len(model.decoder.layers):
        raise ValueError("decoder expansion cannot change width, positions, or remove blocks")
    missing, unexpected = model.load_state_dict(original, strict=False)
    expected_missing = {key for key in model.state_dict() if key.startswith("decoder.layers.") and int(key.split(".")[2]) >= layers}
    if set(missing) != expected_missing or unexpected:
        raise ValueError("decoder expansion has incompatible source tensors")
    for block in model.decoder.layers[layers:]:
        for output in [block.self_attn.out_proj, block.multihead_attn.out_proj, block.linear2]:
            output.weight.zero_()
            output.bias.zero_()


def word_tokens(words):
    """Encode full words, retaining repeated letters and the end token."""
    width = max(map(len, words)) + 1
    targets = torch.zeros(len(words), width, dtype=torch.long)
    for i, word in enumerate(words):
        values = [ord(c) - 96 for c in word] + [END]
        targets[i, :len(values)] = torch.tensor(values)
    inputs = F.pad(targets[:, :-1], (1, 0), value=START)
    return inputs, targets, (targets != PAD).sum(1)


class SpellingDecoder(nn.Module):
    """Condition each letter on the stroke and earlier candidate letters."""

    def __init__(self, raw_channels=8, width=192, layers=3, max_tokens=64):
        super().__init__()
        self.memory = nn.Conv1d(256 + raw_channels, width, 1)
        self.memory_position = nn.Parameter(torch.randn(1, 32, width) * 0.01)
        self.symbols = nn.Embedding(29, width, padding_idx=PAD)
        self.positions = nn.Embedding(max_tokens, width)
        block = nn.TransformerDecoderLayer(
            width, 6, width * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(block, layers, norm=nn.LayerNorm(width))
        self.output = nn.Linear(width, 29)

    def encode_memory(self, sequence, raw, embedding):
        raw = F.interpolate(raw, size=32, mode="linear", align_corners=True)
        global_view = embedding[:, :, None].expand(-1, -1, 32)
        joined = torch.cat([sequence, raw, global_view], dim=1)
        return self.memory(joined).transpose(1, 2) + self.memory_position

    def forward(self, memory, inputs):
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device)
        tokens = self.symbols(inputs) + self.positions(positions)[None]
        causal = torch.full((length, length), -torch.inf, device=inputs.device).triu(1)
        return self.output(self.decoder(tokens, memory, tgt_mask=causal))

    def candidate_cost(self, memory, candidates, inputs, targets):
        """Compute exact spelling likelihoods for naturally retrieved candidates."""
        batch, count = candidates.shape
        ids = candidates.reshape(-1)
        expanded = memory[:, None].expand(-1, count, -1, -1).reshape(-1, 32, memory.shape[-1])
        labels = targets[ids]
        logits = self(expanded, inputs[ids])
        token_cost = -logits.log_softmax(-1).gather(2, labels[:, :, None]).squeeze(2)
        return (token_cost * (labels != PAD)).sum(1).reshape(batch, count)
