"""Compare complete candidate words with a stroke using bidirectional attention."""
import torch
from torch import nn

from autoregressive_model import PAD, END, SpellingDecoder, word_tokens
from retrieval_model import LAYOUT


def candidate_geometry(words):
    """Encode public candidate geometry, preserving consecutive repeated letters."""
    _, tokens, lengths = word_tokens(words)
    geometry = torch.zeros(len(words), tokens.shape[1], 4)
    for index, word in enumerate(words):
        points = torch.tensor([LAYOUT[c] for c in word], dtype=torch.float32)
        centered = points-points.mean(0)
        normalized = centered/centered.square().sum(1).mean().sqrt().clamp_min(1e-6)
        arc = torch.cat([torch.zeros(1), torch.diff(points, dim=0).norm(dim=1).cumsum(0)])
        arc = arc/arc[-1].clamp_min(1e-6)
        size = len(word)
        geometry[index, :size, :2] = normalized
        geometry[index, :size, 2] = arc
        geometry[index, size, :2] = normalized[-1]
        geometry[index, size, 2:] = torch.tensor([1., 1.])
    return tokens, lengths, geometry


class CandidateMatcher(SpellingDecoder):
    """Score one public word-stroke pair independently of other candidates.

    Full candidate tokens are hypotheses, not query labels. Bidirectional token
    attention lets each letter use the candidate's complete route. Cross-attention
    compares that route with the observed stroke. A zero-initialized output keeps
    the existing retrieval score intact before training.
    """
    def __init__(self, raw_channels=8, width=384, layers=6):
        super().__init__(raw_channels, width, layers)
        self.geometry = nn.Linear(4, width, bias=False)
        self.compatibility = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.geometry.weight)
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)
        # The old character-output projection is retained for checkpoint transfer.
        # It does not participate in the direct compatibility objective.
        self.output.requires_grad_(False)

    def initialize(self, source, seed=901021):
        from autoregressive_model import expand_decoder, decoder_dimensions
        from widen_spelling import widen_decoder, break_widened_symmetry
        base = SpellingDecoder(self.memory.in_channels-256, self.symbols.embedding_dim,
                               len(self.decoder.layers), self.positions.num_embeddings)
        if decoder_dimensions(source)[0] == self.symbols.embedding_dim:
            expand_decoder(base, source)
        else:
            widen_decoder(base, source)
            break_widened_symmetry(base, source, 0.01, seed)
        missing, unexpected = self.load_state_dict(base.state_dict(), strict=False)
        expected = {key for key in self.state_dict() if key.startswith(('geometry.', 'compatibility.'))}
        if set(missing) != expected or unexpected:
            raise ValueError('candidate initialization has incompatible tensors')

    def pair_scores(self, memory, tokens, geometry):
        if tokens.ndim != 2 or geometry.shape != (*tokens.shape, 4) or len(memory) != len(tokens):
            raise ValueError('each pair needs one stroke, candidate tokens, and public geometry')
        valid = tokens != PAD
        if torch.any(valid.sum(1) < 3) or torch.any((tokens == END).sum(1) != 1):
            raise ValueError('each candidate needs its full word and one end token')
        position = torch.arange(tokens.shape[1], device=tokens.device)
        query = self.symbols(tokens)+self.positions(position)[None]+self.geometry(geometry)
        hidden = self.decoder(query, memory, tgt_key_padding_mask=~valid)
        mean = (hidden*valid[..., None]).sum(1)/valid.sum(1)[:, None]
        end = hidden.gather(1, (valid.sum(1)-1)[:, None, None].expand(-1, 1, hidden.shape[-1])).squeeze(1)
        return self.compatibility(torch.cat([mean, end], 1)).squeeze(1)

    def candidate_scores(self, memory, candidates, tokens, lengths, geometry, chunk=512):
        """Score natural candidates; neither target words nor true counts are inputs."""
        if candidates.ndim != 2 or len(candidates) != len(memory):
            raise ValueError('one candidate list is required per query')
        batch, count = candidates.shape
        ids = candidates.flatten()
        queries = torch.arange(batch, device=memory.device).repeat_interleave(count)
        pieces = []
        for start in range(0, len(ids), chunk):
            part = ids[start:start+chunk]
            width = int(lengths[part].max())
            pieces.append(self.pair_scores(memory[queries[start:start+chunk]],
                          tokens[part, :width], geometry[part, :width]))
        return torch.cat(pieces).reshape(batch, count)
