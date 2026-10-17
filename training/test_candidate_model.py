"""Check candidate-order independence and full-word compatibility supervision."""
import unittest
import torch

from autoregressive_model import SpellingDecoder
from candidate_model import CandidateMatcher, candidate_geometry


class CandidateMatcherTests(unittest.TestCase):
    def test_geometry_preserves_repeats_and_endpoints(self):
        tokens, lengths, geometry = candidate_geometry(['book', 'booking', 'qaz'])
        self.assertEqual(lengths.tolist(), [5, 8, 4])
        torch.testing.assert_close(geometry[0, 1], geometry[0, 2])
        self.assertEqual(geometry[0, 4, 3], 1)
        torch.testing.assert_close(geometry[0, 4, :2], geometry[0, 3, :2])
        self.assertTrue(torch.all(geometry[0, 1:4, 2] >= geometry[0, :3, 2]))
        self.assertEqual(int(tokens[0, 4]), 28)

    def test_supervised_candidates_are_shuffled_and_never_duplicated(self):
        from candidate_training import training_candidates
        torch.manual_seed(123)
        scores = torch.randn(32, 60000)
        labels = torch.arange(32)
        ids, prior, targets = training_candidates(scores, labels)
        torch.testing.assert_close(ids.gather(1, targets[:, None])[:, 0], labels)
        torch.testing.assert_close(prior, scores.gather(1, ids))
        self.assertGreater(int(targets.unique().numel()), 8)
        self.assertTrue(all(len(set(row)) == 16 for row in ids.tolist()))
        self.assertTrue(torch.all((ids == labels[:, None]).sum(1) == 1))

    def test_neutral_initialization_and_permutation(self):
        torch.manual_seed(9)
        old = SpellingDecoder(width=48, layers=1)
        model = CandidateMatcher(width=48, layers=1).eval()
        model.initialize(old.state_dict())
        tokens, lengths, geometry = candidate_geometry(['drawing', 'draw', 'gesture', 'queen'])
        ids = torch.tensor([[0, 1, 2], [3, 2, 1]])
        memory = torch.randn(2, 32, 48)
        with torch.no_grad():
            zero = model.candidate_scores(memory, ids, tokens, lengths, geometry)
            torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
            model.compatibility[-1].weight.normal_(std=.1)
            first = model.candidate_scores(memory, ids, tokens, lengths, geometry, chunk=1)
            second = model.candidate_scores(memory, ids.flip(1), tokens, lengths, geometry, chunk=6)
            torch.testing.assert_close(first.flip(1), second, atol=1e-5, rtol=1e-5)
            permutation = torch.tensor([2, 0, 3, 1])
            remapped = permutation.argsort()[ids]
            reordered = model.candidate_scores(memory, remapped, tokens[permutation],
                                                lengths[permutation], geometry[permutation])
            torch.testing.assert_close(first, reordered, atol=1e-5, rtol=1e-5)
        model.train()
        loss = model.candidate_scores(memory, ids, tokens, lengths, geometry).square().mean()
        loss.backward()
        self.assertGreater(float(model.geometry.weight.grad.norm()), 0)
        self.assertGreater(float(model.decoder.layers[0].multihead_attn.in_proj_weight.grad.norm()), 0)
        self.assertIsNone(model.output.weight.grad)
        with self.assertRaises(ValueError):
            model.pair_scores(memory, torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, 4))


if __name__ == '__main__':
    unittest.main()
