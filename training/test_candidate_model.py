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
            first = model.candidate_scores(memory, ids, tokens, lengths, geometry, chunk=2)
            second = model.candidate_scores(memory, ids.flip(1), tokens, lengths, geometry, chunk=6)
            torch.testing.assert_close(first.flip(1), second, atol=1e-5, rtol=1e-5)
        model.train()
        loss = model.candidate_scores(memory, ids, tokens, lengths, geometry).square().mean()
        loss.backward()
        self.assertGreater(float(model.geometry.weight.grad.norm()), 0)
        self.assertGreater(float(model.memory.weight.grad.norm()) if model.memory.weight.grad is not None else 0, -1)
        self.assertIsNone(model.output.weight.grad)
        with self.assertRaises(ValueError):
            model.pair_scores(memory, torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, 4))


if __name__ == '__main__':
    unittest.main()
