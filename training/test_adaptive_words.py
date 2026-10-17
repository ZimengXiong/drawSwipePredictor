"""Check query-dependent ensemble invariants and trainable gate outputs."""
import unittest

import torch
from torch.nn import functional as F

from adaptive_word_model import AdaptiveWordGate


class AdaptiveWordTests(unittest.TestCase):
    def test_initial_weights_reproduce_uniform_full_vocabulary_scores(self):
        model = AdaptiveWordGate(4)
        paths, scores = torch.randn(3, 64, 2), torch.randn(3, 4, 60)
        combined, weights = model(paths, scores)
        torch.testing.assert_close(weights, torch.full((3, 4), 0.25))
        torch.testing.assert_close(combined, scores.mean(1), atol=0, rtol=0)
        F.cross_entropy(32 * combined, torch.tensor([0, 20, 59])).backward()
        self.assertGreater(float(model.head[-1].weight.grad.norm()), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        optimizer.step()
        _, trained = model(paths, scores)
        self.assertTrue(torch.isfinite(trained).all())
        self.assertTrue(torch.all(trained > 0))
        torch.testing.assert_close(trained.sum(1), torch.ones(3))

    def test_candidate_order_cannot_change_query_weights(self):
        torch.manual_seed(12)
        model = AdaptiveWordGate(4).eval()
        with torch.no_grad():
            model.head[-1].weight.normal_(0, 0.1)
        paths, scores = torch.randn(2, 64, 2), torch.randn(2, 4, 20)
        order = torch.randperm(20)
        first, weights = model(paths, scores)
        second, reordered = model(paths, scores[:, :, order])
        torch.testing.assert_close(reordered, weights)
        torch.testing.assert_close(second, first[:, order])


if __name__ == '__main__':
    unittest.main()
