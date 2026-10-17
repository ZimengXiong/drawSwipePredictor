"""Check omission probabilities and route scoring without target-word inputs."""
import math
import unittest

import torch
from torch import nn

from visited_route_model import RouteBank, route_variants


class UniformDecoder(nn.Module):
    def forward(self, memory, inputs):
        return torch.zeros(*inputs.shape, 29, device=inputs.device)


class VisitedRouteTests(unittest.TestCase):
    def test_omissions_conserve_probability_and_retain_created_duplicates(self):
        rows = route_variants('abaca')
        self.assertAlmostEqual(sum(math.exp(prior) for _, prior in rows), 1.0)
        self.assertEqual(rows[0][0], 'abaca')
        self.assertIn('aaca', [route for route, _ in rows])
        self.assertTrue(all(route[0] == 'a' and route[-1] == 'a' for route, _ in rows))
        self.assertEqual(route_variants('book'), [('bok', 0.0)])
        self.assertEqual(route_variants('abaca', 0), [('abaca', 0.0)])
        for probability in [-0.1, 1.0]:
            with self.assertRaises(ValueError):
                route_variants('abaca', probability)

    def test_candidate_likelihood_matches_explicit_route_marginal(self):
        words = ['abaca', 'hello', 'helo', 'lunchbox']
        bank = RouteBank(words, 'cpu')
        costs = bank.costs(UniformDecoder(), torch.zeros(1, 32, 8), torch.arange(len(words)), chunk=2)
        expected = []
        for word in words:
            probability = sum(math.exp(prior) * 29 ** (-len(route)-1)
                              for route, prior in route_variants(word))
            expected.append(-math.log(probability))
        torch.testing.assert_close(costs, torch.tensor(expected))
        self.assertEqual(float(costs[1]), float(costs[2]))
        reordered = bank.costs(UniformDecoder(), torch.zeros(1, 32, 8), torch.tensor([3, 0]), chunk=1)
        torch.testing.assert_close(reordered, costs[torch.tensor([3, 0])])
        with self.assertRaisesRegex(ValueError, 'one query'):
            bank.costs(UniformDecoder(), torch.zeros(2, 32, 8), torch.arange(len(words)))


if __name__ == '__main__':
    unittest.main()
