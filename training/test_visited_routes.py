"""Check omission probabilities and route scoring without target-word inputs."""
import math
import unittest

import torch
from torch import nn

from visited_route_model import RouteBank, route_variants
from autoregressive_model import SpellingDecoder
from analyze_route_ambiguity import route_oracle
from evaluate_visited_routes import score_trials


class UniformDecoder(nn.Module):
    def forward(self, memory, inputs):
        return torch.zeros(*inputs.shape, 29, device=inputs.device)


class VisitedRouteTests(unittest.TestCase):
    def test_analytic_oracle_counts_indistinguishable_spellings(self):
        repeated = route_oracle(['abcd', 'abccd'], 0.35)
        self.assertEqual(repeated['visited_identity_oracle_accuracy'], 0.5)
        omitted = route_oracle(['abcde', 'abde'], 0.35)
        self.assertAlmostEqual(omitted['visited_identity_oracle_accuracy'], (0.65 + 1 + 2*0.35/3)/2)

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

    def test_packed_queries_preserve_independent_route_costs(self):
        torch.manual_seed(31)
        bank = RouteBank(['abaca', 'hello', 'lunchbox'], 'cpu')
        model = SpellingDecoder(width=48, layers=1).eval()
        memory = torch.randn(2, 32, 48)
        candidates = torch.tensor([[0, 1], [2, 0]])
        packed = bank.costs_batch(model, memory, candidates, chunk=3)
        separate = torch.stack([bank.costs(model, memory[i:i+1], candidates[i], chunk=1)
                                for i in range(2)])
        torch.testing.assert_close(packed, separate, atol=1e-5, rtol=1e-6)

    def test_trial_reporting_counts_missing_targets_and_preserves_all_queries(self):
        candidates = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        similarities = torch.tensor([[3., 2., 1.]] * 3)
        costs = -similarities
        trials = score_trials(candidates, similarities, costs,
                              torch.tensor([1, 99, 8]), torch.tensor([2, 2, 2]))
        self.assertEqual(len(trials), 8)
        for trial in trials:
            self.assertEqual(trial['validation']['memory']['samples'], 3)
            self.assertEqual(trial['validation']['memory']['correct_top1'], 1)
            self.assertEqual(trial['validation']['memory']['correct_top5'], 2)

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
