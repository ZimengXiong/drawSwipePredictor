"""Check route geometry and probability marginalization without query labels."""
import unittest

import torch

from waypoint_evidence import WaypointRouteBank, fit_route_cost


class WaypointEvidenceTests(unittest.TestCase):
    def test_similarity_pose_and_padding(self):
        torch.manual_seed(7)
        x = torch.randn(3, 32, 2)
        counts = torch.tensor([3, 7, 32])
        matrix = torch.tensor([[1.2, 0.3], [-0.3, 1.2]])
        target = x@matrix+torch.tensor([4., -2.])
        scales = torch.zeros_like(x)
        cost = fit_route_cost(x, target, scales, counts)
        torch.testing.assert_close(cost, torch.zeros(3), atol=1e-10, rtol=0)
        target[0, 3:] = 1000
        torch.testing.assert_close(fit_route_cost(x, target, scales, counts), cost, atol=1e-10, rtol=0)
        target[1, 3, 0] += 2
        self.assertGreater(float(fit_route_cost(x, target, scales, counts)[1]), float(cost[1]))

    def test_candidate_permutation_and_chunking(self):
        torch.manual_seed(18)
        bank = WaypointRouteBank(['keyboard', 'gesture', 'qaz', 'abaca'], 'cpu')
        prediction = {'means': torch.randn(2, 31, 32, 2),
                      'log_scales': torch.zeros(2, 31, 32, 2),
                      'count_log_probabilities': torch.randn(2, 31).log_softmax(1)}
        ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        prior = torch.full((31,), -torch.log(torch.tensor(31.)))
        first = bank.costs(prediction, ids, prior, chunk=2)
        second = bank.costs(prediction, ids.flip(1), prior, chunk=4096)
        torch.testing.assert_close(first.flip(1), second)
        self.assertTrue(torch.all(torch.isfinite(first)))
        torch.testing.assert_close(bank.priors.exp().sum(1), torch.ones(4))
        with self.assertRaises(ValueError):
            fit_route_cost(torch.zeros(1, 32, 2), torch.zeros(1, 32, 2),
                           torch.zeros(1, 32, 2), torch.tensor([1]))


if __name__ == '__main__':
    unittest.main()
