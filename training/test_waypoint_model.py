"""Check unknown-count inference and dense corner supervision."""
import unittest
import torch
from waypoint_model import WaypointDecoder, supervised_loss


class WaypointModelTests(unittest.TestCase):
    def test_inference_returns_all_counts_without_a_count_oracle(self):
        torch.manual_seed(51)
        model = WaypointDecoder(width=48,layers=1).eval()
        memory, points = torch.randn(2,64,48), torch.randn(2,64,2)
        result = model.infer(memory,points,chunk=7)
        self.assertEqual(result['means'].shape,(2,31,32,2))
        self.assertTrue(torch.isfinite(result['means']).all())
        self.assertTrue(torch.isfinite(result['log_scales']).all())
        torch.testing.assert_close(result['count_log_probabilities'].exp().sum(1),torch.ones(2))
        self.assertEqual(result['valid'].sum(1).tolist(),list(range(2,33)))
        for count in [2,7,32]:
            expected, scales, valid = model.conditioned(memory,points,torch.full((2,),count))
            torch.testing.assert_close(result['means'][:,count-2],expected)
            torch.testing.assert_close(result['log_scales'][:,count-2],scales)
            torch.testing.assert_close(expected[:,0],points[:,0])
            torch.testing.assert_close(expected[:,count-1],points[:,-1])

    def test_corner_and_count_heads_receive_gradients(self):
        model = WaypointDecoder(width=48,layers=1)
        memory, points = torch.randn(2,64,48), torch.randn(2,64,2)
        targets, counts = torch.randn(2,32,2), torch.tensor([4,9])
        loss, _, _ = supervised_loss(model,memory,points,targets,counts)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.location.weight.grad.norm()),0)
        self.assertGreater(float(model.count_head[-1].weight.grad.norm()),0)
        with self.assertRaises(ValueError):
            model.conditioned(memory,points,torch.tensor([1,33]))


if __name__ == '__main__':
    unittest.main()
