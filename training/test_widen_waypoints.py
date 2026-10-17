"""Check corner means, uncertainty, counts, and RNG across model expansion."""
import unittest
import torch

from aligned_waypoint_model import AlignedWaypointDecoder
from widen_waypoints import widen_waypoints


class WidenWaypointTests(unittest.TestCase):
    def test_all_count_hypotheses_and_random_state_are_preserved(self):
        torch.manual_seed(900)
        old = AlignedWaypointDecoder(width=48, layers=1).eval()
        new = AlignedWaypointDecoder(width=96, layers=2).eval()
        with torch.no_grad():
            for layer in [old.location, old.log_scale, old.gaps, old.arc_encoding]:
                layer.weight.normal_(std=.01)
        state = torch.random.get_rng_state()
        widen_waypoints(new, old)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        sequence, raw, embedding = torch.randn(2, 128, 32), torch.randn(2, 8, 64), torch.randn(2, 128)
        points = torch.randn(2, 64, 2)
        with torch.no_grad():
            first = old.infer(old.encode_memory(sequence, raw, embedding), points)
            second = new.infer(new.encode_memory(sequence, raw, embedding), points)
        for name in ['means', 'log_scales', 'count_log_probabilities']:
            torch.testing.assert_close(first[name], second[name], atol=1e-4, rtol=1e-4)
        self.assertTrue(torch.equal(first['valid'], second['valid']))
        with self.assertRaises(ValueError):
            widen_waypoints(old, new)


if __name__ == '__main__':
    unittest.main()
