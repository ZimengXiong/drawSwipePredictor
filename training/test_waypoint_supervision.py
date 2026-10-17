"""Check observed-stroke normalization of latent corner targets."""
import unittest
import numpy as np
from retrieval_model import curve
from prepare_waypoint_supervision import normalize_targets


class WaypointSupervisionTests(unittest.TestCase):
    def test_normalization_matches_query_features_and_preserves_coordinate_frame(self):
        raw = np.array([[10., 20.], [10., 20.], [30., 25.], [40., -10.], [50., 15.]])
        targets = raw[[0, 2, 3, 4]] + np.array([[0.,0.], [2.,3.], [-4.,1.], [0.,0.]])
        points, corners = normalize_targets(raw, targets)
        np.testing.assert_array_equal(points, curve(raw))
        transformed, transformed_corners = normalize_targets(raw*13+np.array([200.,-100.]), targets*13+np.array([200.,-100.]))
        np.testing.assert_allclose(points, transformed, atol=1e-6)
        np.testing.assert_allclose(corners, transformed_corners, atol=1e-6)
        np.testing.assert_allclose(points[[0,-1]], corners[[0,-1]], atol=1e-6)

    def test_invalid_and_stationary_strokes_are_rejected(self):
        for raw in [[[0.,0.],[0.,0.]], [[0.,0.],[float('nan'),1.]]]:
            with self.assertRaises(ValueError):
                normalize_targets(raw, [[0.,0.],[1.,1.]])
        with self.assertRaises(ValueError):
            normalize_targets([[0.,0.],[1.,1.]], [[0.,0.]])


if __name__ == '__main__':
    unittest.main()
