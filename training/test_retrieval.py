"""Regression checks for training geometry and real-data conversion."""

import unittest

import numpy as np
import torch

from fetch_real_swipes import convert
from retrieval_model import GestureEncoder, augment, curve, features


class RetrievalTests(unittest.TestCase):
    def test_features_ignore_translation_and_uniform_scale(self):
        path = np.array([[0, 0], [3, 2], [1, 4], [5, 1]], dtype=float)
        a = curve(path)
        b = curve(path * 37 + [125, -80])
        np.testing.assert_allclose(a, b, atol=1e-6)
        torch.testing.assert_close(features(torch.tensor(a[None])), features(torch.tensor(b[None])))

    def test_augmentation_and_gradient_are_finite(self):
        torch.manual_seed(17)
        path = torch.tensor(curve([[0, 0], [3, 2], [1, 4], [5, 1]]))[None].repeat(8, 1, 1)
        distorted = augment(path, force_memory=True)
        self.assertTrue(torch.isfinite(distorted).all())
        self.assertGreater((distorted - path).abs().mean().item(), 0.01)
        model = GestureEncoder()
        output = model(features(distorted))
        output[:, 0].sum().backward()
        self.assertEqual(output.shape, (8, 128))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(8))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_real_coordinates_restore_canvas_aspect(self):
        row = {"id": 1, "session": "donor", "word": "Hello!", "canvas_width": 400,
               "canvas_height": 160, "data": [{"x": i / 10, "y": i / 10, "t": i} for i in range(6)]}
        sample = convert(row, {"hello"}, "train")
        self.assertEqual(sample["word"], "hello")
        self.assertEqual(sample["points"][1], [40.0, 16.0])
        self.assertEqual(sample["times_ms"], list(range(6)))
        row["data"][3]["x"] = float("nan")
        self.assertIsNone(convert(row, {"hello"}, "train"))


if __name__ == "__main__":
    unittest.main()
