"""Check split boundaries and native artifact coverage for geometry mining."""
from pathlib import Path
import tempfile
import unittest
import torch

from prepare_complete_geometry import model_inventory, validate_roles
from train_complete_geometry import derived_features, metrics


class CompleteGeometryTests(unittest.TestCase):
    def test_derived_margin_uses_each_query_and_rejects_popularity(self):
        raw = torch.zeros(2, 3, 20)
        raw[..., 4] = torch.tensor([[3.0, 1.0, 2.0], [8.0, 9.0, 7.0]])
        result = derived_features(raw)
        torch.testing.assert_close(result[..., 2], torch.tensor([[2.0, 0.0, 1.0], [1.0, 2.0, 0.0]]))
        self.assertTrue(torch.all(raw[..., 2] == 0))
        raw[..., 2] = 0.9
        with self.assertRaisesRegex(ValueError, "unused popularity"):
            derived_features(raw)

    def test_metrics_count_missing_targets_and_exact_classes(self):
        candidates = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        scores = torch.tensor([[3.0, 2.0, 1.0]] * 3)
        result = metrics(scores, candidates, torch.tensor([0, 99, 7]), torch.tensor([2, 2, 2]))
        self.assertEqual(result["memory"]["samples"], 3)
        self.assertEqual(result["memory"]["correct_top1"], 1)
        self.assertEqual(result["memory"]["correct_top5"], 2)

    def test_training_rejects_reserved_and_overlapping_writers(self):
        development = {"vocabulary": 60000, "samples": 12000, "sha256": "dev", "seed": 700003,
                       "writer_start": 2000000, "writer_end": 2012000}
        training = {**development, "sha256": "train", "seed": 700001, "samples": 1000000,
                    "writer_start": 0, "writer_end": 1000000}
        validate_roles([training], development, "train", 100)
        for invalid in [development, {**training, "seed": 700007}]:
            with self.assertRaises(ValueError):
                validate_roles([invalid], development, "train", 100)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_roles([training, training], development, "train", 100)
        validate_roles([development], development, "development", 12000)
        for info, count in [(development, 100), ({**development, "sha256": "confirmation"}, 12000)]:
            with self.assertRaisesRegex(ValueError, "every row"):
                validate_roles([info], development, "development", count)

    def test_inventory_detects_added_and_modified_native_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "encoder.bin").write_bytes(b"first")
            original = model_inventory(path)
            (path / "weights.pt").write_bytes(b"unused training checkpoint")
            self.assertEqual(original, model_inventory(path))
            (path / "manifest.json").write_text("{}")
            self.assertNotEqual(original, model_inventory(path))
            before = model_inventory(path)
            (path / "encoder.bin").write_bytes(b"second")
            self.assertNotEqual(before, model_inventory(path))


if __name__ == "__main__":
    unittest.main()
