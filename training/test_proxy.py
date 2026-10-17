"""Check geometry generation, multi-mode scoring, and honest validation behavior."""
from pathlib import Path
import json
import struct
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from intrinsic_features import intrinsic_features
from motor_sampler import MotorSampler
from retrieval_model import curve
from train_pair import evaluate
from train_proxy import RECORD, corpus, proxy_scores

ROOT = Path(__file__).resolve().parents[1]


class ProxyTests(unittest.TestCase):
    def test_intrinsic_view_ignores_rotation_translation_and_scale(self):
        paths = torch.tensor(curve([[0, 0], [2, 1], [-1, 3], [4, 2]]))[None]
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        torch.testing.assert_close(intrinsic_features(paths), intrinsic_features(paths @ rotation * 23 + 50), atol=2e-5, rtol=2e-5)

    def test_fresh_motor_draws_have_finite_geometry(self):
        sampler = MotorSampler(["hello", "alpacas", "lunchbox"], "cpu")
        labels = sampler.labels(32)
        first, second = sampler.sample(labels), sampler.sample(labels)
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreater((first - second).abs().mean().item(), 0.01)
        torch.testing.assert_close(first.square().sum(2).mean(1), torch.ones(32), atol=1e-5, rtol=1e-5)

    def test_word_mode_order_cannot_change_scores(self):
        query = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
        proxies = torch.randn(20, 4, 128, requires_grad=True)
        score = proxy_scores(query, proxies)
        torch.testing.assert_close(score, proxy_scores(query, proxies.roll(1, dims=1)))
        score.sum().backward()
        self.assertTrue(torch.isfinite(proxies.grad).all())

    def test_validation_does_not_inject_missing_targets(self):
        class FixedRanker(nn.Module):
            def forward(self, query, templates, distance):
                return -distance
        data = {"query": torch.zeros(3, 8, 32), "candidates": torch.arange(6).repeat(3, 1),
                "distances": torch.arange(6).float().repeat(3, 1), "labels": torch.full((3,), 6),
                "difficulty": torch.arange(3)}
        result = evaluate(FixedRanker(), data, torch.zeros(7, 8, 32))
        for metrics in result.values():
            self.assertEqual(metrics["top1"], 0)
            self.assertEqual(metrics["top5"], 0)

    def test_corpus_rejects_repeated_writer_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            rows = np.zeros(2, dtype=RECORD)
            rows["writer"] = 17 * 1_000_003
            path.write_bytes(struct.pack("<8sQIQ", b"SWPCOR01", 2, 2, 17) + rows.tobytes())
            with self.assertRaisesRegex(ValueError, "writer identities"):
                corpus(path)

    def test_rust_corpus_matches_python_curve(self):
        subprocess.run(["cargo", "build", "--release", "-p", "swipe-engine", "--example", "memory_corpus"], cwd=ROOT, check=True, capture_output=True)
        with tempfile.TemporaryDirectory() as directory:
            output, verification = Path(directory) / "tiny.bin", Path(directory) / "reference.json"
            subprocess.run([str(ROOT / "target/release/examples/memory_corpus"), "--output", str(output), "--samples", "16", "--seed", "800019", "--verification", str(verification)], check=True, capture_output=True)
            rows, info = corpus(output)
            self.assertEqual(info["samples"], 16)
            for row, case in zip(rows, json.loads(verification.read_text())):
                np.testing.assert_allclose(row["points"], curve(case["points"]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
