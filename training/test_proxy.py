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
from character_edit import ctc_edit_components, edit_log_likelihood
from character_model import character_targets, candidate_ctc_cost
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

    def test_training_metadata_does_not_change_sampled_geometry(self):
        sampler = MotorSampler(["helical", "abracadabra"], "cpu")
        labels = torch.arange(2).repeat(16)
        torch.manual_seed(81)
        expected = sampler.sample(labels, strength=1)
        torch.manual_seed(81)
        actual, metadata = sampler.sample(labels, strength=1, return_metadata=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue((metadata["keys"][:, 0] == sampler.keys[labels, 0] + 1).all())
        self.assertTrue((metadata["lengths"] == sampler.lengths[labels] - metadata["dropped"].long()).all())

    def test_latent_deletion_matches_explicit_ctc_enumeration(self):
        words = ["abaca", "abcdefg", "abcd"]
        targets, lengths, _ = character_targets(words)
        probabilities = torch.randn(3, 18, 27, requires_grad=True).log_softmax(2)
        candidates = torch.arange(3)[:, None]
        ordinary, deleted = ctc_edit_components(probabilities, candidates, targets, lengths)
        expected = -candidate_ctc_cost(probabilities, candidates, targets, lengths)
        torch.testing.assert_close(ordinary, expected, atol=1e-5, rtol=1e-5)
        for i, word in enumerate(words):
            variants = [word[:j] + word[j + 1:] for j in range(1, len(word) - 1)]
            # Keep repeated retained letters: CTC must insert a separating blank.
            variant_targets = torch.tensor([[ord(c) - 96 for c in w] for w in variants])
            variant_lengths = torch.full((len(variants),), len(word) - 1, dtype=torch.long)
            costs = candidate_ctc_cost(probabilities[i:i + 1], torch.arange(len(variants))[None], variant_targets, variant_lengths)
            torch.testing.assert_close(deleted[i, 0], (-costs).logsumexp(1)[0], atol=1e-5, rtol=1e-5)
        combined = edit_log_likelihood(ordinary, deleted, lengths[:, None], 0.3)
        torch.testing.assert_close(combined[2], ordinary[2])
        gradients = torch.autograd.grad(combined.sum(), probabilities)[0]
        self.assertTrue(torch.isfinite(gradients).all())

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
