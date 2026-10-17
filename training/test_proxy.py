"""Check geometry generation, multi-mode scoring, and honest validation behavior."""
from pathlib import Path
import json
import io
import struct
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from intrinsic_features import intrinsic_features
from amortized_motor import AmortizedMotor, retained_training_keys
from autoregressive_model import SpellingDecoder, word_tokens, END
from cross_model import RouteCrossRanker
from large_encoder import expand_encoder, expand_word_space
from generative_fit import MotorFitter, route_mixture_cost
from character_edit import ctc_edit_components, edit_log_likelihood
from character_model import character_targets, candidate_ctc_cost
from motor_sampler import MotorSampler
from retrieval_model import curve, build_encoder, encoder_features
from train_pair import evaluate
from train_proxy import RECORD, corpus, proxy_scores, real_word_groups, balanced_real_indices, proxy_checkpoint_state, training_proxy_vectors
from reference_density import mixture_similarity
from evaluate_training_exposure import exposure_bias
from average_word_models import average_state
from export_ensemble import pair_configuration
from evaluate_spelling import retrieval_pair_weight

ROOT = Path(__file__).resolve().parents[1]


class ProxyTests(unittest.TestCase):
    def test_spelling_retrieval_selection_preserves_models_and_development_split(self):
        selection = {"test_data_used": False, "development": {"sha256": "dev"},
                     "full_vocabulary": 60000,
                     "checkpoints": [{"sha256": name} for name in ["a", "b"]],
                     "selected": {"weights": [0.75, 0.25], "validation": {}}}
        self.assertEqual(retrieval_pair_weight(selection, "a", "b", "dev", 60000), 0.25)
        with self.assertRaisesRegex(ValueError, "declared development"):
            retrieval_pair_weight(selection, "a", "b", "confirmation", 60000)
        with self.assertRaisesRegex(ValueError, "vocabulary differs"):
            retrieval_pair_weight(selection, "a", "b", "dev", 59999)
        with self.assertRaisesRegex(ValueError, "every selected component"):
            retrieval_pair_weight(selection, "a", "wrong", "dev", 60000)
        legacy = {**selection, "development": {"sha256": "dev", "vocabulary": 60000}}
        del legacy["full_vocabulary"]
        self.assertEqual(retrieval_pair_weight(legacy, "a", "b", "dev", 60000), 0.25)
        selection["test_data_used"] = True
        with self.assertRaisesRegex(ValueError, "declared development"):
            retrieval_pair_weight(selection, "a", "b", "dev", 60000)

    def test_raw_proxy_resume_preserves_optimizer_trajectory_and_inference(self):
        torch.manual_seed(900547)
        raw = nn.Parameter(torch.randn(7, 128) * torch.arange(1, 8)[:, None])
        optimizer = torch.optim.AdamW([raw], lr=0.001)
        query = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
        labels = torch.arange(4)
        def update(parameter, opt):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(32 * proxy_scores(query, parameter), labels).backward()
            opt.step()
        update(raw, optimizer)
        state = proxy_checkpoint_state(raw)
        torch.testing.assert_close(proxy_scores(query, state["proxies"]), proxy_scores(query, raw))
        content = io.BytesIO()
        torch.save({**state, "optimizer": optimizer.state_dict()}, content)
        content.seek(0)
        saved = torch.load(content, weights_only=False)
        resumed = nn.Parameter(training_proxy_vectors(saved).clone())
        other = torch.optim.AdamW([resumed], lr=0.001)
        other.load_state_dict(saved["optimizer"])
        update(raw, optimizer)
        update(resumed, other)
        torch.testing.assert_close(resumed, raw, atol=0, rtol=0)
        torch.testing.assert_close(training_proxy_vectors({"proxies": state["proxies"]}), state["proxies"])
        saved["raw_proxies"] = saved["raw_proxies"] + 0.1
        with self.assertRaisesRegex(ValueError, "raw word vectors differ"):
            training_proxy_vectors(saved)

    def test_native_pair_export_does_not_drop_selected_models(self):
        selected = {"checkpoints": [{"sha256": name} for name in ["a", "b", "c"]],
                    "selected": {"weights": [0.0, 0.25, 0.75], "validation": {}}}
        self.assertEqual(pair_configuration(selected, "c", "b")["auxiliary_weight"], 0.25)
        with self.assertRaisesRegex(ValueError, "every selected component"):
            pair_configuration(selected, "c", "a")
        selected["selected"]["weights"] = [0.25, 0.25, 0.5]
        with self.assertRaisesRegex(ValueError, "every selected component"):
            pair_configuration(selected, "c", "b")
        legacy = {"primary_sha256": "c", "auxiliary_sha256": "b",
                  "selected": {"auxiliary_weight": 0.5, "proxy_weight": 2.0}}
        self.assertEqual(pair_configuration(legacy, "c", "b"), legacy["selected"])
        with self.assertRaisesRegex(ValueError, "sources differ"):
            pair_configuration(legacy, "a", "b")

    def test_word_model_averaging_preserves_tracking_buffers_and_rejects_shape_changes(self):
        first = {"weights": torch.randn(3, 2), "tracked": torch.tensor(7)}
        second = {"weights": torch.randn(3, 2), "tracked": torch.tensor(11)}
        self.assertEqual(int(average_state(first, second, 0.5)["tracked"]), 11)
        torch.testing.assert_close(average_state(first, first, 0.4)["weights"], first["weights"])
        with self.assertRaisesRegex(ValueError, "incompatible state tensor"):
            average_state(first, {"weights": torch.randn(4, 2), "tracked": torch.tensor(11)}, 0.5)

    def test_training_exposure_uses_recordings_and_respects_word_permutations(self):
        counts = torch.tensor([100.0, 3.0, 1.0, 0.0])
        for balanced in [False, True]:
            bias = exposure_bias(counts, balanced)
            torch.testing.assert_close(bias, exposure_bias(counts * 7, balanced))
            torch.testing.assert_close(bias.flip(0), exposure_bias(counts.flip(0), balanced))
            self.assertEqual(float(bias[-1]), 0)
        balanced_bias = exposure_bias(counts, True)
        torch.testing.assert_close(balanced_bias[:3], balanced_bias[0].expand(3))
        self.assertGreater(float(exposure_bias(counts, False)[0]), float(exposure_bias(counts, False)[1]))

    def test_expanded_word_space_preserves_ranks_and_activates_new_coordinates(self):
        torch.manual_seed(900449)
        original, expanded = build_encoder("large").eval(), build_encoder("wide512").eval()
        expand_word_space(expanded, original.state_dict())
        paths = torch.randn(4, 64, 2)
        proxies = torch.nn.functional.normalize(torch.randn(13, 128), dim=1)
        wide_proxies = nn.Parameter(torch.nn.functional.pad(proxies, (0, 384)))
        old_scores = proxy_scores(original(encoder_features(original, paths)), proxies)
        new_scores = proxy_scores(expanded(encoder_features(expanded, paths)), wide_proxies)
        torch.testing.assert_close(torch.nn.functional.normalize(old_scores, dim=1),
                                   torch.nn.functional.normalize(new_scores, dim=1), atol=1e-6, rtol=1e-5)
        self.assertEqual(old_scores.argmax(1).tolist(), new_scores.argmax(1).tolist())
        loss = torch.nn.functional.cross_entropy(32 * new_scores, torch.arange(4))
        loss.backward()
        self.assertTrue(torch.isfinite(wide_proxies.grad).all())
        self.assertGreater(wide_proxies.grad[:, 128:].abs().sum().item(), 0)
        self.assertGreater(expanded.head[-1].weight.grad[128:].abs().sum().item(), 0)

    def test_reference_density_is_invariant_to_reference_order_and_duplication(self):
        similarities = torch.tensor([[[0.1, 0.8, 0.3], [0.5, 0.4, 0.2]]])
        for temperature in [0.0, 0.01, 0.1]:
            score = mixture_similarity(similarities, temperature)
            torch.testing.assert_close(score, mixture_similarity(similarities.flip(-1), temperature))
            torch.testing.assert_close(score, mixture_similarity(similarities.repeat_interleave(2, -1), temperature))

    def test_motor_sampler_covers_short_and_long_dictionary_routes(self):
        words = ["at", "hello", "electroencephalographic"]
        sampler = MotorSampler(words, "cpu", max_keys=None)
        labels = torch.arange(3).repeat(4)
        for strength in [0.0, 0.55, 1.0]:
            paths, metadata = sampler.sample(labels, strength=strength, return_metadata=True)
            self.assertEqual(tuple(paths.shape), (12, 64, 2))
            self.assertTrue(torch.isfinite(paths).all())
            expected = sampler.lengths[labels] - metadata["dropped"].long()
            self.assertTrue((metadata["lengths"] == expected).all())
            self.assertGreater(int(sampler.lengths[2]), 12)
            torch.testing.assert_close(paths.square().sum(2).mean(1), torch.ones(12), atol=1e-5, rtol=1e-5)

    def test_spelling_likelihood_is_causal_and_candidate_order_independent(self):
        torch.manual_seed(900419)
        model = SpellingDecoder(width=48, layers=1).eval()
        inputs, targets, lengths = word_tokens(["letter", "later", "lane"])
        self.assertEqual(targets[0, :7].tolist(), [12, 5, 20, 20, 5, 18, END])
        self.assertEqual(lengths.tolist(), [7, 6, 5])
        memory = torch.randn(2, 32, 48)
        original = model(memory, inputs[:2])
        changed = inputs[:2].clone()
        changed[:, 4:] = 26
        torch.testing.assert_close(model(memory, changed)[:, :4], original[:, :4])
        candidates = torch.tensor([[0, 2], [1, 0]])
        costs = model.candidate_cost(memory, candidates, inputs, targets)
        reversed_costs = model.candidate_cost(memory, candidates.flip(1), inputs, targets)
        torch.testing.assert_close(costs.flip(1), reversed_costs)
        costs.sum().backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

    def test_attention_preserves_baseline_and_learns_corrections(self):
        torch.manual_seed(900409)
        model = build_encoder("attention")
        paths = torch.randn(3, 64, 2)
        inputs = encoder_features(model, paths)
        model.train()
        self.assertFalse(model.base.training)
        with torch.no_grad():
            baseline = model.base(inputs[:, :8])
        torch.testing.assert_close(model(inputs), baseline)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        target = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)
        for step in range(2):
            optimizer.zero_grad()
            loss = (model(inputs) - target).square().mean()
            loss.backward()
            if step == 1:
                gradient = model.blocks[0].qkv.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertGreater((model(inputs) - baseline).abs().max().item(), 0)

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

    def test_motor_fitter_preserves_endpoints_in_omission_hypotheses(self):
        fitter = MotorFitter(["abaca", "world", "abcd"], "cpu")
        candidates = torch.arange(3)[None]
        keys, lengths, valid, variants = fitter.hypotheses(candidates)
        keys, lengths = keys[valid.reshape(-1)], lengths[valid.reshape(-1)]
        originals = fitter.keys[candidates.reshape(-1)][:, None].expand(-1, variants, -1).reshape(-1, fitter.keys.shape[1])[valid.reshape(-1)]
        torch.testing.assert_close(keys[:, 0], originals[:, 0])
        ends = keys[torch.arange(len(keys)), lengths - 1]
        expected = fitter.keys[torch.arange(3), fitter.lengths - 1][:, None].expand(-1, variants).reshape(-1)[valid.reshape(-1)]
        torch.testing.assert_close(ends, expected)
        self.assertEqual(int(valid[-1].sum()), 1)

    def test_motor_fitter_has_finite_gradients(self):
        fitter = MotorFitter(["helical", "pineapples"], "cpu")
        parameters = torch.zeros(2, 70 + fitter.keys.shape[1] - 2, requires_grad=True)
        paths = fitter.render(fitter.keys, fitter.lengths, parameters)
        self.assertTrue(torch.isfinite(paths).all())
        (paths - torch.randn_like(paths)).square().mean().backward()
        self.assertTrue(torch.isfinite(parameters.grad).all())

    def test_larger_initialization_preserves_trained_predictions(self):
        torch.set_num_threads(2)
        original, expanded = build_encoder("large").eval(), build_encoder("giant").eval()
        with torch.no_grad():
            for block in original.blocks:
                block.layers[-1].weight.fill_(0.4)
        expand_encoder(expanded, original.state_dict())
        paths = torch.randn(2, 64, 2)
        with torch.no_grad():
            expected = original(encoder_features(original, paths))
            actual = expanded(encoder_features(expanded, paths))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        expanded.train()
        query = expanded(encoder_features(expanded, paths))
        (query * torch.randn_like(query)).sum().backward()
        added_gradients = expanded.blocks[0].layers[3].weight.grad[:, 192:]
        self.assertTrue(torch.isfinite(added_gradients).all())
        self.assertGreater(added_gradients.abs().sum().item(), 0)

    def test_cross_ranker_cannot_use_candidate_position(self):
        torch.set_num_threads(2)
        model = RouteCrossRanker(route_geometry=True).eval()
        # Use a nonzero score head so the test covers the learned correction.
        torch.nn.init.normal_(model.head[-1].weight, std=0.1)
        targets, lengths, _ = character_targets(["hello", "world", "helical"])
        sequence, raw, embedding = torch.randn(2, 128, 32), torch.randn(2, 8, 64), torch.randn(2, 128)
        candidates = torch.tensor([[0, 1, 2], [2, 0, 1]])
        similarity = torch.randn(2, 3)
        with torch.no_grad():
            expected = model(sequence, raw, embedding, candidates, similarity, targets, lengths)
            actual = model(sequence, raw, embedding, candidates.flip(1), similarity.flip(1), targets, lengths).flip(1)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_motor_initializer_preserves_start_and_learns_through_renderer(self):
        torch.set_num_threads(2)
        torch.manual_seed(11231)
        words = ["helical", "lunchbox", "alpacas"]
        fitter = MotorFitter(words, "cpu")
        sampler = MotorSampler(words, "cpu")
        paths, known = sampler.sample(torch.arange(3), strength=1.0, return_metadata=True)
        model = AmortizedMotor(fitter.keys.shape[1])
        keys, lengths = retained_training_keys(known, model.max_keys)
        parameters = model(paths, keys, lengths, fitter)
        torch.testing.assert_close(parameters, torch.zeros_like(parameters), atol=0, rtol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for step in range(2):
            parameters = model(paths, keys, lengths, fitter)
            loss = (fitter.render(keys, lengths, parameters) - paths).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if step == 1:
                self.assertGreater(model.view[1].weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_real_sampling_balances_words_without_discarding_recordings(self):
        torch.manual_seed(11237)
        labels = np.array([5] * 100 + [77] * 3 + [420], dtype=np.int64)
        groups = real_word_groups(labels, "cpu")
        self.assertEqual(set(groups[0].tolist()), set(range(len(labels))))
        indices = balanced_real_indices(groups, 9000).numpy()
        _, counts = np.unique(labels[indices], return_counts=True)
        np.testing.assert_allclose(counts / len(indices), np.full(3, 1 / 3), atol=0.025)

    def test_omission_mixture_normalizes_route_count(self):
        lengths = torch.tensor([[4, 5, 8]])
        errors = torch.full((1, 3, 7), 0.01)
        errors[0, 0, 1:] = torch.inf
        errors[0, 1, 4:] = torch.inf
        mixture = route_mixture_cost(errors, torch.zeros_like(errors), lengths, 0.02, 0.001)
        torch.testing.assert_close(mixture, torch.full((1, 3), 0.01), atol=1e-7, rtol=1e-6)
        errors[0, 0, 1:] = 0.0
        protected = route_mixture_cost(errors, torch.zeros_like(errors), lengths, 0.02, 0.001)
        self.assertAlmostEqual(float(protected[0, 0]), 0.01, places=6)

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
            subprocess.run([str(ROOT / "target/release/examples/memory_corpus"), "--output", str(output), "--samples", "16", "--seed", "800019", "--verification", str(verification), "--verification-limit", "16"], check=True, capture_output=True)
            rows, info = corpus(output)
            self.assertEqual(info["samples"], 16)
            references = json.loads(verification.read_text())
            self.assertEqual(len(references), 16)
            for row, case in zip(rows, references):
                np.testing.assert_allclose(row["points"], curve(case["points"]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
