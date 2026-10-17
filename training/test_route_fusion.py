"""Check complete-word evidence fusion and source binding."""
import unittest
import torch

from combine_route_spelling import fusion_trials, validate_sources


class RouteFusionTests(unittest.TestCase):
    def test_zero_route_preserves_base_and_missing_targets_remain_misses(self):
        candidates = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        base = torch.tensor([[3., 2., 1.]] * 3)
        routes = torch.tensor([[10., 0., 0.]] * 3)
        trials = fusion_trials(base, routes, candidates, torch.tensor([0, 99, 7]), torch.tensor([2, 2, 2]))
        self.assertEqual(trials[0]['route_weight'], 0)
        self.assertEqual(trials[0]['validation']['memory']['correct_top1'], 1)
        self.assertTrue(all(row['validation']['memory']['samples'] == 3 for row in trials))
        self.assertTrue(all(row['validation']['memory']['correct_top5'] == 2 for row in trials))

    def test_different_word_components_or_context_cannot_be_combined(self):
        dev = {'sha256': 'dev', 'samples': 12000}
        shared = {'test_data_used': False, 'development': dev, 'full_vocabulary': 60000, 'encoder_sha256': 'encoder'}
        route = {**shared, 'word_components': [{'sha256': 'word', 'weight': 1}], 'word_selection_sha256': 'selection'}
        spelling = {**shared, 'decoder_sha256': 'decoder', 'samples': 12000,
                    'word_retrieval': route['word_components'], 'retrieval_selection_sha256': 'selection',
                    'selected': {'proxy_weight': 4}}
        validate_sources(route, spelling, dev, 'encoder', 'decoder')
        for altered in [{**route, 'word_components': []}, {**route, 'encoder_sha256': 'other'}, {**route, 'test_data_used': True}]:
            with self.assertRaises(ValueError):
                validate_sources(altered, spelling, dev, 'encoder', 'decoder')


if __name__ == '__main__':
    unittest.main()
