"""Verify preserved spelling likelihoods and trainable widened parameters."""
import unittest
import struct

import torch
from torch.nn import functional as F

from autoregressive_model import SpellingDecoder, load_decoder
from widen_spelling import widen_decoder
from export_spelling import spelling_binary


class SpellingWidthTests(unittest.TestCase):
    def test_native_header_retains_width_and_every_parameter(self):
        model = SpellingDecoder(width=384, layers=1)
        binary = spelling_binary(model, 11, 12, 13, 8, 2.0, 0.7)
        self.assertEqual(binary[:8], b'SWPSP003')
        self.assertEqual(struct.unpack_from('<II', binary, 44), (1, 384))
        self.assertEqual(len(binary), 52 + 4 * sum(p.numel() for p in model.parameters()))

    def test_wider_deeper_decoder_preserves_scores_and_can_learn(self):
        torch.manual_seed(91)
        old = SpellingDecoder(width=48, layers=1).eval()
        new = SpellingDecoder(width=96, layers=3).eval()
        widen_decoder(new, old.state_dict())
        sequence, raw, embedding = torch.randn(2, 128, 32), torch.randn(2, 8, 64), torch.randn(2, 128)
        inputs = torch.tensor([[27, 3, 1, 20], [27, 4, 15, 7]])
        first = old(old.encode_memory(sequence, raw, embedding), inputs)
        second = new(new.encode_memory(sequence, raw, embedding), inputs)
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
        F.cross_entropy(second.flatten(0, 1), torch.tensor([3, 1, 20, 28, 4, 15, 7, 28])).backward()
        self.assertGreater(float(new.decoder.layers[-1].linear2.weight.grad.norm()), 0)
        torch.optim.AdamW(new.parameters(), lr=0.001).step()
        restored = load_decoder(new.state_dict(), 8).eval()
        torch.testing.assert_close(restored(restored.encode_memory(sequence, raw, embedding), inputs),
                                   new(new.encode_memory(sequence, raw, embedding), inputs))
        with self.assertRaisesRegex(ValueError, 'larger multiple'):
            widen_decoder(old, new.state_dict())


if __name__ == '__main__':
    unittest.main()
