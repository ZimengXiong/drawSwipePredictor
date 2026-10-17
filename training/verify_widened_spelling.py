#!/usr/bin/env python3
"""Check full-size spelling expansion on canonical implementation references."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from autoregressive_model import SpellingDecoder, load_decoder, word_tokens
from retrieval_model import build_encoder, curve
from train_character import encode
from train_joint import load_checkpoint
from widen_spelling import widen_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved, encoder_hash = load_checkpoint(args.checkpoint)
    source, decoder_hash = load_checkpoint(args.decoder)
    if source['metadata']['encoder_sha256'] != encoder_hash or source['words'] != saved['words']:
        raise ValueError('decoder context differs')
    encoder = build_encoder(saved['metadata']['arguments']['architecture']).to('mps').eval()
    encoder.load_state_dict(saved['model'])
    old = load_decoder(source['model'], 8).to('mps').eval()
    new = SpellingDecoder(8, width=384, layers=12).to('mps').eval()
    widen_decoder(new, source['model'])
    cases = json.loads(args.references.read_text())
    reports = []
    with torch.no_grad():
        for case in cases:
            paths = torch.tensor(curve(case['points'])[None], device='mps')
            context = encode(encoder, paths)
            inputs, targets, _ = [part.to('mps') for part in word_tokens(case['words'])]
            costs = []
            for model in [old, new]:
                memory = model.encode_memory(*context).expand(len(inputs), -1, -1)
                logits = model(memory, inputs)
                nll = -logits.log_softmax(-1).gather(2, targets[..., None]).squeeze(-1)
                costs.append((nll * (targets != 0)).sum(1))
            error = float((costs[0]-costs[1]).abs().max())
            if error > 0.0002:
                raise ValueError(f'widened cost differs by {error}')
            reports.append({'case': case['word'], 'maximum_cost_error': error})
    report = {'test_data_used': False, 'encoder_sha256': encoder_hash, 'decoder_sha256': decoder_hash,
              'reference_sha256': hashlib.sha256(args.references.read_bytes()).hexdigest(),
              'width': 384, 'layers': 12, 'parameters': sum(p.numel() for p in new.parameters()),
              'cases': reports, 'limits': 'Canonical Python initialization parity only; no training, native compatibility, or recognition improvement claimed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
