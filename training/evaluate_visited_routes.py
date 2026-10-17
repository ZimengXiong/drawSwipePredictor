#!/usr/bin/env python3
"""Compare visited-route evidence using natural full-vocabulary candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from autoregressive_model import load_decoder
from evaluate_spelling import selected_word_weights
from retrieval_model import build_encoder
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores
from visited_route_model import RouteBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--word-checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--word-selection', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--samples', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=900809)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved, encoder_hash = load_checkpoint(args.checkpoint)
    decoded, decoder_hash = load_checkpoint(args.decoder)
    rows, info = corpus(args.data)
    if (saved['metadata']['test_data_used'] or decoded['metadata']['test_data_used']
            or saved['metadata']['validation_corpus']['sha256'] != info['sha256']
            or decoded['metadata']['encoder_sha256'] != encoder_hash
            or decoded['words'] != saved['words'] or len(saved['words']) != 60000
            or not 0 < args.samples <= len(rows)):
        raise ValueError('route context, vocabulary, or declared development differs')
    selection_bytes = args.word_selection.read_bytes()
    components = [load_checkpoint(path) for path in args.word_checkpoint]
    weights = selected_word_weights(json.loads(selection_bytes), [key for _, key in components], info['sha256'], 60000)
    encoder = build_encoder(saved['metadata']['arguments'].get('architecture', 'small')).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(saved['model'])
    word_models = []
    for (record, key), weight in zip(components, weights):
        if (record['words'] != saved['words'] or record['metadata']['test_data_used']
                or record['metadata']['validation_corpus']['sha256'] != info['sha256']):
            raise ValueError('word source differs from declared vocabulary or development')
        model = build_encoder(record['metadata']['arguments'].get('architecture', 'small')).to('mps').eval().requires_grad_(False)
        model.load_state_dict(record['model'])
        word_models.append((model, record['proxies'].to('mps'), weight))
    decoder = load_decoder(decoded['model'], getattr(encoder, 'input_channels', 8)).to('mps').eval().requires_grad_(False)
    bank = RouteBank(saved['words'], 'mps')
    selected = np.arange(len(rows)) if args.samples == len(rows) else np.sort(np.random.default_rng(args.seed).choice(len(rows), args.samples, replace=False))
    candidates, similarities, route_costs = [], [], []
    began = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(selected), 64):
            paths = torch.tensor(np.array(rows['points'][selected[start:start+64]]), device='mps')
            scores = sum(weight * proxy_scores(encode(model, paths)[2], proxies) for model, proxies, weight in word_models)
            similarity, ids = scores.topk(128, dim=1)
            memory = decoder.encode_memory(*encode(encoder, paths))
            costs = torch.stack([bank.costs(decoder, memory[i:i+1], ids[i]) for i in range(len(paths))])
            candidates.append(ids.cpu()); similarities.append(similarity.cpu()); route_costs.append(costs.cpu())
            print(json.dumps({'finished': min(start+64, len(selected)), 'total': len(selected), 'seconds': time.monotonic()-began}), flush=True)
    candidates, similarities, route_costs = [torch.cat(parts) for parts in [candidates, similarities, route_costs]]
    truth = torch.tensor(np.array(rows['label'][selected], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows['difficulty'][selected], dtype=np.int64))
    trials = []
    for weight in [None, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        scores = similarities if weight is None else -route_costs + weight * 32 * similarities
        order = scores.argsort(1, descending=True, stable=True)[:, :5]
        hits = candidates.gather(1, order) == truth[:, None]
        metrics = {}
        for level, name in enumerate(['clean', 'loose', 'memory']):
            mask = difficulty == level
            size = int(mask.sum())
            one, five = int(hits[mask, 0].sum()), int(hits[mask].any(1).sum())
            metrics[name] = {'samples': size, 'correct_top1': one, 'correct_top5': five, 'top1': one/max(size, 1), 'top5': five/max(size, 1)}
        trials.append({'proxy_weight': weight, 'validation': metrics})
    report = {'test_data_used': False, 'development': info, 'samples': len(selected), 'full_vocabulary': 60000,
              'encoder_sha256': encoder_hash, 'decoder_sha256': decoder_hash,
              'word_components': [{'sha256': key, 'weight': weight} for (_, key), weight in zip(components, weights)],
              'word_selection_sha256': hashlib.sha256(selection_bytes).hexdigest(),
              'selected_rows_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
              'omission_prior': 0.2835, 'selected': max(trials, key=lambda r: r['validation']['memory']['correct_top1']), 'trials': trials,
              'rules': 'Every selected row; full 60000-word retrieval; natural top128; exact complete word; no target insertion, route-count oracle, or visited-key labels at inference',
              'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['evaluate_visited_routes.py', 'visited_route_model.py']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
