#!/usr/bin/env python3
"""Train query-dependent word ensembles on disjoint synthetic writers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from adaptive_word_model import AdaptiveWordGate
from prepare_complete_geometry import validate_roles
from retrieval_model import build_encoder, encoder_features
from train_joint import load_checkpoint
from train_proxy import corpus, proxy_scores, sample_corpora


def expert_scores(experts, paths):
    with torch.no_grad():
        return torch.stack([proxy_scores(model(encoder_features(model, paths)), proxies)
                            for model, proxies in experts], dim=1)


@torch.no_grad()
def evaluate(gate, experts, rows):
    gate.eval()
    # Controls use the same full-vocabulary score matrices and every row.
    counts = torch.zeros(3, 3, 2, device='mps')
    weight_sums = torch.zeros(3, len(experts), device='mps')
    for start in range(0, len(rows), 128):
        part = rows[start:start+128]
        paths = torch.tensor(np.array(part['points']), device='mps')
        truth = torch.tensor(np.array(part['label'], dtype=np.int64), device='mps')
        levels = torch.tensor(np.array(part['difficulty'], dtype=np.int64), device='mps')
        scores = expert_scores(experts, paths)
        learned, weights = gate(paths, scores)
        # The final two components are the previously selected mature pair.
        for mode, values in enumerate([learned, scores.mean(1), scores[:, -2:].mean(1)]):
            ranked = values.topk(5, dim=1).indices
            hit = ranked == truth[:, None]
            hits = torch.stack([hit[:, 0], hit.any(1)], dim=1).float()
            for level in range(3):
                counts[mode, level] += (hits * (levels == level)[:, None]).sum(0)
        for level in range(3):
            weight_sums[level] += (weights * (levels == level)[:, None]).sum(0)
    result = {}
    for mode, name in enumerate(['adaptive', 'uniform', 'mature_pair']):
        result[name] = {}
        for level, difficulty in enumerate(['clean', 'loose', 'memory']):
            size = int((rows['difficulty'] == level).sum())
            one, five = [int(value) for value in counts[mode, level]]
            result[name][difficulty] = {'samples': size, 'correct_top1': one, 'correct_top5': five,
                                        'top1': one/size, 'top5': five/size}
    return result, weight_sums.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--train', type=Path, action='append', required=True)
    parser.add_argument('--data', type=Path, default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=900823)
    args = parser.parse_args()
    if args.output.exists() or len(args.checkpoint) != 4 or min(args.epochs, args.steps, args.batch) < 1:
        raise ValueError('use a new output directory, four distinct experts, and positive counts')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    records = [load_checkpoint(path) for path in args.checkpoint]
    if len({key for _, key in records}) != 4:
        raise ValueError('supply each frozen expert once')
    words = records[0][0]['words']
    rows, info = corpus(args.data)
    experts, sources = [], []
    for (record, key), path in zip(records, args.checkpoint):
        metadata = record['metadata']
        if metadata['test_data_used'] or metadata['validation_corpus']['sha256'] != info['sha256'] or record['words'] != words or len(words) != 60000:
            raise ValueError('expert vocabulary or declared development differs')
        architecture = metadata['arguments'].get('architecture', 'small')
        model = build_encoder(architecture).to('mps').eval().requires_grad_(False)
        model.load_state_dict(record['model'])
        experts.append((model, record['proxies'].to('mps')))
        sources.append({'path': str(path), 'sha256': key, 'architecture': architecture})
    shards = [corpus(path) for path in args.train]
    validate_roles([item for _, item in shards], info, 'train', 1)
    gate = AdaptiveWordGate(len(experts)).to('mps')
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.0003, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00001)
    metadata = {'test_data_used': False, 'development': info, 'full_vocabulary': len(words),
                'word_experts': sources, 'train_corpora': [item for _, item in shards],
                'parameters': sum(p.numel() for p in gate.parameters()),
                'objective': 'Full 60000-class cross entropy; frozen experts; learned convex query-dependent weights',
                'query_inputs': 'Stroke features and expert score statistics only; no target, writer, difficulty, word identity, or frequency',
                'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['train_adaptive_words.py', 'adaptive_word_model.py']},
                'arguments': {key: [str(p) for p in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    args.output.mkdir(parents=True)
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    best = -1
    began = time.monotonic()
    for epoch in range(args.epochs+1):
        loss_sum = 0.0
        if epoch:
            gate.train()
            for _ in range(args.steps):
                paths, labels = sample_corpora(shards, args.batch, rng, 'mps')
                scores = expert_scores(experts, paths)
                combined, weights = gate(paths, scores)
                loss = F.cross_entropy(32 * combined, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach())
            schedule.step()
        if epoch % 5 and epoch != args.epochs:
            continue
        validation, sums = evaluate(gate, experts, rows)
        report = {'epoch': epoch, 'loss': loss_sum/max(args.steps, 1), 'validation': validation,
                  'expert_weight_sums_by_difficulty': sums, 'seconds': time.monotonic()-began}
        with (args.output / 'history.jsonl').open('a') as output:
            output.write(json.dumps(report)+'\n')
        score = validation['adaptive']['memory']['correct_top1']
        if score > best:
            best = score
            checkpoint = {'model': {k: v.detach().cpu() for k, v in gate.state_dict().items()},
                          'words': words, 'epoch': epoch, 'metadata': metadata}
            temporary = args.output / 'best.pt.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / 'best.pt')
            (args.output / 'best-metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
