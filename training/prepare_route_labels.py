#!/usr/bin/env python3
"""Sample disjoint training writers and export their visited-key supervision."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from prepare_complete_geometry import validate_roles
from train_joint import load_checkpoint
from train_proxy import corpus


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train', type=Path, action='append', required=True)
    parser.add_argument('--cli', type=Path, default=Path('../target/release/examples/label_memory_routes'))
    parser.add_argument('--samples', type=int, default=260000)
    parser.add_argument('--held-out', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=900773)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.held_out < args.samples:
        raise ValueError('use a new output path and a nonempty internal holdout')
    saved, checkpoint_hash = load_checkpoint(args.checkpoint)
    words, metadata = saved['words'], saved['metadata']
    if metadata['test_data_used'] or len(words) != 60000:
        raise ValueError('expected a full-vocabulary training source')
    shards = [corpus(path) for path in args.train]
    validate_roles([info for _, info in shards], metadata['validation_corpus'], 'train', args.samples)
    edges = np.cumsum([len(rows) for rows, _ in shards])
    selected = np.random.default_rng(args.seed).choice(int(edges[-1]), args.samples, replace=False)
    ids = np.searchsorted(edges, selected, side='right')
    starts = np.r_[0, edges[:-1]]
    points = np.empty((args.samples, 64, 2), dtype=np.float32)
    labels = np.empty(args.samples, dtype=np.int64)
    difficulty = np.empty(args.samples, dtype=np.int64)
    writers = np.empty(args.samples, dtype=np.uint64)
    for index, (rows, _) in enumerate(shards):
        mask = ids == index
        chosen = rows[selected[mask] - starts[index]]
        points[mask], labels[mask] = chosen['points'], chosen['label']
        difficulty[mask], writers[mask] = chosen['difficulty'], chosen['writer']
    assert len(np.unique(writers)) == args.samples
    cli_hash = digest(args.cli)
    routes = np.zeros((args.samples, 64), dtype=np.int64)
    for start in range(0, args.samples, 4096):
        stop = min(start + 4096, args.samples)
        inputs = ''.join(json.dumps({'word': words[labels[i]], 'writer': int(writers[i]),
                                    'difficulty': ['clean', 'loose', 'memory'][difficulty[i]]}) + '\n'
                         for i in range(start, stop))
        result = subprocess.run([str(args.cli.resolve())], input=inputs, capture_output=True, text=True, check=True)
        keys = [json.loads(line)['visited_keys'] for line in result.stdout.splitlines()]
        if len(keys) != stop - start:
            raise ValueError('incomplete route labels')
        for i, route in enumerate(keys, start):
            word = words[labels[i]]
            collapsed = ''.join(c for j, c in enumerate(word) if j == 0 or c != word[j - 1])
            variants = {collapsed, *(collapsed[:j] + collapsed[j+1:] for j in range(1, len(collapsed)-1))}
            if route not in variants or (difficulty[i] == 0 and route != collapsed):
                raise ValueError('invalid retained route')
            tokens = [ord(c) - 96 for c in route] + [28]
            if len(tokens) > 64:
                raise ValueError('route exceeds token positions')
            routes[i, :len(tokens)] = tokens
    if digest(args.cli) != cli_hash:
        raise ValueError('label executable changed during generation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, points=points, routes=routes)
    report = {'test_data_used': False, 'split': 'train', 'samples': args.samples,
              'internal_holdout': args.held_out, 'checkpoint_sha256': checkpoint_hash,
              'seed': args.seed, 'corpora': [info for _, info in shards],
              'writer_selection_sha256': hashlib.sha256(writers.tobytes()).hexdigest(),
              'cli_sha256': cli_hash, 'data_sha256': digest(args.output),
              'source_sha256': digest(Path(__file__)),
              'query_inputs': 'Normalized points only; route tokens are loss targets. No word IDs, writer IDs, or difficulty stored as model inputs.'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
