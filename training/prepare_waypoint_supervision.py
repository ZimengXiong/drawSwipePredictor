#!/usr/bin/env python3
"""Create point-only inputs and latent corner supervision on fresh training writers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from prepare_complete_geometry import validate_roles
from train_joint import load_checkpoint


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def normalize_targets(raw, targets):
    """Apply the observed stroke's centering and RMS scale to its corner targets."""
    raw, targets = np.asarray(raw, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError('expected a finite moving stroke')
    if targets.ndim != 2 or targets.shape[1] != 2 or not 2 <= len(targets) <= 32 or not np.isfinite(targets).all():
        raise ValueError('expected two to thirty-two finite waypoint targets')
    origin = raw[0].copy()
    relative = raw - origin
    distances = np.linalg.norm(np.diff(relative, axis=0), axis=1)
    relative = relative[np.r_[True, distances > 1e-10]]
    if len(relative) < 2:
        raise ValueError('stationary stroke')
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(relative, axis=0), axis=1))]
    ticks = np.linspace(0, cumulative[-1], 64)
    sampled = np.stack([np.interp(ticks, cumulative, relative[:, axis]) for axis in range(2)], axis=1)
    center = sampled.mean(0)
    scale = max(np.sqrt(np.mean(np.sum((sampled-center)**2, axis=1))), 1e-10)
    return ((sampled-center)/scale).astype(np.float32), ((targets-origin-center)/scale).astype(np.float32)


def validate_arc_targets(values, count):
    values=np.asarray(values,dtype=np.float64)
    if (values.shape!=(count,) or not np.isfinite(values).all()
            or values[0]!=0 or values[-1]!=1 or np.any(np.diff(values)<0)):
        raise ValueError('arc targets must cover zero to one in chronological order')
    return values.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cli', type=Path, default=Path('../target/release/examples/label_memory_waypoints'))
    parser.add_argument('--samples', type=int, default=300000)
    parser.add_argument('--internal-holdout', type=int, default=10000)
    parser.add_argument('--writer-seed', type=int, default=720051)
    parser.add_argument('--seed', type=int, default=900919)
    parser.add_argument('--random-route-fraction', type=float, default=0.2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-progress', action='store_true')
    parser.add_argument('--reference-data', type=Path)
    args = parser.parse_args()
    if (args.output.exists() or not 0 < args.internal_holdout < args.samples < 1000003
            or not 0 <= args.random_route_fraction <= 1):
        raise ValueError('use a new output, a valid writer block, and a nonempty internal holdout')
    saved, context_hash = load_checkpoint(args.checkpoint)
    words, metadata = saved['words'], saved['metadata']
    if metadata['test_data_used'] or len(words) != 60000 or max(map(len, words)) > 32:
        raise ValueError('expected the complete declared vocabulary within thirty-two positions')
    start = args.writer_seed * 1000003
    info = {'samples': args.samples, 'vocabulary': 60000, 'seed': args.writer_seed,
            'writer_start': start, 'writer_end': start+args.samples, 'sha256': 'pending'}
    previous = [metadata['train_corpus'], *metadata.get('extra_train_corpora', [])]
    validate_roles([*previous, info], metadata['validation_corpus'], 'train', args.samples)
    rng = np.random.default_rng(args.seed)
    points = np.empty((args.samples, 64, 2), np.float32)
    targets = np.zeros((args.samples, 32, 2), np.float32)
    counts = np.empty(args.samples, np.int64)
    source = Path(__file__)
    source_hash, generator_hash = digest(source), digest(Path('../crates/swipe-engine/src/synthetic.rs'))
    random_count = 0
    progress=np.zeros((args.samples,32),np.float32) if args.with_progress else None
    reference_hash=digest(args.reference_data) if args.reference_data else None
    with tempfile.TemporaryDirectory(prefix='drawswipe-waypoint-labels-') as directory:
        cli = Path(directory)/'labels'
        shutil.copy2(args.cli, cli)
        cli_hash = digest(cli)
        for begin in range(0, args.samples, 1024):
            end = min(begin+1024, args.samples)
            requests = []
            for index in range(begin, end):
                if rng.random() < args.random_route_fraction:
                    length = int(rng.integers(2, 33))
                    letters = [int(rng.integers(26))]
                    for _ in range(1, length):
                        value = int(rng.integers(25))
                        letters.append(value + (value >= letters[-1]))
                    word = ''.join(chr(97+value) for value in letters)
                    random_count += 1
                else:
                    word = words[int(rng.integers(len(words)))]
                level = 0 if index % 10 == 0 else 1 if index % 10 < 3 else 2
                requests.append({'word': word, 'writer': start+index,
                                 'difficulty': ['clean','loose','memory'][level]})
            result = subprocess.run([str(cli)], input=''.join(json.dumps(row)+'\n' for row in requests),
                                    capture_output=True, text=True, check=True)
            records = [json.loads(line) for line in result.stdout.splitlines()]
            if len(records) != end-begin:
                raise ValueError('incomplete waypoint-label output')
            for index, record in enumerate(records, begin):
                if len(record['waypoints']) != len(record['visited_keys']):
                    raise ValueError('waypoint count differs from retained visits')
                path, corners = normalize_targets(record['points'], record['waypoints'])
                points[index], targets[index, :len(corners)], counts[index] = path, corners, len(corners)
                if progress is not None:
                    progress[index,:len(corners)]=validate_arc_targets(record['arc_positions'],len(corners))
            if end % 16384 == 0 or end == args.samples:
                print(json.dumps({'finished': end, 'total': args.samples}), flush=True)
        if digest(cli) != cli_hash or digest(source) != source_hash or digest(Path('../crates/swipe-engine/src/synthetic.rs')) != generator_hash:
            raise ValueError('label implementation changed during preparation')
    if args.reference_data:
        with np.load(args.reference_data,allow_pickle=False) as reference:
            for name,array in [('points',points),('waypoints',targets),('counts',counts)]:
                if not np.array_equal(reference[name],array):
                    raise ValueError('new supervision changed original '+name)
        if digest(args.reference_data)!=reference_hash:
            raise ValueError('reference data changed during preparation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.npz.tmp')
    with temporary.open('wb') as output:
        arrays={'points':points,'waypoints':targets,'counts':counts}
        if progress is not None:arrays['arc_positions']=progress
        np.savez(output,**arrays)
    temporary.replace(args.output)
    info['sha256'] = digest(args.output)
    report = {'test_data_used': False, 'split': 'train', 'samples': args.samples,
              'internal_holdout': args.internal_holdout, 'checkpoint_sha256': context_hash,
              'validation_corpus': metadata['validation_corpus'], 'corpora': [info],
              'target_schema': 'waypoints_arc_v2' if args.with_progress else 'waypoints_v1',
              'reference_data_sha256':reference_hash, 'base_arrays_identical_to_reference':bool(args.reference_data),
              'seed': args.seed, 'random_route_samples': random_count,
              'data_sha256': info['sha256'], 'cli_sha256': cli_hash,
              'source_sha256': source_hash, 'generator_sha256': generator_hash,
              'count_histogram': np.bincount(counts, minlength=33).tolist(),
              'query_inputs': 'Observed normalized points only. Counts and unrounded coordinates are supervised targets; word identities, writer IDs, and difficulty are absent from model inputs.',
              'limits': 'Training supervision only. Interior drift uses the fixed corner-midpoint convention; targets are latent controls, not observed pen samples. No recognition or benchmark accuracy claimed.'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
