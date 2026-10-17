#!/usr/bin/env python3
"""Bind a trained shape correction to its unchanged complete native model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

from export_retrieval import fingerprint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    base, destination = args.base.resolve(), args.output.resolve()
    if destination.exists():
        raise ValueError('use a new export directory')
    selected = json.loads(args.selection.read_text())
    weights = json.loads(args.weights.read_text())
    manifest = json.loads((base / 'manifest.json').read_text())
    if (selected['test_data_used'] or selected['full_vocabulary'] != 60000
            or selected['development_samples'] != 12000
            or selected['base_manifest_sha256'] != digest(base / 'manifest.json')
            or selected['correction_sha256'] != digest(args.weights)):
        raise ValueError('correction selection or base binding differs')
    if ('geometry_ranker' in manifest or weights['feature_schema'] != 'complete_geometry_margin_v1'
            or weights['shortlist'] != 128 or weights['spelling_tag'] != manifest['spelling']['tag']):
        raise ValueError('unsupported correction schema or spelling binding')
    cases = json.loads((base / 'spelling-verification.json').read_text())
    command = [str(args.cli.resolve()), '--model', str(base), '--complete-features', '--candidates', '128']
    inputs = ''.join(json.dumps({'points': case['points']}) + '\n' for case in cases)
    output = subprocess.run(command, input=inputs, capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in output.stdout.splitlines()]
    if len(rows) != len(cases):
        raise ValueError('incomplete native reference extraction')
    mean, scale = torch.tensor(weights['mean']), torch.tensor(weights['scale'])
    for case, row in zip(cases, rows):
        candidates = row['candidates']
        if len(candidates) != 128:
            raise ValueError('reference extraction must retain the natural 128 candidates')
        features = torch.tensor([candidate['features'] for candidate in candidates])
        if torch.any(features[:, 2] != 0):
            raise ValueError('unexpected popularity input')
        features[:, 2] = features[:, 4] - features[:, 4].min()
        values = ((features - mean) / scale).clamp(-12, 12)
        for index, layer in enumerate(weights['layers']):
            values = values @ torch.tensor(layer['weights']).T + torch.tensor(layer['bias'])
            if index + 1 != len(weights['layers']):
                values = values.relu()
        costs = features[:, 1] + weights['residual_weight'] * values[:, 0]
        order = costs.argsort(stable=True)[:5].tolist()
        case['predictions'] = [{'word': candidates[i]['word'], 'rating': float(costs[i])} for i in order]
    shutil.copytree(base, destination)
    shutil.copyfile(args.weights, destination / 'geometry-ranker.json')
    (destination / 'spelling-verification.json').write_text(json.dumps(cases, indent=2) + '\n')
    manifest['spelling']['verification_sha256'] = digest(destination / 'spelling-verification.json')
    manifest['geometry_ranker'] = {
        'sha256': digest(args.weights), 'tag': fingerprint(args.weights.read_bytes()),
        'selection_sha256': digest(args.selection),
        'base_manifest_sha256': digest(base / 'manifest.json'),
        'feature_schema': weights['feature_schema'],
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'model': str(destination), 'manifest_sha256': digest(destination / 'manifest.json'),
                      'canonical_cases': len(cases), 'test_data_used': False}))


if __name__ == '__main__':
    main()
