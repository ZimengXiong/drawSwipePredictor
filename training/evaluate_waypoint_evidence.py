#!/usr/bin/env python3
"""Test point-only recovered corners on unchanged natural spelling candidates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from aligned_waypoint_model import AlignedWaypointDecoder
from prepare_waypoint_supervision import digest
from retrieval_model import build_encoder
from train_character import encode
from train_complete_geometry import metrics
from train_joint import load_checkpoint
from train_proxy import corpus
from waypoint_evidence import WaypointRouteBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--corners', type=Path, required=True)
    parser.add_argument('--training-data', type=Path, required=True)
    parser.add_argument('--fusion-report', type=Path, required=True)
    parser.add_argument('--route-report', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--samples', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=900977)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.scores.npz').exists():
        raise ValueError('use a new report and cache path')
    torch.set_num_threads(4)
    saved, context_hash = load_checkpoint(args.checkpoint)
    corner, corner_hash = load_checkpoint(args.corners)
    rows, info = corpus(args.data)
    fusion, route = [json.loads(path.read_text()) for path in [args.fusion_report, args.route_report]]
    if (saved['metadata']['test_data_used'] or corner['metadata']['test_data_used']
            or corner['metadata']['encoder_sha256'] != context_hash
            or saved['words'] != corner['words'] or len(saved['words']) != 60000
            or corner['metadata']['model_type'] != 'aligned_waypoints_v2'
            or saved['metadata']['validation_corpus']['sha256'] != info['sha256']
            or not 0 < args.samples <= len(rows)):
        raise ValueError('corner context, complete vocabulary, or declared development differs')
    for report in [fusion, route]:
        if (report['test_data_used'] or report['samples'] != len(rows)
                or report['full_vocabulary'] != 60000 or report['development']['sha256'] != info['sha256']):
            raise ValueError('cached baseline requires full declared development')
    if (not fusion['baseline_full_counts_verified'] or not fusion['inputs_unchanged_through_scoring']
            or fusion['route_report_sha256'] != digest(args.route_report)):
        raise ValueError('baseline integrity or candidate source differs')
    training = corner['metadata']['data']
    if digest(args.training_data) != training['data_sha256']:
        raise ValueError('training count-prior source differs')
    bound = {str(path): digest(path) for path in [args.checkpoint, args.corners,
             args.training_data, args.fusion_report, args.route_report]}
    sources = {name: digest(Path(name)) for name in ['evaluate_waypoint_evidence.py',
               'waypoint_evidence.py', 'waypoint_model.py', 'aligned_waypoint_model.py',
               'retrieval_model.py', 'large_encoder.py', 'visited_route_model.py']}
    for path, report in [(args.fusion_report, fusion), (args.route_report, route)]:
        cache = path.with_suffix('.scores.npz')
        if digest(cache) != report['score_cache_sha256']:
            raise ValueError('cached candidate evidence differs')
        bound[str(cache)] = report['score_cache_sha256']
    with np.load(args.fusion_report.with_suffix('.scores.npz'), allow_pickle=False) as data:
        spelling, candidates, selected = [data[name] for name in ['spelling_costs', 'candidates', 'selected_rows']]
    with np.load(args.route_report.with_suffix('.scores.npz'), allow_pickle=False) as data:
        if not np.array_equal(candidates, data['candidates']) or not np.array_equal(selected, data['selected_rows']):
            raise ValueError('natural candidates or row identities differ')
        similarity = data['similarities']
    if (candidates.shape != (12000, 128) or not np.array_equal(selected, np.arange(12000))
            or np.any((candidates < 0) | (candidates >= 60000))
            or not np.isfinite(similarity).all() or not np.isfinite(spelling).all()):
        raise ValueError('expected complete finite baseline evidence')
    ids = torch.tensor(candidates, dtype=torch.long)
    sizes = torch.tensor([len(word)+1 for word in saved['words']])[ids].float()
    fixed = fusion['fixed_spelling_settings']
    baseline = -torch.tensor(spelling)/sizes.pow(fixed['length_power'])+32*fixed['proxy_weight']*torch.tensor(similarity)
    labels = torch.tensor(np.array(rows['label'], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows['difficulty'], dtype=np.int64))
    baseline_metrics = metrics(baseline, ids, labels, difficulty)
    for name, values in baseline_metrics.items():
        for key in ['correct_top1', 'correct_top5']:
            if values[key] != fusion['trials'][0]['validation'][name][key]:
                raise ValueError('full zero-corner baseline differs')
    with np.load(args.training_data, allow_pickle=False) as data:
        frequencies = np.bincount(data['counts'][training['internal_holdout']:], minlength=33)[2:33]
    if np.any(frequencies == 0):
        raise ValueError('all count hypotheses require training support')
    prior = torch.tensor(np.log(frequencies/frequencies.sum()), device='mps', dtype=torch.float32)
    encoder = build_encoder(saved['metadata']['arguments']['architecture']).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(saved['model'])
    config = corner['metadata']['arguments']
    model = AlignedWaypointDecoder(getattr(encoder, 'input_channels', 8), config['width'], config['layers']).to('mps').eval().requires_grad_(False)
    model.load_state_dict(corner['model'])
    bank = WaypointRouteBank(saved['words'], 'mps')
    selected = np.arange(len(rows)) if args.samples == len(rows) else np.sort(np.random.default_rng(args.seed).choice(len(rows), args.samples, replace=False))
    costs, began = [], time.monotonic()
    with torch.no_grad():
        for start in range(0, len(selected), 16):
            part = selected[start:start+16]
            points = torch.tensor(np.array(rows['points'][part]), device='mps')
            prediction = model.infer(model.encode_memory(*encode(encoder, points)), points)
            costs.append(bank.costs(prediction, ids[part].to('mps'), prior).cpu())
            if start % 256 == 0:
                print(json.dumps({'finished': start+len(part), 'total': len(selected),
                                 'seconds': time.monotonic()-began}), flush=True)
    costs = torch.cat(costs)
    if not torch.isfinite(costs).all():
        raise ValueError('nonfinite corner evidence')
    trials = [{'corner_weight': weight, 'validation': metrics(baseline[selected]-weight*costs,
              ids[selected], labels[selected], difficulty[selected])}
              for weight in [0., 0.1, 0.25, 0.5, 1., 2., 4.]]
    if any(digest(Path(path)) != value for path, value in bound.items()) or any(digest(Path(name)) != value for name, value in sources.items()):
        raise ValueError('inputs or implementation changed while scoring')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.with_suffix('.scores.npz')
    np.savez(cache, corner_costs=costs.numpy(), candidates=candidates[selected], selected_rows=selected)
    report = {'test_data_used': False, 'development': info, 'samples': len(selected),
              'full_development': len(selected) == len(rows), 'full_vocabulary': 60000,
              'encoder_sha256': context_hash, 'corner_sha256': corner_hash, 'corner_epoch': corner['epoch'],
              'source_sha256': sources, 'inputs_sha256': bound, 'score_cache_sha256': digest(cache),
              'baseline_full_counts_verified': True, 'inputs_unchanged_through_scoring': True,
              'training_count_prior': prior.exp().cpu().tolist(), 'omission_prior': 0.2835,
              'fixed_spelling_settings': fixed, 'selected': max(trials, key=lambda r: r['validation']['memory']['correct_top1']),
              'trials': trials, 'native_supported': False,
              'limits': 'Every selected row; same natural128 candidates retrieved from all60000 words. Missing targets remain misses. All31 counts inferred from points; no true query count, corner label, target, writer, or difficulty enters prediction. Count prior uses training rows only. No benchmark or confirmation evaluation. Subset diagnostics are not full-development accuracy.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
