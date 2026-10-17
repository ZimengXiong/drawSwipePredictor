#!/usr/bin/env python3
"""Compare cached point-only evidence with complete and word-only controls."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from prepare_waypoint_supervision import digest
from train_complete_geometry import metrics
from train_proxy import corpus
from train_joint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fusion-report', type=Path, required=True)
    parser.add_argument('--route-report', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, action='append', required=True)
    parser.add_argument('--data', type=Path, default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= len(args.evidence) <= 2:
        raise ValueError('use a new report and one or two evidence sources')
    torch.set_num_threads(4)
    rows, info = corpus(args.data)
    fusion, route = [json.loads(path.read_text()) for path in [args.fusion_report, args.route_report]]
    reports = [json.loads(path.read_text()) for path in args.evidence]
    bound = {str(path): digest(path) for path in [args.fusion_report, args.route_report, args.data, *args.evidence]}
    for report in [fusion, route, *reports]:
        if (report['test_data_used'] or report['development']['sha256'] != info['sha256']
                or report['full_vocabulary'] != 60000 or report['samples'] != len(rows)):
            raise ValueError('require complete declared development and all60000 words')
    if (not fusion['baseline_full_counts_verified'] or not fusion['inputs_unchanged_through_scoring']
            or fusion['route_report_sha256'] != bound[str(args.route_report)]):
        raise ValueError('baseline sources differ')
    for path, report in zip([args.fusion_report, args.route_report, *args.evidence], [fusion, route, *reports]):
        cache = path.with_suffix('.scores.npz')
        if digest(cache) != report['score_cache_sha256']:
            raise ValueError('candidate cache bytes differ')
        bound[str(cache)] = report['score_cache_sha256']
    with np.load(args.fusion_report.with_suffix('.scores.npz'), allow_pickle=False) as data:
        spelling, ids, selected = [data[name] for name in ['spelling_costs', 'candidates', 'selected_rows']]
    if (ids.shape != (12000, 128) or not np.array_equal(selected, np.arange(12000))
            or np.any((ids < 0) | (ids >= 60000)) or spelling.shape != ids.shape
            or not np.isfinite(spelling).all()):
        raise ValueError('all natural lists and query rows must remain present')
    with np.load(args.route_report.with_suffix('.scores.npz'), allow_pickle=False) as data:
        if not np.array_equal(ids, data['candidates']) or not np.array_equal(selected, data['selected_rows']):
            raise ValueError('baseline candidate or query order differs')
        similarity = torch.tensor(data['similarities'])
    corrections, sources = [], []
    for path, report in zip(args.evidence, reports):
        if (not report['full_development'] or not report['baseline_full_counts_verified']
                or not report['inputs_unchanged_through_scoring']
                or bound[str(args.fusion_report)] not in report['inputs_sha256'].values()
                or bound[str(args.route_report)] not in report['inputs_sha256'].values()):
            raise ValueError('evidence must bind the same verified complete baseline')
        with np.load(path.with_suffix('.scores.npz'), allow_pickle=False) as data:
            if not np.array_equal(ids, data['candidates']) or not np.array_equal(selected, data['selected_rows']):
                raise ValueError('evidence must retain identical natural candidates and all rows')
            if 'corner_costs' in data:
                values = -torch.tensor(data['corner_costs'])
                kind = 'corners'
            else:
                values = torch.tensor(data['corrections'])
                kind = report['kind']
        if values.shape != ids.shape or not torch.isfinite(values).all():
            raise ValueError('invalid evidence values; no query may be discarded')
        corrections.append(values)
        sources.append({'path': str(path), 'sha256': bound[str(path)], 'kind': kind})
    ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor(np.array(rows['label'], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows['difficulty'], dtype=np.int64))
    # Both controls use exactly the same candidate lists. Candidate lengths are
    # public spellings, never the unknown query's intended length.
    context, context_hash = load_checkpoint(args.checkpoint)
    if (context_hash != fusion['encoder_sha256'] or context['metadata']['test_data_used']
            or len(context['words']) != 60000 or similarity.shape != ids.shape
            or not torch.isfinite(similarity).all()):
        raise ValueError('baseline vocabulary binding differs')
    bound[str(args.checkpoint)] = context_hash
    sizes = torch.tensor([len(word)+1 for word in context['words']])[ids].float()
    fixed = fusion['fixed_spelling_settings']
    bases = {'word': 128*similarity,
             'complete': -torch.tensor(spelling)/sizes.pow(fixed['length_power'])+32*fixed['proxy_weight']*similarity}
    for name, baseline in bases.items():
        expected = (route if name == 'word' else fusion)['trials'][0]['validation']
        actual = metrics(baseline, ids, labels, difficulty)
        for level, values in actual.items():
            for key in ['correct_top1', 'correct_top5']:
                if values[key] != expected[level][key]:
                    raise ValueError('cached control counts differ')
    trials = []
    for name, baseline in bases.items():
        for weights in itertools.product([0., .1, .25, .5, 1., 2., 4.], repeat=len(corrections)):
            scores = baseline+sum(weight*values for weight, values in zip(weights, corrections))
            trials.append({'base': name, 'weights': weights, 'validation': metrics(scores, ids, labels, difficulty)})
    if any(digest(Path(path)) != value for path, value in bound.items()):
        raise ValueError('inputs changed during cached comparison')
    report = {'test_data_used': False, 'development': info, 'samples': len(rows), 'full_vocabulary': 60000,
              'source_sha256': digest(Path(__file__)), 'inputs_sha256': bound, 'evidence': sources,
              'both_baseline_counts_verified': True, 'inputs_unchanged_through_scoring': True,
              'selected': max(trials, key=lambda row: row['validation']['memory']['correct_top1']),
              'selected_by_base': {name: max([trial for trial in trials if trial['base'] == name],
                                   key=lambda row: row['validation']['memory']['correct_top1']) for name in bases},
              'trials': trials, 'native_supported': False,
              'limits': 'Full separate development only. All rows and identical natural128 candidates retained. Cached scores originate from point-only inference; labels and difficulty are used only for selection and reporting. No target insertion, aliases, benchmark, or confirmation data.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['selected_by_base']), flush=True)


if __name__ == '__main__':
    main()
