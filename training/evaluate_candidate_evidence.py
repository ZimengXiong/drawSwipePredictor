#!/usr/bin/env python3
"""Compare new candidate evidence against a verified complete spelling baseline."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from autoregressive_model import load_decoder
from candidate_model import CandidateMatcher, candidate_geometry
from prepare_waypoint_supervision import digest
from retrieval_model import build_encoder
from train_character import encode
from train_complete_geometry import metrics
from train_joint import load_checkpoint
from train_proxy import corpus
from visited_route_model import RouteBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--kind', choices=['matcher', 'visited_routes'], required=True)
    parser.add_argument('--fusion-report', type=Path, required=True)
    parser.add_argument('--route-report', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--samples', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=900809)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.scores.npz').exists():
        raise ValueError('use new report and evidence-cache paths')
    torch.set_num_threads(4)
    context, context_hash = load_checkpoint(args.checkpoint)
    evidence, evidence_hash = load_checkpoint(args.evidence)
    rows, info = corpus(args.data)
    fusion, route = [json.loads(path.read_text()) for path in [args.fusion_report, args.route_report]]
    if (context['metadata']['test_data_used'] or evidence['metadata']['test_data_used']
            or evidence['metadata']['encoder_sha256'] != context_hash
            or context['words'] != evidence['words'] or len(context['words']) != 60000
            or context['metadata']['validation_corpus']['sha256'] != info['sha256']
            or not 0 < args.samples <= len(rows)):
        raise ValueError('context, evidence, or complete development differs')
    for report in [fusion, route]:
        if (report['test_data_used'] or report['development']['sha256'] != info['sha256']
                or report['full_vocabulary'] != 60000 or report['samples'] != len(rows)):
            raise ValueError('baseline must keep all declared development rows and words')
    if (not fusion['baseline_full_counts_verified'] or not fusion['inputs_unchanged_through_scoring']
            or fusion['route_report_sha256'] != digest(args.route_report)
            or fusion['encoder_sha256'] != context_hash or route['encoder_sha256'] != context_hash):
        raise ValueError('cached baseline binding differs')
    bound = {str(path): digest(path) for path in [args.checkpoint, args.evidence,
             args.fusion_report, args.route_report, args.data]}
    for path, report in [(args.fusion_report, fusion), (args.route_report, route)]:
        cache = path.with_suffix('.scores.npz')
        if digest(cache) != report['score_cache_sha256']:
            raise ValueError('baseline cache bytes differ')
        bound[str(cache)] = report['score_cache_sha256']
    with np.load(args.fusion_report.with_suffix('.scores.npz'), allow_pickle=False) as data:
        costs, candidates, selected = [data[name] for name in ['spelling_costs', 'candidates', 'selected_rows']]
    with np.load(args.route_report.with_suffix('.scores.npz'), allow_pickle=False) as data:
        if not np.array_equal(candidates, data['candidates']) or not np.array_equal(selected, data['selected_rows']):
            raise ValueError('baseline candidate lists or query rows differ')
        similarities = data['similarities']
    if (candidates.shape != (12000, 128) or costs.shape != candidates.shape
            or similarities.shape != candidates.shape or not np.array_equal(selected, np.arange(12000))
            or np.any((candidates < 0) | (candidates >= 60000))
            or not np.isfinite(costs).all() or not np.isfinite(similarities).all()):
        raise ValueError('invalid complete natural candidate cache')
    ids = torch.tensor(candidates, dtype=torch.long)
    lengths = torch.tensor([len(word)+1 for word in context['words']])[ids].float()
    fixed = fusion['fixed_spelling_settings']
    baseline = -torch.tensor(costs)/lengths.pow(fixed['length_power'])+32*fixed['proxy_weight']*torch.tensor(similarities)
    labels = torch.tensor(np.array(rows['label'], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows['difficulty'], dtype=np.int64))
    control = metrics(baseline, ids, labels, difficulty)
    for name, values in control.items():
        for key in ['correct_top1', 'correct_top5']:
            if values[key] != fusion['trials'][0]['validation'][name][key]:
                raise ValueError('zero-evidence baseline does not reproduce full counts')
    source_names = ['evaluate_candidate_evidence.py', 'candidate_model.py', 'autoregressive_model.py',
                    'visited_route_model.py', 'retrieval_model.py', 'large_encoder.py', 'train_character.py']
    sources = {name: digest(Path(name)) for name in source_names}
    encoder = build_encoder(context['metadata']['arguments']['architecture']).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(context['model'])
    raw_channels = getattr(encoder, 'input_channels', 8)
    if args.kind == 'matcher':
        if evidence['metadata'].get('model_type') != 'candidate_matcher_v1' or evidence['metadata']['word_experts'] != route['word_components']:
            raise ValueError('candidate matcher type or frozen word selection differs')
        config = evidence['metadata']['arguments']
        model = CandidateMatcher(raw_channels, config['width'], config['layers'])
        model.load_state_dict(evidence['model'])
        tokens, sizes, geometry = [part.to('mps') for part in candidate_geometry(context['words'])]
    else:
        if evidence['metadata'].get('objective') != 'Autoregressive visited-key sequence including end token; omitted keys remain omitted':
            raise ValueError('expected actual visited-route training supervision')
        model = load_decoder(evidence['model'], raw_channels)
        bank = RouteBank(context['words'], 'mps')
    model = model.to('mps').eval().requires_grad_(False)
    selected = np.arange(len(rows)) if args.samples == len(rows) else np.sort(np.random.default_rng(args.seed).choice(len(rows), args.samples, replace=False))
    corrections, began = [], time.monotonic()
    with torch.no_grad():
        for start in range(0, len(selected), 16):
            part = selected[start:start+16]
            points = torch.tensor(np.array(rows['points'][part]), device='mps')
            memory = model.encode_memory(*encode(encoder, points))
            candidates_part = ids[part].to('mps')
            values = (model.candidate_scores(memory, candidates_part, tokens, sizes, geometry)
                      if args.kind == 'matcher' else -bank.costs_batch(model, memory, candidates_part))
            corrections.append(values.cpu())
            if start % 256 == 0:
                print(json.dumps({'finished': start+len(part), 'total': len(selected),
                                 'seconds': time.monotonic()-began}), flush=True)
    corrections = torch.cat(corrections)
    if not torch.isfinite(corrections).all():
        raise ValueError('nonfinite evidence; every query must remain scored')
    trials = [{'evidence_weight': weight, 'validation': metrics(baseline[selected]+weight*corrections,
              ids[selected], labels[selected], difficulty[selected])}
              for weight in [0., .1, .25, .5, 1., 2., 4.]]
    if any(digest(Path(path)) != value for path, value in bound.items()) or any(digest(Path(name)) != value for name, value in sources.items()):
        raise ValueError('model, source, candidates, or input data changed during scoring')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.with_suffix('.scores.npz')
    np.savez(cache, corrections=corrections.numpy(), candidates=candidates[selected], selected_rows=selected)
    report = {'test_data_used': False, 'kind': args.kind, 'development': info, 'samples': len(selected),
              'full_development': len(selected) == len(rows), 'full_vocabulary': 60000,
              'encoder_sha256': context_hash, 'evidence_sha256': evidence_hash, 'evidence_epoch': evidence['epoch'],
              'source_sha256': sources, 'inputs_sha256': bound, 'score_cache_sha256': digest(cache),
              'fixed_spelling_settings': fixed, 'baseline_full_counts_verified': True,
              'inputs_unchanged_through_scoring': True, 'trials': trials,
              'selected': max(trials, key=lambda row: row['validation']['memory']['correct_top1']),
              'native_supported': False,
              'limits': 'Fixed complete spelling baseline and identical natural128 lists from all60000 words. Every selected row counts, including missing targets. Prediction receives points and public candidate spellings only. No true target, visited count, writer identity, or difficulty enters inference. No benchmark or confirmation used. Subset scores are not full-development accuracy.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
