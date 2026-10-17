#!/usr/bin/env python3
"""Compare fixed spelling evidence with cached point-only visited-route evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from autoregressive_model import load_decoder, word_tokens
from retrieval_model import build_encoder
from train_character import encode
from train_complete_geometry import metrics
from train_joint import load_checkpoint
from train_proxy import corpus

BLENDS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fusion_trials(base, route_costs, candidates, labels, difficulty):
    return [{'route_weight': weight,
             'validation': metrics(base - weight * route_costs, candidates, labels, difficulty)}
            for weight in BLENDS]


def validate_sources(route, spelling, development, context_hash, decoder_hash):
    for report in [route, spelling]:
        if report['test_data_used'] or report['development']['sha256'] != development['sha256'] or report['full_vocabulary'] != 60000:
            raise ValueError('fusion requires declared development and all 60000 words')
        if report['encoder_sha256'] != context_hash:
            raise ValueError('fusion contexts differ')
    if spelling['decoder_sha256'] != decoder_hash or spelling['samples'] != development['samples']:
        raise ValueError('spelling source must have a full-development selection')
    if route['word_components'] != spelling['word_retrieval'] or route['word_selection_sha256'] != spelling['retrieval_selection_sha256']:
        raise ValueError('route and spelling word mixtures differ')
    if spelling.get('joint_context') or spelling['selected']['proxy_weight'] is None:
        raise ValueError('expected a fixed independently bound spelling context')


@torch.no_grad()
def padded_candidate_cost(model, memory, candidates, inputs, targets):
    """Retain the historical full-padding arithmetic for exact baseline checks."""
    batch, count = candidates.shape
    ids = candidates.flatten()
    expanded = memory[:, None].expand(-1, count, -1, -1).reshape(-1, memory.shape[1], memory.shape[2])
    labels = targets[ids]
    logits = model(expanded, inputs[ids])
    nll = -logits.log_softmax(-1).gather(2, labels[..., None]).squeeze(-1)
    return (nll * (labels != 0)).sum(1).reshape(batch, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--spelling-selection', type=Path, required=True)
    parser.add_argument('--route-report', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--allow-development-subset', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source_hashes = {name: digest(Path(name)) for name in [Path(__file__).name,
                     'autoregressive_model.py', 'retrieval_model.py', 'large_encoder.py', 'train_character.py']}
    route_hash, selection_hash = digest(args.route_report), digest(args.spelling_selection)
    saved, context_hash = load_checkpoint(args.checkpoint)
    decoded, decoder_hash = load_checkpoint(args.decoder)
    route, spelling = [json.loads(path.read_text()) for path in [args.route_report, args.spelling_selection]]
    rows, info = corpus(args.data)
    validate_sources(route, spelling, info, context_hash, decoder_hash)
    if (saved['metadata']['test_data_used'] or decoded['metadata']['test_data_used']
            or saved['metadata']['validation_corpus']['sha256'] != info['sha256']
            or decoded['metadata']['encoder_sha256'] != context_hash
            or saved['words'] != decoded['words'] or len(saved['words']) != 60000):
        raise ValueError('checkpoint vocabulary or training provenance differs')
    cache_path = args.route_report.with_suffix('.scores.npz')
    if digest(cache_path) != route['score_cache_sha256']:
        raise ValueError('route score cache differs')
    with np.load(cache_path, allow_pickle=False) as data:
        candidates, similarity, route_costs, selected = [data[name] for name in ['candidates', 'similarities', 'route_costs', 'selected_rows']]
    count = route['samples']
    if (selected.shape != (count,) or candidates.shape != (count, 128)
            or similarity.shape != candidates.shape or route_costs.shape != candidates.shape
            or not np.isfinite(similarity).all() or not np.isfinite(route_costs).all()
            or np.any((candidates < 0) | (candidates >= 60000))
            or np.any((selected < 0) | (selected >= len(rows)))
            or not np.all(np.diff(selected) > 0)
            or hashlib.sha256(selected.tobytes()).hexdigest() != route['selected_rows_sha256']):
        raise ValueError('invalid candidates, scores, or selected rows')
    full = np.array_equal(selected, np.arange(len(rows)))
    if not full and not args.allow_development_subset:
        raise ValueError('full development is required unless a subset diagnostic is explicit')
    encoder = build_encoder(saved['metadata']['arguments']['architecture']).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(saved['model'])
    model = load_decoder(decoded['model'], getattr(encoder, 'input_channels', 8)).to('mps').eval().requires_grad_(False)
    inputs, targets, lengths = [part.to('mps') for part in word_tokens(saved['words'])]
    costs, began = [], time.monotonic()
    with torch.no_grad():
        for start in range(0, count, 192):
            paths = torch.tensor(np.array(rows['points'][selected[start:start+192]]), device='mps')
            context = encode(encoder, paths)
            for offset in range(0, len(paths), 4):
                local = [part[offset:offset+4] for part in context]
                memory = model.encode_memory(*local)
                ids = torch.tensor(candidates[start+offset:start+offset+len(local[0])], device='mps')
                costs.append(padded_candidate_cost(model, memory, ids, inputs, targets).cpu())
            print(json.dumps({'finished': min(start+192, count), 'total': count, 'seconds': time.monotonic()-began}), flush=True)
    costs = torch.cat(costs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_cache = args.output.with_suffix('.scores.npz')
    np.savez(output_cache, spelling_costs=costs.numpy(), selected_rows=selected, candidates=candidates)
    candidate_ids = torch.tensor(candidates, dtype=torch.long)
    sizes = lengths.cpu()[candidate_ids].float()
    fixed = spelling['selected']
    base = -costs / sizes.pow(fixed['length_power']) + 32 * fixed['proxy_weight'] * torch.tensor(similarity)
    labels = torch.tensor(np.array(rows['label'][selected], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows['difficulty'][selected], dtype=np.int64))
    trials = fusion_trials(base, torch.tensor(route_costs), candidate_ids, labels, difficulty)
    if full:
        for name, values in trials[0]['validation'].items():
            for key in ['top1', 'top5']:
                if round(values[key]*values['samples']) != round(fixed['validation'][name][key]*values['samples']):
                    raise ValueError('zero-route baseline differs from the full spelling selection; saved costs retained')
    unchanged = (digest(args.route_report) == route_hash and digest(args.spelling_selection) == selection_hash
                 and digest(args.checkpoint) == context_hash and digest(args.decoder) == decoder_hash
                 and digest(cache_path) == route['score_cache_sha256']
                 and all(digest(Path(name)) == value for name, value in source_hashes.items()))
    if not unchanged:
        raise ValueError('source, checkpoint, selection, or cached evidence changed during inference')
    report = {'test_data_used': False, 'development': info, 'samples': count, 'full_development': full,
              'full_vocabulary': 60000, 'encoder_sha256': context_hash, 'spelling_decoder_sha256': decoder_hash,
              'route_decoder_sha256': route['decoder_sha256'], 'route_report_sha256': route_hash,
              'spelling_selection_sha256': selection_hash, 'score_cache_sha256': digest(output_cache),
              'baseline_full_counts_verified': full, 'fixed_spelling_settings': {key: fixed[key] for key in ['length_power', 'proxy_weight']},
              'selected': max(trials, key=lambda trial: trial['validation']['memory']['correct_top1']), 'trials': trials,
              'native_supported': False, 'source_sha256': source_hashes, 'inputs_unchanged_through_scoring': True,
              'limits': 'Development evidence fusion only. Same natural128 candidates and complete words; no target insertion or route oracle. Subset diagnostics are not full-development or benchmark accuracy.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
