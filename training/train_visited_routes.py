#!/usr/bin/env python3
"""Learn visited-key likelihoods with an internal training-writer holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from autoregressive_model import SpellingDecoder, expand_decoder
from prepare_complete_geometry import validate_roles
from retrieval_model import build_encoder
from train_character import encode
from train_joint import load_checkpoint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_loss(model, context, targets):
    # Teacher forcing supplies route tokens only to the supervised loss.
    width = int((targets != 0).sum(1).max())
    targets = targets[:, :width]
    inputs = F.pad(targets[:, :-1], (1, 0), value=27)
    logits = model(model.encode_memory(*context), inputs)
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=0, reduction='sum'), (targets != 0).sum()


@torch.no_grad()
def heldout_loss(model, encoder, points, targets, count, batch):
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, count, batch):
        end = min(start + batch, count)
        paths = torch.tensor(points[start:end], device='mps')
        labels = torch.tensor(targets[start:end], device='mps')
        loss, size = token_loss(model, encode(encoder, paths), labels)
        total += float(loss)
        tokens += int(size)
    return total / tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--init-decoder', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, default=6, choices=[3, 6, 9, 12])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--seed', type=int, default=900787)
    args = parser.parse_args()
    if args.output.exists() or min(args.epochs, args.steps, args.batch) < 1:
        raise ValueError('use a new output directory and positive training counts')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved, encoder_hash = load_checkpoint(args.checkpoint)
    initial, decoder_hash = load_checkpoint(args.init_decoder)
    metadata = json.loads(args.data.with_suffix('.json').read_text())
    if (saved['metadata']['test_data_used'] or initial['metadata']['test_data_used']
            or metadata['test_data_used'] or metadata['split'] != 'train'
            or metadata['checkpoint_sha256'] != encoder_hash
            or metadata['data_sha256'] != digest(args.data)
            or initial['metadata']['encoder_sha256'] != encoder_hash
            or initial['words'] != saved['words'] or len(saved['words']) != 60000):
        raise ValueError('route training source or decoder binding differs')
    validate_roles(metadata['corpora'], saved['metadata']['validation_corpus'], 'train', metadata['samples'])
    with np.load(args.data, allow_pickle=False) as data:
        points, routes = data['points'], data['routes']
    heldout = metadata['internal_holdout']
    if (points.shape != (metadata['samples'], 64, 2) or routes.shape != (len(points), 64)
            or not np.isfinite(points).all() or np.any((routes < 0) | (routes > 28))
            or not np.all((routes == 28).sum(1) == 1) or np.any(routes == 27)
            or not 0 < heldout < len(points)):
        raise ValueError('invalid point inputs, route targets, or internal split')
    lengths = (routes != 0).sum(1)
    if np.any(routes[np.arange(len(routes)), lengths - 1] != 28):
        raise ValueError('route end tokens must precede padding')
    architecture = saved['metadata']['arguments'].get('architecture', 'small')
    encoder = build_encoder(architecture).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(saved['model'])
    model = SpellingDecoder(getattr(encoder, 'input_channels', 8), layers=args.layers).to('mps')
    expand_decoder(model, initial['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.000005)
    run = {'test_data_used': False, 'encoder_sha256': encoder_hash, 'decoder_initialization_sha256': decoder_hash,
           'training_cache': metadata, 'decoder_parameters': sum(p.numel() for p in model.parameters()),
           'selection': 'Internal disjoint training-writer token loss only; no development recognition score used',
           'objective': 'Autoregressive visited-key sequence including end token; omitted keys remain omitted',
           'query_inputs': 'Normalized points only',
           'source_sha256': {name: digest(Path(name)) for name in ['train_visited_routes.py', 'autoregressive_model.py', 'retrieval_model.py']},
           'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    args.output.mkdir(parents=True)
    (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    best = float('inf')
    start = time.monotonic()
    for epoch in range(args.epochs + 1):
        total, tokens = 0.0, 0
        if epoch:
            model.train()
            for _ in range(args.steps):
                ids = rng.integers(heldout, len(points), size=args.batch)
                paths = torch.tensor(points[ids], device='mps')
                targets = torch.tensor(routes[ids], device='mps')
                loss, size = token_loss(model, encode(encoder, paths), targets)
                optimizer.zero_grad(set_to_none=True)
                (loss / size).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach())
                tokens += int(size)
            schedule.step()
        validation = heldout_loss(model, encoder, points, routes, heldout, args.batch)
        report = {'epoch': epoch, 'train_token_loss': total / max(tokens, 1),
                  'internal_holdout_token_loss': validation, 'seconds': time.monotonic() - start,
                  'recognition_accuracy_measured': False}
        with (args.output / 'history.jsonl').open('a') as out:
            out.write(json.dumps(report) + '\n')
        if validation < best:
            best = validation
            checkpoint = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          'words': saved['words'], 'epoch': epoch, 'metadata': run}
            temporary = args.output / 'best.pt.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / 'best.pt')
            (args.output / 'best-metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
