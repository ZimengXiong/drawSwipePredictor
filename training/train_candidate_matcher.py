#!/usr/bin/env python3
"""Learn direct word-stroke compatibility on audited synthetic training writers."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from candidate_model import CandidateMatcher, candidate_geometry
from candidate_training import load_sources, retrieval_scores, validation_cache, evaluate, training_candidates
from prepare_complete_geometry import validate_roles
from prepare_waypoint_supervision import digest
from retrieval_model import augment, load_recordings, vocabulary
from train_character import encode
from train_joint import load_checkpoint
from train_proxy import corpus, sample_corpora, real_word_groups, balanced_real_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--init-decoder', type=Path, required=True)
    parser.add_argument('--word-checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--word-selection', type=Path, required=True)
    parser.add_argument('--train', type=Path, action='append', required=True)
    parser.add_argument('--data', type=Path, default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--real-train', type=Path, default=Path('data/futo-large/train.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--candidates', type=int, default=16)
    parser.add_argument('--width', type=int, default=384, choices=[192, 384])
    parser.add_argument('--layers', type=int, default=6, choices=[3, 6, 9, 12])
    parser.add_argument('--evaluate-every', type=int, default=5)
    parser.add_argument('--validation-samples', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=901003)
    args = parser.parse_args()
    if (args.output.exists() or min(args.epochs, args.steps, args.batch, args.evaluate_every) < 1
            or args.batch < 10 or not 2 <= args.candidates <= 128):
        raise ValueError('use a new output directory, positive counts, and a valid candidate batch')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows, info = corpus(args.data)
    if not 0 < args.validation_samples <= len(rows):
        raise ValueError('validation sample count exceeds declared development')
    encoder, experts, context, context_hash, sources = load_sources(args.checkpoint, args.word_checkpoint,
                                                                  args.word_selection, info)
    words = context['words']
    initial, initial_hash = load_checkpoint(args.init_decoder)
    if (initial['metadata']['test_data_used'] or initial['metadata']['encoder_sha256'] != context_hash
            or initial['words'] != words or 'context_model' in initial):
        raise ValueError('matcher initialization requires the original frozen context')
    shards = [corpus(path) for path in args.train]
    validate_roles([item for _, item in shards], info, 'train', 1)
    if shards[0][1]['sha256'] != context['metadata']['train_corpus']['sha256']:
        raise ValueError('retain the declared original training corpus')
    canonical_words, _, canonical = vocabulary(Path('../word_freq.txt'), 60000)
    if canonical_words != words:
        raise ValueError('canonical vocabulary order differs')
    canonical = torch.tensor(canonical, device='mps')
    real_hash = digest(args.real_train)
    real, real_labels = load_recordings(args.real_train, {word: i for i, word in enumerate(words)})
    groups = real_word_groups(real_labels, 'mps')
    real, real_labels = torch.tensor(real, device='mps'), torch.tensor(real_labels, device='mps')
    selected = np.sort(np.random.default_rng(900419).choice(len(rows), args.validation_samples, replace=False))
    cache = validation_cache(encoder, experts, rows, selected)
    labels = torch.tensor(np.array(rows['label'][selected], dtype=np.int64))
    difficulty = torch.tensor(np.array(rows['difficulty'][selected], dtype=np.int64))
    tokens, lengths, geometry = [part.to('mps') for part in candidate_geometry(words)]
    model = CandidateMatcher(getattr(encoder, 'input_channels', 8), args.width, args.layers)
    model.initialize(initial['model'], seed=args.seed+118)
    model = model.to('mps')
    output_parameters = list(model.compatibility.parameters())
    output_ids = {id(parameter) for parameter in output_parameters}
    backbone = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in output_ids]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 0.00003},
                                  {'params': output_parameters, 'lr': 0.0003}], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.000005)
    metadata = {'test_data_used': False, 'model_type': 'candidate_matcher_v1',
                'encoder_sha256': context_hash, 'decoder_initialization_sha256': initial_hash,
                'word_experts': sources, 'word_selection_sha256': digest(args.word_selection),
                'train_corpora': [item for _, item in shards], 'validation_corpus': info,
                'full_vocabulary': len(words), 'real_train_sha256': real_hash,
                'real_samples': len(real), 'real_sampling': 'Uniform observed training word types',
                'batch_composition': {'real': args.batch*15//100, 'canonical_all_words': args.batch//10,
                                      'synthetic': args.batch-args.batch*15//100-args.batch//10},
                'parameters': sum(parameter.numel() for parameter in model.parameters()),
                'objective': 'Direct candidate-list cross entropy over128*cosine plus learned compatibility',
                'query_inputs': 'Observed points and each public candidate spelling and keyboard geometry. No target, true visited count, writer, difficulty, popularity, or vocabulary rank enters prediction.',
                'selection': 'Separate fixed development subset; natural128 candidates from all60000 words. Full-development and native checks required before promotion.',
                'selected_rows_sha256': __import__('hashlib').sha256(selected.tobytes()).hexdigest(),
                'source_sha256': {name: digest(Path(name)) for name in ['candidate_model.py', 'candidate_training.py',
                                  'train_candidate_matcher.py', 'autoregressive_model.py', 'widen_spelling.py',
                                  'retrieval_model.py', 'train_character.py', 'train_proxy.py']},
                'arguments': {key: [str(p) for p in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    del initial, context
    args.output.mkdir(parents=True)
    (args.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    best = -1
    began = time.monotonic()
    for epoch in range(args.epochs+1):
        loss_sum = 0.
        if epoch:
            model.train()
            for _ in range(args.steps):
                composition = metadata['batch_composition']
                paths, target = sample_corpora(shards, composition['synthetic'], rng, 'mps')
                ri = balanced_real_indices(groups, composition['real'])
                ci = torch.randint(len(words), (composition['canonical_all_words'],), device='mps')
                paths = torch.cat([paths, augment(real[ri]), augment(canonical[ci])])
                target = torch.cat([target, real_labels[ri], ci])
                scores = retrieval_scores(experts, paths)
                candidates, prior, supervised = training_candidates(scores, target, args.candidates)
                memory = model.encode_memory(*encode(encoder, paths))
                corrections = model.candidate_scores(memory, candidates, tokens, lengths, geometry)
                loss = F.cross_entropy(128*prior+corrections, supervised)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite candidate compatibility loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_sum += float(loss.detach())
            scheduler.step()
        report = {'epoch': epoch, 'training_loss': loss_sum/max(args.steps, 1), 'seconds': time.monotonic()-began}
        improved = False
        if epoch == 0 or epoch == 1 or epoch % args.evaluate_every == 0:
            selected_trial, trials = evaluate(model, cache, tokens, lengths, geometry, labels, difficulty)
            score = selected_trial['validation']['memory']['correct_top1']
            improved = score > best
            best = max(best, score)
            report.update(selected=selected_trial, trials=trials, best_hard_hits=best, improved=improved)
        temporary = args.output/'latest.pt.tmp'
        checkpoint = {'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                      'words': words, 'epoch': epoch, 'metadata': metadata, 'selection': report.get('selected')}
        torch.save(checkpoint, temporary); temporary.replace(args.output/'latest.pt')
        if improved:
            temporary = args.output/'best.pt.tmp'
            torch.save(checkpoint, temporary); temporary.replace(args.output/'best.pt')
            (args.output/'best-metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        with (args.output/'history.jsonl').open('a') as output:
            output.write(json.dumps(report)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'trials'}), flush=True)


if __name__ == '__main__':
    main()
