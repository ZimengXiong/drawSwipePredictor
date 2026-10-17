"""Load bound word experts and evaluate natural candidate compatibility scores."""
import json

import numpy as np
import torch

from evaluate_spelling import selected_word_weights
from retrieval_model import build_encoder, encoder_features
from train_character import encode
from train_complete_geometry import metrics
from train_joint import load_checkpoint
from train_proxy import proxy_scores


@torch.no_grad()
def retrieval_scores(experts, points):
    return sum(weight*proxy_scores(model(encoder_features(model, points)), proxies)
               for model, proxies, weight in experts)


def load_sources(context_path, word_paths, selection_path, development):
    context, context_hash = load_checkpoint(context_path)
    words = context['words']
    records = [load_checkpoint(path) for path in word_paths]
    selection = json.loads(selection_path.read_text())
    weights = selected_word_weights(selection, [key for _, key in records], development['sha256'], 60000)
    for saved in [context, *[record for record, _ in records]]:
        if (saved['metadata']['test_data_used'] or saved['words'] != words or len(words) != 60000
                or saved['metadata']['validation_corpus']['sha256'] != development['sha256']):
            raise ValueError('candidate model context, vocabulary, or development differs')
    architecture = context['metadata']['arguments']['architecture']
    encoder = build_encoder(architecture).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(context['model'])
    experts, sources = [], []
    for (saved, key), weight in zip(records, weights):
        architecture = saved['metadata']['arguments']['architecture']
        model = build_encoder(architecture).to('mps').eval().requires_grad_(False)
        model.load_state_dict(saved['model'])
        experts.append((model, saved['proxies'].to('mps'), weight))
        sources.append({'sha256': key, 'weight': weight})
    return encoder, experts, context, context_hash, sources


@torch.no_grad()
def validation_cache(encoder, experts, rows, selected):
    cache = {name: [] for name in ['sequence', 'raw', 'embedding', 'similarity', 'candidates']}
    for start in range(0, len(selected), 192):
        points = torch.tensor(np.array(rows['points'][selected[start:start+192]]), device='mps')
        context = encode(encoder, points)
        scores = retrieval_scores(experts, points)
        similarities, candidates = scores.topk(128, 1)
        for name, value in zip(cache, [*context, similarities, candidates]):
            cache[name].append(value)
    return {name: torch.cat(parts) for name, parts in cache.items()}


@torch.no_grad()
def evaluate(model, cache, tokens, lengths, geometry, labels, difficulty):
    model.eval()
    corrections = []
    for start in range(0, len(labels), 4):
        context = [cache[name][start:start+4] for name in ['sequence', 'raw', 'embedding']]
        memory = model.encode_memory(*context)
        corrections.append(model.candidate_scores(memory, cache['candidates'][start:start+4],
                                                  tokens, lengths, geometry).cpu())
    correction = torch.cat(corrections)
    if not torch.isfinite(correction).all():
        raise ValueError('nonfinite candidate compatibility scores')
    baseline = 128*cache['similarity'].cpu()
    ids = cache['candidates'].cpu()
    trials = [{'matcher_weight': weight, 'validation': metrics(baseline+weight*correction, ids, labels, difficulty)}
              for weight in [0., .1, .25, .5, 1., 2., 4.]]
    return max(trials, key=lambda row: row['validation']['memory']['correct_top1']), trials


def training_candidates(similarity, labels, count=16):
    """Mix hard negatives with random retrieved alternatives; shuffle positives.

    Training can supply its labeled positive. Evaluation uses natural candidates
    only. No candidate position or identity is passed as a model feature.
    """
    if similarity.ndim != 2 or labels.shape != (len(similarity),) or not 2 <= count <= 128:
        raise ValueError('expected complete class scores, training labels, and two to128 candidates')
    with torch.no_grad():
        classes = torch.arange(similarity.shape[1], device=similarity.device)
        alternatives = similarity.masked_fill(classes[None] == labels[:, None], -torch.inf).topk(128, 1).indices
        priority = torch.rand(alternatives.shape, device=similarity.device)
        priority[:, :min(8, count-1)] += 1
        negatives = alternatives.gather(1, priority.topk(count-1, 1).indices)
        candidates = torch.cat([labels[:, None], negatives], 1)
        permutation = torch.rand(candidates.shape, device=similarity.device).argsort(1)
        candidates = candidates.gather(1, permutation)
        targets = (permutation == 0).long().argmax(1)
        return candidates, similarity.gather(1, candidates), targets
