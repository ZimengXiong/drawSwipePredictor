#!/usr/bin/env python3
"""Measure ambiguity in an idealized visited-key channel without reading strokes."""
import argparse
import hashlib
import json
from pathlib import Path

from visited_route_model import route_variants
import math


def route_oracle(words, omission_probability):
    maxima, canonical = {}, set()
    for word in words:
        rows = route_variants(word, omission_probability)
        canonical.add(rows[0][0])
        combined = {}
        for route, log_prior in rows:
            combined[route] = combined.get(route, 0.0) + math.exp(log_prior)
        for route, probability in combined.items():
            maxima[route] = max(maxima.get(route, 0.0), probability)
    return {'words': len(words), 'unique_canonical_routes': len(canonical),
            'canonical_identity_oracle_accuracy': len(canonical)/len(words),
            'visited_identity_oracle_accuracy': sum(maxima.values())/len(words),
            'possible_visited_routes': len(maxima)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary', type=Path, default=Path('../word_freq.txt'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    words = []
    for line in args.dictionary.read_text().splitlines():
        word = line.split('\t')[0]
        if len(word) >= 2 and word.isascii() and word.islower() and word.isalpha() and len(set(word)) > 1:
            words.append(word)
        if len(words) == 60000:
            break
    if len(words) != 60000:
        raise ValueError('expected the complete deployment vocabulary')
    # This is an analytic oracle for the training target policy, never an
    # inference filter. No recording, label, writer identity, or benchmark is read.
    eligible = [word for word in words if 4 <= len(word) <= 12]
    report = {'test_data_used': False, 'recordings_read': 0, 'deployment_vocabulary': len(words),
              'dictionary_sha256': hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
              'omission_probability': 0.35, **route_oracle(eligible, 0.35),
              'assumptions': 'Uniform training-policy target words; exact visited letter sequence supplied by an oracle; one optional uniformly selected interior omission for routes longer than four keys.',
              'limits': 'Analytic route-channel diagnostic only, not recognition accuracy or a proven bound for all continuous stroke evidence. The target-policy prior and oracle labels are never used by a deployed predictor.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
