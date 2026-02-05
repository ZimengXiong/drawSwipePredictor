#!/usr/bin/env python3
"""
Evaluate a trained swipe model on synthetic blind swipe data.

Usage:
    python evaluate_model.py --model swipe_model.onnx --vocab swipe_model_vocab.json --words word_freq.txt
"""

import argparse
import numpy as np
import json
import onnxruntime as ort
from train_model import (
    get_word_path, interpolate_path, resample_path, normalize_path,
    augment_for_blind_swipe, KEYBOARD_LAYOUT
)
from collections import defaultdict


def load_model(model_path: str):
    """Load ONNX model."""
    return ort.InferenceSession(model_path)


def predict(session, path: np.ndarray, vocabulary: list, top_k: int = 5) -> list:
    """Run prediction on a path."""
    # Ensure path is normalized
    resampled = resample_path(path, 32)
    normalized = normalize_path(resampled)
    features = normalized.flatten().astype(np.float32)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: features.reshape(1, -1)})[0]

    # Softmax
    logits = logits[0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    # Get top-k predictions
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(vocabulary[i], probs[i]) for i in top_indices]


def evaluate(
    session,
    vocabulary: list,
    test_words: list,
    samples_per_word: int = 50,
    severity: float = 1.0
):
    """Evaluate model accuracy."""
    # Build word to index mapping
    word_to_idx = {w: i for i, w in enumerate(vocabulary)}

    # Filter test words to only those in vocabulary
    test_words = [w for w in test_words if w in word_to_idx]

    total = 0
    correct_top1 = 0
    correct_top4 = 0
    correct_top10 = 0

    # Track per-word accuracy
    word_correct = defaultdict(int)
    word_total = defaultdict(int)

    print(f"Evaluating on {len(test_words)} words, {samples_per_word} samples each...")
    print(f"Augmentation severity: {severity}")
    print()

    for word in test_words:
        ideal_path = get_word_path(word)
        if len(ideal_path) < 2:
            continue

        ideal_path = interpolate_path(ideal_path)

        for _ in range(samples_per_word):
            # Augment for blind swipe
            augmented = augment_for_blind_swipe(ideal_path.copy(), severity)

            # Predict
            predictions = predict(session, augmented, vocabulary, top_k=10)
            predicted_words = [p[0] for p in predictions]

            total += 1
            word_total[word] += 1

            if predicted_words[0] == word:
                correct_top1 += 1
                word_correct[word] += 1

            if word in predicted_words[:4]:
                correct_top4 += 1

            if word in predicted_words[:10]:
                correct_top10 += 1

    print(f"Results (severity={severity}):")
    print(f"  Top-1 Accuracy: {100 * correct_top1 / total:.1f}%")
    print(f"  Top-4 Accuracy: {100 * correct_top4 / total:.1f}%")
    print(f"  Top-10 Accuracy: {100 * correct_top10 / total:.1f}%")
    print()

    # Show worst performing words
    word_acc = [(w, word_correct[w] / word_total[w]) for w in word_total]
    word_acc.sort(key=lambda x: x[1])

    print("Hardest words (lowest accuracy):")
    for word, acc in word_acc[:10]:
        print(f"  {word}: {100*acc:.0f}%")

    print()
    print("Easiest words (highest accuracy):")
    for word, acc in word_acc[-10:]:
        print(f"  {word}: {100*acc:.0f}%")

    return {
        'top1': correct_top1 / total,
        'top4': correct_top4 / total,
        'top10': correct_top10 / total,
    }


def load_words(path: str, max_words: int = 1000) -> list:
    """Load words from frequency file."""
    words = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                word = parts[0].lower().strip()
                if word and all(c in KEYBOARD_LAYOUT for c in word) and len(word) >= 2:
                    words.append(word)
                    if len(words) >= max_words:
                        break
    return words


def main():
    parser = argparse.ArgumentParser(description='Evaluate blind swipe model')
    parser.add_argument('--model', type=str, required=True, help='ONNX model path')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary JSON path')
    parser.add_argument('--words', type=str, required=True, help='Word frequency file')
    parser.add_argument('--max-words', type=int, default=500, help='Max words to test')
    parser.add_argument('--samples', type=int, default=50, help='Samples per word')
    parser.add_argument('--severity', type=float, default=1.0, help='Augmentation severity')
    args = parser.parse_args()

    # Load model and vocabulary
    print(f"Loading model from {args.model}...")
    session = load_model(args.model)

    print(f"Loading vocabulary from {args.vocab}...")
    with open(args.vocab, 'r') as f:
        vocabulary = json.load(f)
    print(f"Vocabulary size: {len(vocabulary)}")

    # Load test words
    print(f"Loading test words from {args.words}...")
    test_words = load_words(args.words, args.max_words)
    print(f"Test words: {len(test_words)}")

    # Evaluate at different severity levels
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50 + "\n")

    # Light augmentation (easier)
    print("--- Light Augmentation (severity=0.5) ---")
    evaluate(session, vocabulary, test_words, args.samples, severity=0.5)

    # Normal augmentation
    print("\n--- Normal Augmentation (severity=1.0) ---")
    evaluate(session, vocabulary, test_words, args.samples, severity=1.0)

    # Heavy augmentation (harder)
    print("\n--- Heavy Augmentation (severity=1.5) ---")
    evaluate(session, vocabulary, test_words, args.samples, severity=1.5)


if __name__ == '__main__':
    main()
