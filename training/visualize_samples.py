#!/usr/bin/env python3
"""
Visualize synthetic blind swipe samples for review.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import random

# QWERTY Layout
KEYBOARD_LAYOUT = {
    'q': (0.0, 0.0), 'w': (1.0, 0.0), 'e': (2.0, 0.0), 'r': (3.0, 0.0),
    't': (4.0, 0.0), 'y': (5.0, 0.0), 'u': (6.0, 0.0), 'i': (7.0, 0.0),
    'o': (8.0, 0.0), 'p': (9.0, 0.0),
    'a': (0.5, 1.0), 's': (1.5, 1.0), 'd': (2.5, 1.0), 'f': (3.5, 1.0),
    'g': (4.5, 1.0), 'h': (5.5, 1.0), 'j': (6.5, 1.0), 'k': (7.5, 1.0),
    'l': (8.5, 1.0),
    'z': (1.5, 2.0), 'x': (2.5, 2.0), 'c': (3.5, 2.0), 'v': (4.5, 2.0),
    'b': (5.5, 2.0), 'n': (6.5, 2.0), 'm': (7.5, 2.0),
}

def get_word_path(word: str) -> np.ndarray:
    points = []
    for char in word.lower():
        if char in KEYBOARD_LAYOUT:
            points.append(KEYBOARD_LAYOUT[char])
    if len(points) < 2:
        return np.array([])
    return np.array(points)

def interpolate_path(path: np.ndarray, step: float = 0.3) -> np.ndarray:
    if len(path) < 2:
        return path
    result = [path[0]]
    for i in range(1, len(path)):
        p1, p2 = path[i-1], path[i]
        dist = np.linalg.norm(p2 - p1)
        if dist > step:
            n_steps = int(dist / step)
            for j in range(1, n_steps + 1):
                t = j / (n_steps + 1)
                result.append(p1 + t * (p2 - p1))
        result.append(p2)
    return np.array(result)

def augment_for_blind_swipe(path: np.ndarray, severity: float = 1.0) -> np.ndarray:
    """
    REALISTIC augmentations for blind swipe typing.

    Based on real user behavior:
    - Know relative key locations (directions are correct)
    - Horizontal/vertical scaling varies (same row keys may drift up/down)
    - EXAGGERATE angles (turns are more dramatic, not duller)
    - Round the corners (smooth transitions, but still exaggerated direction)
    - Strokes between keys are relatively straight
    """
    path = path.copy()

    # 1. Global scale variance (moderate)
    scale = np.random.lognormal(0, 0.2 * severity)
    scale = np.clip(scale, 0.5, 2.0)
    path *= scale

    # 2. Independent horizontal/vertical scaling
    x_scale = np.random.lognormal(0, 0.15 * severity)
    y_scale = np.random.lognormal(0, 0.15 * severity)
    x_scale = np.clip(x_scale, 0.7, 1.4)
    y_scale = np.clip(y_scale, 0.7, 1.4)
    centroid = path.mean(axis=0)
    path[:, 0] = (path[:, 0] - centroid[0]) * x_scale + centroid[0]
    path[:, 1] = (path[:, 1] - centroid[1]) * y_scale + centroid[1]

    # 3. EXAGGERATE ANGLES - users overshoot direction changes
    # Push points away from centroid to make angles more dramatic
    if len(path) > 3:
        centroid = path.mean(axis=0)
        exaggeration = 1.0 + random.uniform(0.1, 0.4) * severity
        for i in range(len(path)):
            vec = path[i] - centroid
            path[i] = centroid + vec * exaggeration

    # 4. Per-point vertical drift (same row keys at different heights)
    vertical_drift = np.cumsum(np.random.normal(0, 0.08 * severity, len(path)))
    vertical_drift = gaussian_filter1d(vertical_drift, sigma=3)
    path[:, 1] += vertical_drift

    # 5. Small rotation
    angle = np.random.normal(0, 4 * severity) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    centroid = path.mean(axis=0)
    path = (path - centroid) @ rot_matrix.T + centroid

    # 6. CORNER ROUNDING - smooth transitions (but angles still exaggerated)
    if len(path) > 5:
        sigma = random.uniform(0.8, 1.8) * severity
        path[:, 0] = gaussian_filter1d(path[:, 0], sigma)
        path[:, 1] = gaussian_filter1d(path[:, 1], sigma)

    # 7. Minimal jitter (strokes are clean)
    jitter = np.random.normal(0, 0.02 * severity, path.shape)
    path += jitter

    # 8. Global translation
    global_shift = np.random.normal(0, 0.3 * severity, 2)
    path += global_shift

    return path

def normalize_path(path: np.ndarray) -> np.ndarray:
    """Normalize: center at origin, scale by path length."""
    if len(path) < 2:
        return path
    total_len = 0.0
    for i in range(1, len(path)):
        total_len += np.linalg.norm(path[i] - path[i-1])
    if total_len < 1e-9:
        total_len = 1.0
    centroid = path.mean(axis=0)
    return (path - centroid) / total_len

def plot_word_samples(words, n_samples=5, severity=1.0):
    """Plot ideal path and augmented samples for each word."""
    n_words = len(words)
    fig, axes = plt.subplots(n_words, n_samples + 1, figsize=(3 * (n_samples + 1), 3 * n_words))

    if n_words == 1:
        axes = [axes]

    for row, word in enumerate(words):
        # Get ideal path
        ideal = get_word_path(word)
        if len(ideal) < 2:
            continue
        ideal_interp = interpolate_path(ideal)

        # Plot ideal (first column)
        ax = axes[row][0]
        ax.plot(ideal_interp[:, 0], -ideal_interp[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax.scatter(ideal_interp[0, 0], -ideal_interp[0, 1], c='green', s=100, zorder=5, label='start')
        ax.scatter(ideal_interp[-1, 0], -ideal_interp[-1, 1], c='red', s=100, zorder=5, label='end')
        # Mark letter positions
        letter_path = get_word_path(word)
        for i, (char, pos) in enumerate(zip(word, letter_path)):
            ax.annotate(char, (pos[0], -pos[1]), fontsize=12, ha='center', va='bottom',
                       color='blue', fontweight='bold')
        ax.set_title(f'"{word}" - IDEAL', fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Plot augmented samples
        for col in range(n_samples):
            ax = axes[row][col + 1]

            # Generate augmented sample
            augmented = augment_for_blind_swipe(ideal_interp.copy(), severity)
            normalized = normalize_path(augmented)

            # Plot
            ax.plot(normalized[:, 0], -normalized[:, 1], 'b-', linewidth=1.5, alpha=0.7)
            ax.scatter(normalized[0, 0], -normalized[0, 1], c='green', s=80, zorder=5)
            ax.scatter(normalized[-1, 0], -normalized[-1, 1], c='red', s=80, zorder=5)

            ax.set_title(f'Sample {col + 1} (normalized)', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Blind Swipe Samples (severity={severity})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_same_word_variations(word, n_samples=12, severity=1.0):
    """Plot many variations of the same word to see consistency."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    ideal = get_word_path(word)
    if len(ideal) < 2:
        return
    ideal_interp = interpolate_path(ideal)

    for i, ax in enumerate(axes):
        if i == 0:
            # First plot is ideal
            path = ideal_interp
            title = f'"{word}" IDEAL'
            color = 'blue'
            lw = 2
        else:
            # Rest are augmented
            augmented = augment_for_blind_swipe(ideal_interp.copy(), severity)
            path = normalize_path(augmented)
            title = f'Variation {i}'
            color = 'purple'
            lw = 1.5

        ax.plot(path[:, 0], -path[:, 1], c=color, linewidth=lw, alpha=0.7)
        ax.scatter(path[0, 0], -path[0, 1], c='green', s=60, zorder=5)
        ax.scatter(path[-1, 0], -path[-1, 1], c='red', s=60, zorder=5)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Variations of "{word}" (severity={severity})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_confusable_words(word_pairs, n_samples=4, severity=1.0):
    """Plot pairs of words that might be confusable to see differences."""
    n_pairs = len(word_pairs)
    fig, axes = plt.subplots(n_pairs, n_samples * 2 + 2, figsize=(3 * (n_samples * 2 + 2), 3 * n_pairs))

    if n_pairs == 1:
        axes = [axes]

    for row, (word1, word2) in enumerate(word_pairs):
        # Word 1 ideal
        ideal1 = interpolate_path(get_word_path(word1))
        # Word 2 ideal
        ideal2 = interpolate_path(get_word_path(word2))

        cols_per_word = n_samples + 1

        # Plot word1
        ax = axes[row][0]
        ax.plot(ideal1[:, 0], -ideal1[:, 1], 'b-', linewidth=2)
        ax.scatter(ideal1[0, 0], -ideal1[0, 1], c='green', s=80, zorder=5)
        ax.scatter(ideal1[-1, 0], -ideal1[-1, 1], c='red', s=80, zorder=5)
        ax.set_title(f'"{word1}" IDEAL', fontweight='bold', color='blue')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for j in range(n_samples):
            ax = axes[row][j + 1]
            aug = normalize_path(augment_for_blind_swipe(ideal1.copy(), severity))
            ax.plot(aug[:, 0], -aug[:, 1], 'b-', linewidth=1.5, alpha=0.7)
            ax.scatter(aug[0, 0], -aug[0, 1], c='green', s=60, zorder=5)
            ax.scatter(aug[-1, 0], -aug[-1, 1], c='red', s=60, zorder=5)
            ax.set_title(f'{word1} #{j+1}', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Plot word2
        ax = axes[row][cols_per_word]
        ax.plot(ideal2[:, 0], -ideal2[:, 1], 'r-', linewidth=2)
        ax.scatter(ideal2[0, 0], -ideal2[0, 1], c='green', s=80, zorder=5)
        ax.scatter(ideal2[-1, 0], -ideal2[-1, 1], c='red', s=80, zorder=5)
        ax.set_title(f'"{word2}" IDEAL', fontweight='bold', color='red')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for j in range(n_samples):
            ax = axes[row][cols_per_word + j + 1]
            aug = normalize_path(augment_for_blind_swipe(ideal2.copy(), severity))
            ax.plot(aug[:, 0], -aug[:, 1], 'r-', linewidth=1.5, alpha=0.7)
            ax.scatter(aug[0, 0], -aug[0, 1], c='green', s=60, zorder=5)
            ax.scatter(aug[-1, 0], -aug[-1, 1], c='red', s=60, zorder=5)
            ax.set_title(f'{word2} #{j+1}', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Potentially Confusable Word Pairs (severity={severity})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Test words
    test_words = ['the', 'hello', 'world', 'quick', 'jump']

    print("Generating visualizations...")

    # 1. Multiple words with samples
    fig1 = plot_word_samples(test_words, n_samples=5, severity=1.0)
    fig1.savefig('samples_multiple_words.png', dpi=150, bbox_inches='tight')
    print("Saved: samples_multiple_words.png")

    # 2. Many variations of same word
    fig2 = plot_same_word_variations('hello', n_samples=12, severity=1.0)
    fig2.savefig('samples_hello_variations.png', dpi=150, bbox_inches='tight')
    print("Saved: samples_hello_variations.png")

    # 3. Confusable pairs
    confusable = [
        ('the', 'they'),
        ('was', 'saw'),
        ('from', 'form'),
        ('there', 'three'),
    ]
    fig3 = plot_confusable_words(confusable, n_samples=3, severity=1.0)
    fig3.savefig('samples_confusable_pairs.png', dpi=150, bbox_inches='tight')
    print("Saved: samples_confusable_pairs.png")

    # 4. Different severity levels
    fig4, axes = plt.subplots(3, 6, figsize=(18, 9))
    word = 'python'
    ideal = interpolate_path(get_word_path(word))

    for row, severity in enumerate([0.5, 1.0, 1.5]):
        ax = axes[row][0]
        ax.plot(ideal[:, 0], -ideal[:, 1], 'b-', linewidth=2)
        ax.set_title(f'IDEAL', fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f'severity={severity}', fontsize=12, fontweight='bold')

        for col in range(1, 6):
            ax = axes[row][col]
            aug = normalize_path(augment_for_blind_swipe(ideal.copy(), severity))
            ax.plot(aug[:, 0], -aug[:, 1], 'purple', linewidth=1.5, alpha=0.7)
            ax.scatter(aug[0, 0], -aug[0, 1], c='green', s=50, zorder=5)
            ax.scatter(aug[-1, 0], -aug[-1, 1], c='red', s=50, zorder=5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'"{word}" at Different Augmentation Severities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig4.savefig('samples_severity_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: samples_severity_comparison.png")

    print("\nDone! Check the generated PNG files.")
    plt.show()
