#!/usr/bin/env python3
"""
CTC-based sequence-to-sequence model for blind swipe typing.

This model learns to decode gesture paths into letter sequences,
allowing it to generalize to ANY word - even words never seen in training.

Architecture:
    Path points → Feature Extraction → Conv1D → BiLSTM Encoder → CTC Decoder

Features per timestep (7 dimensions):
    - (x, y): normalized position
    - (dx, dy): delta to next point (local direction)
    - angle: direction angle (radians / pi, normalized to [-1, 1])
    - speed: segment length relative to average
    - curvature: change in direction angle

Example:
    Input: [(x1,y1), (x2,y2), ...] (normalized path for "hello")
    Output: "hello" (decoded letter by letter)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from typing import List, Tuple
import random
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os
import math

# MPS doesn't support CTC loss, so we compute loss on CPU
# Model runs on MPS (fast), loss on CPU (small overhead)
USE_CPU_FOR_CTC = True

# Character set: a-z + blank (for CTC)
CHARS = list("abcdefghijklmnopqrstuvwxyz")
BLANK_IDX = 0  # CTC blank token
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 1-26 for a-z
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # 27 (blank + 26 letters)

# QWERTY Layout
KEYBOARD_LAYOUT = {
    "q": (0.0, 0.0),
    "w": (1.0, 0.0),
    "e": (2.0, 0.0),
    "r": (3.0, 0.0),
    "t": (4.0, 0.0),
    "y": (5.0, 0.0),
    "u": (6.0, 0.0),
    "i": (7.0, 0.0),
    "o": (8.0, 0.0),
    "p": (9.0, 0.0),
    "a": (0.5, 1.0),
    "s": (1.5, 1.0),
    "d": (2.5, 1.0),
    "f": (3.5, 1.0),
    "g": (4.5, 1.0),
    "h": (5.5, 1.0),
    "j": (6.5, 1.0),
    "k": (7.5, 1.0),
    "l": (8.5, 1.0),
    "z": (1.5, 2.0),
    "x": (2.5, 2.0),
    "c": (3.5, 2.0),
    "v": (4.5, 2.0),
    "b": (5.5, 2.0),
    "n": (6.5, 2.0),
    "m": (7.5, 2.0),
}


# ============================================================================
# Path Generation
# ============================================================================


def get_word_path(word: str) -> np.ndarray:
    """Convert word to keyboard path (ideal, exact key centers)."""
    points = []
    for char in word.lower():
        if char in KEYBOARD_LAYOUT:
            points.append(KEYBOARD_LAYOUT[char])
    if len(points) < 2:
        return np.array([])
    return np.array(points)


def interpolate_path(path: np.ndarray, step: float = 0.3) -> np.ndarray:
    """Interpolate path with straight lines for smooth gesture."""
    if len(path) < 2:
        return path
    result = [path[0]]
    for i in range(1, len(path)):
        p1, p2 = path[i - 1], path[i]
        dist = np.linalg.norm(p2 - p1)
        if dist > step:
            n_steps = int(dist / step)
            for j in range(1, n_steps + 1):
                t = j / (n_steps + 1)
                result.append(p1 + t * (p2 - p1))
        result.append(p2)
    return np.array(result)


def resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """Resample to fixed number of points."""
    if len(path) < 2:
        return np.zeros((n_points, 2))

    dists = [0.0]
    for i in range(1, len(path)):
        dists.append(dists[-1] + np.linalg.norm(path[i] - path[i - 1]))

    total_len = dists[-1]
    if total_len < 1e-9:
        return np.tile(path[0], (n_points, 1))

    resampled = []
    for i in range(n_points):
        target = (i / (n_points - 1)) * total_len
        for j in range(1, len(dists)):
            if dists[j] >= target:
                t = (target - dists[j - 1]) / (dists[j] - dists[j - 1] + 1e-9)
                resampled.append(path[j - 1] + t * (path[j] - path[j - 1]))
                break
        else:
            resampled.append(path[-1])

    return np.array(resampled)


def normalize_path(path: np.ndarray) -> np.ndarray:
    """Normalize: center and scale."""
    if len(path) < 2:
        return path

    total_len = sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path)))
    if total_len < 1e-9:
        total_len = 1.0

    centroid = path.mean(axis=0)
    return (path - centroid) / total_len


# Number of input features per timestep
INPUT_DIM = 7  # x, y, dx, dy, angle, speed, curvature


def extract_features(path: np.ndarray) -> np.ndarray:
    """
    Extract rich per-timestep features from a normalized path.

    For each of the N points, computes 7 features:
        (x, y, dx, dy, angle, speed, curvature)

    This gives the model local directional context at each timestep,
    vastly improving CTC alignment compared to raw (x, y) alone.
    """
    n = len(path)
    features = np.zeros((n, INPUT_DIM), dtype=np.float32)

    # Position features
    features[:, 0] = path[:, 0]
    features[:, 1] = path[:, 1]

    # Compute segment displacements and lengths
    deltas = np.diff(path, axis=0)  # (n-1, 2)
    seg_lengths = np.linalg.norm(deltas, axis=1)  # (n-1,)
    avg_seg = seg_lengths.mean() if seg_lengths.mean() > 1e-9 else 1.0

    # Delta features (forward difference, last point repeats)
    features[:-1, 2] = deltas[:, 0]
    features[:-1, 3] = deltas[:, 1]
    features[-1, 2] = deltas[-1, 0]
    features[-1, 3] = deltas[-1, 1]

    # Angle features (direction of each segment, normalized to [-1, 1])
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])  # (n-1,)
    features[:-1, 4] = angles / math.pi
    features[-1, 4] = angles[-1] / math.pi

    # Speed features (segment length / average segment length)
    features[:-1, 5] = seg_lengths / avg_seg
    features[-1, 5] = seg_lengths[-1] / avg_seg

    # Curvature features (change in angle between consecutive segments)
    if n > 2:
        angle_diffs = np.diff(angles)
        # Wrap to [-pi, pi]
        angle_diffs = (angle_diffs + math.pi) % (2 * math.pi) - math.pi
        features[1:-1, 6] = angle_diffs / math.pi
    # First and last curvature stay 0

    return features


# ============================================================================
# Augmentation
# ============================================================================


def augment_for_blind_swipe(path: np.ndarray, severity: float = 1.0) -> np.ndarray:
    """
    Augment a swipe path to simulate drawing from memory on a blank trackpad.

    The user has NO keyboard visible. They remember the approximate shape of
    the swipe gesture and draw it from memory. This means:

    1. The overall shape is approximately right but proportions are distorted
    2. Angles are remembered roughly (±20-30°)
    3. Some segments are compressed/stretched relative to others
    4. Sharp turns get rounded (momentum) or exaggerated
    5. The path may be drawn at any scale, rotation, or position
    6. Parts of the path the user is less confident about are sloppier
    7. Fine details (small direction changes) may be lost entirely
    """
    path = path.copy()

    # --- Global shape transforms (mental keyboard can be any size/orientation) ---

    # Arbitrary scale: the user's mental model has no absolute size
    scale = np.random.lognormal(0, 0.3 * severity)
    scale = np.clip(scale, 0.3, 3.0)
    path *= scale

    # Arbitrary aspect ratio: mental keyboard proportions vary widely
    # (some people remember wider, some taller layouts)
    x_scale = np.clip(np.random.lognormal(0, 0.25 * severity), 0.5, 2.0)
    y_scale = np.clip(np.random.lognormal(0, 0.25 * severity), 0.5, 2.0)
    centroid = path.mean(axis=0)
    path[:, 0] = (path[:, 0] - centroid[0]) * x_scale + centroid[0]
    path[:, 1] = (path[:, 1] - centroid[1]) * y_scale + centroid[1]

    # Rotation: mental keyboard orientation is approximate
    angle = np.random.normal(0, 8 * severity) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    centroid = path.mean(axis=0)
    path = (path - centroid) @ rot.T + centroid

    # --- Segment-level distortions (remembered proportions are approximate) ---

    # Non-uniform segment stretching: some parts of the path get
    # compressed or stretched relative to others. This is THE key
    # distortion in drawing from memory - relative proportions are
    # only approximately correct.
    if len(path) > 4:
        n = len(path)
        t = np.linspace(0, 1, n)
        # Create smooth warp function that stretches some regions
        n_waves = random.randint(1, 3)
        warp = np.zeros(n)
        for _ in range(n_waves):
            freq = random.uniform(0.5, 2.5)
            phase = random.uniform(0, 2 * math.pi)
            amp = random.uniform(0.05, 0.2) * severity
            warp += amp * np.sin(freq * math.pi * t + phase)
        t_warped = t + warp
        t_warped = np.clip(t_warped, 0, 1)
        # Ensure monotonic
        for i in range(1, n):
            if t_warped[i] <= t_warped[i - 1]:
                t_warped[i] = t_warped[i - 1] + 1e-6
        t_warped = (t_warped - t_warped[0]) / (t_warped[-1] - t_warped[0] + 1e-9)
        new_path = np.zeros_like(path)
        for dim in range(2):
            new_path[:, dim] = np.interp(t_warped, t, path[:, dim])
        path = new_path

    # --- Angle/direction distortions (memory recall errors) ---

    # Exaggerate or flatten the overall shape
    if len(path) > 3:
        centroid = path.mean(axis=0)
        # Can be < 1.0 (flattened) or > 1.0 (exaggerated)
        exaggeration = np.clip(
            np.random.normal(1.0, 0.2 * severity), 0.7, 1.5
        )
        for i in range(len(path)):
            vec = path[i] - centroid
            path[i] = centroid + vec * exaggeration

    # --- Smooth drift (hand wanders during drawing) ---

    # Vertical drift: hand drifts up or down during the drawing
    drift = gaussian_filter1d(
        np.cumsum(np.random.normal(0, 0.1 * severity, len(path))), 3
    )
    path[:, 1] += drift

    # Horizontal drift: hand drifts left or right
    h_drift = gaussian_filter1d(
        np.cumsum(np.random.normal(0, 0.06 * severity, len(path))), 4
    )
    path[:, 0] += h_drift

    # --- Corner rounding (finger momentum) ---
    # Drawing from memory, users tend to round sharp corners even more
    # because they're moving continuously rather than targeting specific keys
    if len(path) > 5:
        sigma = random.uniform(1.0, 2.5) * severity
        path[:, 0] = gaussian_filter1d(path[:, 0], sigma)
        path[:, 1] = gaussian_filter1d(path[:, 1], sigma)

    # --- Spatially-varying confidence ---
    # Some parts of the word shape are remembered better than others.
    # Well-remembered parts are precise; fuzzy parts get extra noise.
    if len(path) > 4 and random.random() < 0.6:
        n = len(path)
        t = np.linspace(0, 1, n)
        # Random "uncertainty envelope" - some regions are sloppier
        n_bumps = random.randint(1, 3)
        envelope = np.zeros(n)
        for _ in range(n_bumps):
            center = random.uniform(0, 1)
            width = random.uniform(0.1, 0.4)
            amplitude = random.uniform(0.5, 2.5) * severity
            envelope += amplitude * np.exp(-((t - center) ** 2) / (2 * width**2))
        local_noise = np.random.normal(0, 0.05, path.shape)
        local_noise[:, 0] *= envelope
        local_noise[:, 1] *= envelope
        path += local_noise

    # --- General drawing noise (hand tremor/trackpad precision) ---
    path += np.random.normal(0, 0.03 * severity, path.shape)

    # --- Position: arbitrary, since there's no reference frame ---
    path += np.random.normal(0, 0.5 * severity, 2)

    return path


# ============================================================================
# Dataset
# ============================================================================


class CTCSwipeDataset(Dataset):
    """Dataset for CTC training - outputs letter sequences.

    Simulates blind swipe drawing: users recall the shape of a word's
    swipe path from memory and draw it on a blank trackpad. There is
    no keyboard visible - the user is reproducing the remembered gesture
    shape with all the distortions that entails (wrong proportions,
    rounded corners, missing details, arbitrary scale/rotation).
    """

    def __init__(
        self,
        words: List[str],
        samples_per_word: int = 200,
        n_points: int = 64,
        augment_severity: float = 1.0,
    ):
        self.n_points = n_points
        self.augment_severity = augment_severity
        self.samples_per_word = samples_per_word

        # Pre-compute ideal paths and filter valid words
        self.data = []
        for word in words:
            path = get_word_path(word)
            if len(path) >= 2:
                path = interpolate_path(path)
                # Label is the character indices
                label = [CHAR_TO_IDX[c] for c in word.lower() if c in CHAR_TO_IDX]
                if len(label) >= 2:
                    self.data.append((word, path, label))

        print(f"  Valid words for CTC: {len(self.data)}")

    def __len__(self):
        return len(self.data) * self.samples_per_word

    def __getitem__(self, idx):
        word_idx = idx // self.samples_per_word
        word, ideal_path, label = self.data[word_idx]

        # Simulate drawing the path from memory
        augmented = augment_for_blind_swipe(
            ideal_path.copy(), self.augment_severity
        )

        # Resample and normalize
        resampled = resample_path(augmented, self.n_points)
        normalized = normalize_path(resampled)

        # Extract rich features (x, y, dx, dy, angle, speed, curvature)
        features = extract_features(normalized)

        features_tensor = torch.from_numpy(features)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor, len(label)


def collate_fn(batch):
    """Custom collate for variable length labels."""
    features, labels, label_lengths = zip(*batch)

    # Stack features (all same length due to resampling)
    features = torch.stack(features)

    # Concatenate labels and track lengths
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return features, labels, label_lengths


# ============================================================================
# Model
# ============================================================================


class CTCSwipeModel(nn.Module):
    """
    Conv1D + BiLSTM encoder with CTC decoder for gesture-to-text.

    Architecture:
        1. Conv1D front-end: extracts local patterns (n-gram-like features)
        2. BiLSTM encoder: captures sequential dependencies
        3. Linear projection: maps to character probabilities

    Input: (batch, seq_len, input_dim) - extracted path features
    Output: (seq_len, batch, num_classes) - character probabilities
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        conv_channels: int = 128,
    ):
        super().__init__()

        # Conv1D front-end for local feature extraction
        self.conv = nn.Sequential(
            # First conv block: input_dim -> conv_channels
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            # Second conv block: conv_channels -> conv_channels
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.encoder = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # Back to (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        encoder_out, _ = self.encoder(x)  # (batch, seq_len, hidden*2)
        logits = self.fc(encoder_out)  # (batch, seq_len, num_classes)

        # CTC expects (seq_len, batch, num_classes)
        logits = logits.permute(1, 0, 2)

        # Log softmax for CTC loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        return log_probs


# ============================================================================
# Decoding
# ============================================================================


def ctc_greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Greedy CTC decoding.

    Args:
        log_probs: (seq_len, batch, num_classes)

    Returns:
        List of decoded strings
    """
    # Get most likely class at each timestep
    _, predictions = log_probs.max(dim=2)  # (seq_len, batch)
    predictions = predictions.permute(1, 0)  # (batch, seq_len)

    decoded = []
    for pred in predictions:
        chars = []
        prev = BLANK_IDX
        for idx in pred.tolist():
            if idx != BLANK_IDX and idx != prev:
                if idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
            prev = idx
        decoded.append("".join(chars))

    return decoded


def ctc_beam_decode(log_probs: torch.Tensor, beam_width: int = 10) -> List[str]:
    """
    Beam search CTC decoding for better accuracy.
    """
    # Simple greedy for now - can add beam search later
    return ctc_greedy_decode(log_probs)


# ============================================================================
# Training
# ============================================================================


def train_ctc_model(
    words: List[str],
    n_points: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    samples_per_word: int = 200,
    device: str = "cpu",
    checkpoint_dir: str = "training/checkpoints",
    checkpoint_every: int = 1,
    resume_from: str = None,
    warmup_epochs: int = 3,
):
    """Train the CTC model with improved training schedule."""

    # Setup device
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_cpu_ctc = True  # MPS doesn't support CTC
        print("  Note: Using MPS for model, CPU for CTC loss")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        use_cpu_ctc = False
    else:
        device = torch.device("cpu")
        use_cpu_ctc = False

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nTraining CTC Model")
    print(f"  Device: {device}")
    print(f"  Words: {len(words)}")
    print(f"  Points per path: {n_points}")
    print(f"  Input features: {INPUT_DIM} (x, y, dx, dy, angle, speed, curvature)")
    print(f"  Samples per word: {samples_per_word}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Checkpoints: {checkpoint_dir}/ (every {checkpoint_every} epoch)")

    # Create datasets
    train_dataset = CTCSwipeDataset(
        words, samples_per_word, n_points, augment_severity=1.0
    )
    val_dataset = CTCSwipeDataset(
        words, samples_per_word=30, n_points=n_points, augment_severity=0.7
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create model with Conv1D front-end
    model = CTCSwipeModel(
        input_dim=INPUT_DIM, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # CTC Loss (always on CPU if using MPS)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Warmup + cosine annealing schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    best_model_state = None
    history = {"train_loss": [], "val_word_acc": [], "val_char_acc": []}
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"\n  Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", history)
        best_val_acc = checkpoint.get("val_word_acc", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best acc: {best_val_acc:.1f}%")

    print("\nStarting training...\n")

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        )
        for batch_idx, (features, labels, label_lengths) in enumerate(pbar):
            features = features.to(device)

            optimizer.zero_grad()

            # Forward pass on device (MPS/CUDA/CPU)
            log_probs = model(features)  # (seq_len, batch, num_classes)

            # CTC loss - compute on CPU if using MPS
            if use_cpu_ctc:
                log_probs_cpu = log_probs.cpu()
                labels_cpu = labels  # already on CPU from dataloader
                label_lengths_cpu = label_lengths
                input_lengths = torch.full(
                    (features.size(0),), n_points, dtype=torch.long
                )
                loss = ctc_loss(
                    log_probs_cpu, labels_cpu, input_lengths, label_lengths_cpu
                )
                # Backward - gradients flow back to MPS
                loss.backward()
            else:
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)
                input_lengths = torch.full(
                    (features.size(0),), n_points, dtype=torch.long, device=device
                )
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
                loss.backward()

            if torch.isfinite(loss):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Mid-epoch checkpoint every 5000 batches
            if (batch_idx + 1) % 5000 == 0:
                mid_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}_batch_{batch_idx + 1}.pt",
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss / n_batches,
                    },
                    mid_path,
                )
                tqdm.write(f"   Mid-epoch save: {mid_path}")

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_char_correct = 0
        val_char_total = 0

        with torch.no_grad():
            for features, labels, label_lengths in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False
            ):
                features = features.to(device)

                log_probs = model(features)
                decoded = ctc_greedy_decode(
                    log_probs.cpu() if use_cpu_ctc else log_probs
                )

                # Reconstruct target words
                label_list = labels.tolist()
                offset = 0
                for i, length in enumerate(label_lengths.tolist()):
                    target_indices = label_list[offset : offset + length]
                    target_word = "".join(
                        IDX_TO_CHAR.get(idx, "") for idx in target_indices
                    )
                    pred_word = decoded[i]

                    val_total += 1
                    if pred_word == target_word:
                        val_correct += 1

                    # Character-level accuracy
                    for j, (p, t) in enumerate(zip(pred_word, target_word)):
                        val_char_total += 1
                        if p == t:
                            val_char_correct += 1
                    val_char_total += abs(len(pred_word) - len(target_word))

                    offset += length

        val_word_acc = 100.0 * val_correct / max(val_total, 1)
        val_char_acc = 100.0 * val_char_correct / max(val_char_total, 1)

        history["train_loss"].append(train_loss / max(n_batches, 1))
        history["val_word_acc"].append(val_word_acc)
        history["val_char_acc"].append(val_char_acc)

        if val_word_acc > best_val_acc:
            best_val_acc = val_word_acc
            best_model_state = model.state_dict().copy()

        print(
            f"Epoch {epoch + 1:2d}/{epochs}: "
            f"Loss: {train_loss / max(n_batches, 1):.4f}, "
            f"Word Acc: {val_word_acc:.1f}%, "
            f"Char Acc: {val_char_acc:.1f}%"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_word_acc": val_word_acc,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"   Saved: {checkpoint_path}")

    print(f"\nBest word accuracy: {best_val_acc:.1f}%")

    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_dataset.data


def export_ctc_model(model, output_path: str, n_points: int):
    """Export CTC model to ONNX."""
    model.eval()
    model_cpu = model.to("cpu")

    dummy_input = torch.randn(1, n_points, INPUT_DIM)

    try:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["log_probs"],
            dynamic_axes={"input": {0: "batch_size"}, "log_probs": {1: "batch_size"}},
            opset_version=11,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ONNX export failed due to missing dependency. "
            "Install with: pip install onnx onnxscript"
        ) from e

    print(f"Model exported to {output_path}")

    # Save character mapping and model config
    vocab_path = output_path.replace(".onnx", "_chars.json")
    with open(vocab_path, "w") as f:
        json.dump({
            "chars": CHARS,
            "blank_idx": BLANK_IDX,
            "input_dim": INPUT_DIM,
            "n_points": n_points,
        }, f)
    print(f"Character mapping saved to {vocab_path}")


# ============================================================================
# Main
# ============================================================================


def load_words(path: str, max_words: int = 50000) -> List[str]:
    """Load words from frequency file."""
    words = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                word = parts[0].lower().strip()
                if word and all(c in KEYBOARD_LAYOUT for c in word) and len(word) >= 2:
                    words.append(word)
                    if len(words) >= max_words:
                        break
    return words


def test_generalization(
    model, train_words: set, test_words: List[str], n_points: int, device
):
    """Test if model generalizes to unseen words."""
    model.eval()

    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0

    print("\nTesting generalization...")

    with torch.no_grad():
        for word in tqdm(test_words[:500], desc="Testing"):
            path = get_word_path(word)
            if len(path) < 2:
                continue

            path = interpolate_path(path)
            # Light augmentation
            path = augment_for_blind_swipe(path, severity=0.5)
            path = resample_path(path, n_points)
            path = normalize_path(path)

            # Extract rich features
            feat = extract_features(path)
            features = torch.from_numpy(feat).unsqueeze(0).to(device)
            log_probs = model(features)
            # Decode on CPU (needed for MPS)
            decoded = ctc_greedy_decode(log_probs.cpu())[0]

            is_seen = word in train_words
            is_correct = decoded == word

            if is_seen:
                seen_total += 1
                if is_correct:
                    seen_correct += 1
            else:
                unseen_total += 1
                if is_correct:
                    unseen_correct += 1

    print(
        f"\n  Seen words:   {100 * seen_correct / max(seen_total, 1):.1f}% ({seen_correct}/{seen_total})"
    )
    print(
        f"  Unseen words: {100 * unseen_correct / max(unseen_total, 1):.1f}% ({unseen_correct}/{unseen_total})"
    )

    return seen_correct, seen_total, unseen_correct, unseen_total


def main():
    parser = argparse.ArgumentParser(description="Train CTC model for blind swipe")
    parser.add_argument("--words", type=str, required=True, help="Word frequency file")
    parser.add_argument(
        "--output", type=str, default="swipe_ctc.onnx", help="Output model"
    )
    parser.add_argument(
        "--max-words", type=int, default=10000, help="Max training words"
    )
    parser.add_argument(
        "--test-words", type=int, default=5000, help="Words for testing generalization"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=3, help="LSTM layers")
    parser.add_argument("--samples", type=int, default=200, help="Samples per word")
    parser.add_argument("--points", type=int, default=64, help="Points per path")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="mps", help="Device (mps/cuda/cpu)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint file"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup epochs for learning rate"
    )
    args = parser.parse_args()

    # Load all words
    print(f"Loading words from {args.words}...")
    all_words = load_words(args.words, args.max_words + args.test_words)

    # Split: train on first N, test generalization on rest
    train_words = all_words[: args.max_words]
    test_words = all_words[args.max_words : args.max_words + args.test_words]

    print(f"Training words: {len(train_words)}")
    print(f"Test words (for generalization): {len(test_words)}")

    # Train
    model, data = train_ctc_model(
        words=train_words,
        n_points=args.points,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_word=args.samples,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume,
        warmup_epochs=args.warmup,
    )

    # Test generalization
    device = torch.device(
        args.device
        if args.device != "mps" or torch.backends.mps.is_available()
        else "cpu"
    )
    if args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_word_set = set(train_words)
    test_generalization(model, train_word_set, test_words, args.points, device)

    # Export
    export_ctc_model(model, args.output, args.points)

    print("\nDone!")


if __name__ == "__main__":
    main()
