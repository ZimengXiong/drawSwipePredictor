#!/usr/bin/env python3
"""Initialize word modes from diverse training traces, never evaluation traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import build_encoder, encoder_features
from train_proxy import corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--centers", type=int, default=4)
    args = parser.parse_args()
    if not 2 <= args.centers <= 8:
        parser.error("use 2 to 8 centers")
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows, info = corpus(args.corpus)
    if info["sha256"] != saved["metadata"]["train_corpus"]["sha256"]:
        raise ValueError("proxy modes must use the declared training writers")
    architecture = saved["metadata"]["arguments"].get("architecture", "small")
    model = build_encoder(architecture).to("mps").eval()
    model.load_state_dict(saved["model"])
    vectors = []
    with torch.no_grad():
        for start in range(0, len(rows), 1024):
            paths = torch.tensor(np.array(rows["points"][start:start + 1024]), device="mps")
            vectors.append(model(encoder_features(model, paths)).cpu().numpy())
    vectors = np.concatenate(vectors)
    labels = np.array(rows["label"], dtype=np.int64)
    first = F.normalize(saved["proxies"], dim=-1).numpy()
    if first.ndim != 2:
        raise ValueError("initial mode discovery requires one existing proxy per word")
    centers = np.repeat(first[:, None], args.centers, axis=1)
    closest = np.einsum('nd,nd->n', vectors, first[labels])
    chosen = np.full((len(first), args.centers - 1), -1, dtype=np.int64)
    for mode in range(1, args.centers):
        distance = 1 - closest
        largest = np.full(len(first), -np.inf, dtype=np.float32)
        np.maximum.at(largest, labels, distance)
        matches = np.flatnonzero(distance == largest[labels])
        indices = np.full(len(first), -1, dtype=np.int64)
        indices[labels[matches]] = matches
        present = indices >= 0
        centers[present, mode] = vectors[indices[present]]
        chosen[:, mode - 1] = indices
        closest = np.maximum(closest, np.einsum('nd,nd->n', vectors, centers[labels, mode]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vectors=centers, words=np.asarray(saved["words"]))
    report = {"centers": args.centers, "source": info, "test_data_used": False,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "policy": "First center is the learned proxy; additional centers are farthest training examples within each word",
              "words_with_realized_training_modes": int((chosen[:, 0] >= 0).sum())}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
