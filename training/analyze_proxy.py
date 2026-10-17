#!/usr/bin/env python3
"""Inspect development errors and shortlist recall without opening the benchmark."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import GestureEncoder, build_encoder, encoder_features, features
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="mps", weights_only=False)
    rows, info = corpus(args.data)
    if info["sha256"] != saved["metadata"]["validation_corpus"]["sha256"]:
        raise ValueError("diagnostics require the declared development corpus")
    selected = np.flatnonzero(rows["difficulty"] == 2)[:args.samples]
    paths = torch.tensor(np.array(rows["points"][selected]), device="mps")
    labels = torch.tensor(np.array(rows["label"][selected], dtype=np.int64), device="mps")
    model = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
    model.load_state_dict(saved["model"])
    bank = saved["proxies"]
    hits, failures = [], []
    with torch.no_grad():
        for start in range(0, len(paths), 256):
            ranked = (proxy_scores(model(encoder_features(model, paths[start:start + 256])), bank)).topk(128, dim=1).indices
            truth = labels[start:start + 256, None]
            hits.append(torch.stack([(ranked[:, :k] == truth).any(1) for k in [1, 5, 32, 128]], 1))
            for target, predictions in zip(truth[:, 0].tolist(), ranked[:, :5].tolist()):
                if predictions[0] != target and len(failures) < 30:
                    failures.append({"target": saved["words"][target], "predictions": [saved["words"][i] for i in predictions]})
    hits = torch.cat(hits).float().cpu()
    accuracy = hits.mean(0).tolist()
    target_words = [saved["words"][int(i)] for i in labels.cpu()]
    routes = ["".join(c for i, c in enumerate(word) if i == 0 or c != word[i - 1]) for word in saved["words"]]
    route_counts = Counter(routes)
    groups = {}
    for name, mask in {
        "length_4_to_5": [4 <= len(word) <= 5 for word in target_words],
        "length_6_to_8": [6 <= len(word) <= 8 for word in target_words],
        "length_9_to_12": [9 <= len(word) <= 12 for word in target_words],
        "shared_collapsed_route": [route_counts[routes[int(i)]] > 1 for i in labels.cpu()],
        "unique_collapsed_route": [route_counts[routes[int(i)]] == 1 for i in labels.cpu()],
    }.items():
        mask = torch.tensor(mask)
        groups[name] = {"samples": int(mask.sum()),
                        "recall": dict(zip(["top1", "top5", "top32", "top128"], hits[mask].mean(0).tolist()))}
    report = {"checkpoint_epoch": saved["epoch"] + 1, "development_samples": len(paths),
              "recall": dict(zip(["top1", "top5", "top32", "top128"], accuracy)), "errors": failures,
              "groups": groups, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "test_data_used": False, "development_sha256": info["sha256"],
              "scoring": "Exact full word; route groups are diagnostics only and grant no alias credit"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
