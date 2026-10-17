#!/usr/bin/env python3
"""Inspect development errors and shortlist recall without opening the benchmark."""
import argparse
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
    accuracy = torch.cat(hits).float().mean(0).tolist()
    report = {"checkpoint_epoch": saved["epoch"] + 1, "development_samples": len(paths),
              "recall": dict(zip(["top1", "top5", "top32", "top128"], accuracy)), "errors": failures,
              "test_data_used": False, "development_sha256": info["sha256"]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
