#!/usr/bin/env python3
"""Mine close candidate routes from declared training or development writers."""
import argparse
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
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=931719)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows, info = corpus(args.corpus)
    if info["sha256"] != saved["metadata"][args.split + "_corpus"]["sha256"]:
        raise ValueError("pair mining must preserve the retriever's data split")
    chooser = np.random.default_rng(args.seed)
    select = chooser.choice(len(rows), min(args.samples, len(rows)), replace=False)
    paths = np.array(rows["points"][select])
    labels = np.array(rows["label"][select], dtype=np.int64)
    model = build_encoder(saved["metadata"]["arguments"].get("architecture", "small")).to("mps").eval()
    model.load_state_dict(saved["model"])
    bank = saved["proxies"].to("mps")
    candidates = np.zeros((len(paths), 128), dtype=np.int32)
    distances = np.zeros((len(paths), 128), dtype=np.float32)
    positive_distance = np.zeros(len(paths), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(paths), 256):
            query = model(encoder_features(model, torch.tensor(paths[start:start + 256], device="mps")))
            score = proxy_scores(query, bank)
            values, indices = score.topk(128, dim=1)
            candidates[start:start + len(query)] = indices.cpu().numpy()
            distances[start:start + len(query)] = (1 - values).clamp_min(0).cpu().numpy()
            truth = torch.tensor(labels[start:start + len(query)], device="mps")
            positive_distance[start:start + len(query)] = (1 - score.gather(1, truth[:, None])[:, 0]).clamp_min(0).cpu().numpy()
            if start % 25600 == 0:
                print(f"{start}/{len(paths)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, paths=paths, labels=labels, candidates=candidates, distances=distances,
        positive_distance=positive_distance, difficulty=np.array(rows["difficulty"][select]), writers=np.array(rows["writer"][select]))
    report = {"split": args.split, "samples": len(paths), "seed": args.seed, "source": info,
              "retriever_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "checkpoint_epoch": saved["epoch"] + 1, "test_data_used": False,
              "candidate_recall": float((candidates == labels[:, None]).any(1).mean()),
              "policy": "Keep actual top128 candidates. Inject the target for training only, never during validation."}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
