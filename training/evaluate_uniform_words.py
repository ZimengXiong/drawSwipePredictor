#!/usr/bin/env python3
"""Evaluate an unrestricted uniform word ensemble on declared development data."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from retrieval_model import build_encoder, encoder_features
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/memory-proxy/validation.bin"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(4)
    rows, info = corpus(args.data)
    models, checkpoints, words = [], [], None
    for path in args.checkpoint:
        content = path.read_bytes()
        saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
        metadata = saved["metadata"]
        if metadata["test_data_used"] or metadata["validation_corpus"]["sha256"] != info["sha256"]:
            raise ValueError("uniform ensemble requires the declared development writers")
        if len(saved["words"]) != 60000 or (words is not None and words != saved["words"]):
            raise ValueError("uniform ensemble requires identical complete vocabularies")
        words = saved["words"]
        architecture = metadata["arguments"].get("architecture", "small")
        model = build_encoder(architecture).to("mps").eval().requires_grad_(False)
        model.load_state_dict(saved["model"])
        models.append((model, saved["proxies"].to("mps")))
        checkpoints.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "architecture": architecture, "epoch": saved["epoch"] + 1})
    if len({item["sha256"] for item in checkpoints}) != len(checkpoints):
        raise ValueError("supply each component once")
    count = len(models)
    paths = torch.tensor(np.array(rows["points"]), device="mps")
    labels = torch.tensor(np.array(rows["label"], dtype=np.int64), device="mps")
    difficulty = torch.tensor(np.array(rows["difficulty"], dtype=np.int64), device="mps")
    totals = torch.zeros(3, 4, device="mps")
    with torch.no_grad():
        for start in range(0, len(paths), 192):
            part = paths[start:start + 192]
            scores = sum(proxy_scores(model(encoder_features(model, part)), proxies) / count for model, proxies in models)
            ranked = scores.topk(128, dim=1).indices
            truth = labels[start:start + len(part), None]
            hits = torch.stack([(ranked[:, :k] == truth).any(1) for k in [1, 5, 32, 128]], dim=1).float()
            for level in range(3):
                mask = difficulty[start:start + len(part)] == level
                totals[level] += (hits * mask[:, None]).sum(0)
    validation = {}
    for level, name in enumerate(["clean", "loose", "memory"]):
        size = int((rows["difficulty"] == level).sum())
        validation[name] = {"samples": size, **dict(zip(["top1", "top5", "top32", "top128"], (totals[level].cpu() / size).tolist()))}
    selected = {"weights": [1 / count] * count, "validation": validation}
    report = {"test_data_used": False, "development": info, "checkpoints": checkpoints, "full_vocabulary": 60000,
              "selected": selected, "trials": [selected], "source_sha256": source_hash,
              "selection": "One fixed equal-weight mixture; no weight tuning", "required_native_components": count,
              "native_verified": False,
              "limits": "Development only. Every component is retained. Larger native mixtures must be implemented and verified before promotion; no benchmark or confirmation evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
