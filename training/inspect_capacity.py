#!/usr/bin/env python3
"""Measure residual capacity and gradients using declared training strokes only."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from retrieval_model import build_encoder, encoder_features
from train_proxy import corpus, proxy_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/memory-proxy/train.bin"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    content = args.checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    rows, info = corpus(args.train)
    metadata = saved["metadata"]
    if metadata["test_data_used"] or info["sha256"] != metadata["train_corpus"]["sha256"]:
        raise ValueError("capacity inspection requires the declared training corpus")
    model = build_encoder(metadata["arguments"]["architecture"]).eval()
    model.load_state_dict(saved["model"])
    # Fixed training rows make successive checkpoint diagnostics comparable.
    indices = np.random.default_rng(900631).choice(len(rows), 32, replace=False)
    paths = torch.tensor(np.array(rows["points"][indices]))
    labels = torch.tensor(np.array(rows["label"][indices], dtype=np.int64))
    measurements = []
    handles = []

    def observe(index):
        def hook(module, inputs, output):
            source = inputs[0].detach()
            branch = output.detach()
            measurements.append({"block": index,
                                 "residual_rms": branch.square().mean().sqrt().item(),
                                 "input_rms": source.square().mean().sqrt().item()})
        return hook

    for index, block in enumerate(model.blocks):
        handles.append(block.layers.register_forward_hook(observe(index)))
    query = model(encoder_features(model, paths))
    loss = F.cross_entropy(32 * proxy_scores(query, saved["proxies"]), labels)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training diagnostic loss")
    loss.backward()
    for handle in handles:
        handle.remove()
    for result, block in zip(measurements, model.blocks):
        gradients = [p.grad for p in block.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("missing or nonfinite residual gradient")
        result["gradient_norm"] = sum(gradient.square().sum().item() for gradient in gradients) ** 0.5
        # The source encoder has 192 hidden channels in each residual block.
        added = block.layers[3].weight.grad[:, 192:]
        result["added_hidden_output_gradient_norm"] = added.norm().item()
        result["added_hidden_output_weight_norm"] = block.layers[3].weight[:, 192:].norm().item()
    report = {"checkpoint_sha256": hashlib.sha256(content).hexdigest(),
              "epoch": saved["epoch"] + 1, "architecture": metadata["arguments"]["architecture"],
              "training_corpus": info, "samples": len(indices), "loss": loss.item(),
              "blocks": measurements, "test_data_used": False,
              "limits": "Training-only activation and gradient diagnostic in evaluation mode; no accuracy estimate or parameter update."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
