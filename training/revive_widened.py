#!/usr/bin/env python3
"""Repair inactive widened channels without changing checkpoint predictions."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from retrieval_model import build_encoder, encoder_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=900359)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output checkpoint")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    content = args.checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    if saved["metadata"]["arguments"]["architecture"] != "giant":
        raise ValueError("repair applies only to the widened giant encoder")
    model = build_encoder("giant").eval()
    model.load_state_dict(saved["model"])
    paths = torch.randn(8, 64, 2)
    with torch.no_grad():
        expected = model(encoder_features(model, paths))
    fresh = build_encoder("giant")
    revived = []
    with torch.no_grad():
        for i in range(6):
            block, initial = model.blocks[i], fresh.blocks[i]
            if torch.count_nonzero(block.layers[0].weight[192:]) or torch.count_nonzero(block.layers[3].weight[:, 192:]):
                raise ValueError("added channels have already changed; refusing a destructive repair")
            block.layers[0].weight[192:].copy_(initial.layers[0].weight[192:])
            block.layers[0].bias[192:].copy_(initial.layers[0].bias[192:])
            norm = block.layers[1]
            norm.weight[192:].fill_(1); norm.bias[192:].zero_()
            norm.running_mean[192:].zero_(); norm.running_var[192:].fill_(1)
            revived.append({"block": i, "hidden_channels": 320})
        actual = model(encoder_features(model, paths))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    report = {"source_sha256": hashlib.sha256(content).hexdigest(), "source_epoch": saved["epoch"] + 1,
        "seed": args.seed, "revived": revived, "max_embedding_change": float((actual - expected).abs().max()),
        "reference": "Eight random geometry probes; no evaluation data",
        "test_data_used": False, "optimizer_parameter_order_unchanged": True}
    saved["model"] = model.state_dict()
    saved["metadata"]["widening_repair"] = report
    torch.save(saved, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
