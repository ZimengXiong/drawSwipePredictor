#!/usr/bin/env python3
"""Export a trained route encoder and vocabulary for dependency-free Rust inference."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from retrieval_model import GestureEncoder, build_encoder, encoder_features, LAYOUT, curve, features, vocabulary

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return value


def floats(tensor) -> bytes:
    return tensor.detach().cpu().numpy().astype("<f4").tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["metadata"]["arguments"].get("architecture", "small")
    if architecture not in ["small", "large", "intrinsic", "giant", "wide512", "massive"]:
        raise ValueError(f"{architecture} requires a native implementation before export")
    model = build_encoder(architecture).eval()
    model.load_state_dict(checkpoint["model"])
    encoded = bytearray({"small": b"SWPRT001", "large": b"SWPRT002", "intrinsic": b"SWPRT003", "giant": b"SWPRT004",
                         "wide512": b"SWPRT005", "massive": b"SWPRT006"}[architecture])
    # Fold normalization into convolution weights; the native runtime uses only
    # convolution, ReLU, dense layers, and vector normalization.
    def convolution(conv, norm):
        factor = norm.weight / (norm.running_var + norm.eps).sqrt()
        encoded.extend(floats(conv.weight * factor[:, None, None]))
        encoded.extend(floats((conv.bias - norm.running_mean) * factor + norm.bias))
    for i in range(0, len(model.stages), 3):
        convolution(model.stages[i], model.stages[i + 1])
    if architecture in ["large", "intrinsic", "giant", "wide512", "massive"]:
        for block in model.blocks:
            convolution(block.layers[0], block.layers[1])
            convolution(block.layers[3], block.layers[4])
        convolution(model.downsample[0], model.downsample[1])
    for i in [1, 4]:
        encoded.extend(floats(model.head[i].weight))
        encoded.extend(floats(model.head[i].bias))
    args.output.mkdir(parents=True, exist_ok=True)
    limit = args.vocab or len(checkpoint["words"])
    words, counts, paths = vocabulary(ROOT / "word_freq.txt", limit)
    model = model.to(args.device)
    if "proxies" in checkpoint:
        if checkpoint["words"] != words:
            raise ValueError("learned proxy word order differs from the deployment vocabulary")
        vectors = torch.nn.functional.normalize(checkpoint["proxies"], dim=-1).numpy()
        if limit != len(vectors):
            raise ValueError("learned proxy vocabulary must match the exported vocabulary")
    else:
        vectors = []
        with torch.no_grad():
            for start in range(0, len(words), 1024):
                values = torch.tensor(paths[start:start + 1024], device=args.device)
                vectors.append(model(encoder_features(model, values)).cpu().numpy())
        vectors = np.concatenate(vectors)
    calibration = None
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())
        if calibration["checkpoint_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            raise ValueError("calibration belongs to a different checkpoint")
        saved = np.load(args.calibration.with_suffix(".npz"), allow_pickle=False)
        if saved["words"].tolist() != words:
            raise ValueError("calibrated vocabulary differs from exported vocabulary")
        vectors = saved["vectors"]
    centers = vectors.shape[1] if vectors.ndim == 3 else 1
    if vectors.shape[-1] != getattr(model, "embedding_dim", 128):
        raise ValueError("word-vector width differs from the encoder output")
    bank = bytearray(b"SWPBK002" if centers > 1 else b"SWPBK001")
    bank.extend(struct.pack("<QII", fingerprint(encoded), len(words), vectors.shape[-1]))
    if centers > 1:
        bank.extend(struct.pack("<I", centers))
    popularity = np.log1p(counts) / np.log1p(counts.max())
    for word, pop, vector in zip(words, popularity, vectors):
        name = word.encode("ascii")
        bank.extend(struct.pack("<H", len(name)))
        bank.extend(name)
        bank.extend(struct.pack("<f", float(pop)))
        bank.extend(vector.astype("<f4").tobytes())
    verification = []
    with torch.no_grad():
        for word in ["hello", "world", "keyboard", "lunchbox", "alpacas"]:
            raw = np.asarray([LAYOUT[c] for c in word], dtype=np.float64)
            raw[:, 1] *= 1.4
            angle = 0.23
            raw = raw @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            raw = raw * 75 + [420, 190]
            expected = model(encoder_features(model, torch.tensor(curve(raw)[None], device=args.device))).cpu().tolist()[0]
            verification.append({"word": word, "points": raw.tolist(), "embedding": expected})
    training_metadata = json.loads(json.dumps(checkpoint["metadata"]).replace(str(ROOT) + "/", ""))
    report = {
        "schema_version": 1, "architecture": architecture,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "points": 64, "channels": getattr(model, "input_channels", 8), "embedding": vectors.shape[-1], "vocabulary": len(words), "route_centers": centers,
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "validation_score": checkpoint.get("retriever_validation", {}).get("memory", {}).get("top1", checkpoint["best_score"]),
        "validation_score_includes_character": "character_model" in checkpoint and "retriever_validation" not in checkpoint,
        "encoder_sha256": hashlib.sha256(encoded).hexdigest(),
        "words_sha256": hashlib.sha256(bank).hexdigest(),
        "training": training_metadata,
        "prior_weight": calibration["selected"]["prior"] if calibration else (0.0 if "proxies" in checkpoint else 0.025),
        "tta_degrees": calibration["selected"]["tta_degrees"] if calibration else 0.0,
        "adaptive_prior": calibration["selected"].get("prior_form") == "adaptive" if calibration else False,
        "calibration": calibration["selected"] if calibration else None,
        "calibration_sources": calibration.get("source_sha256") if calibration else None,
        "data_source": "https://huggingface.co/datasets/futo-org/swipe.futo.org",
        "data_license": "MIT",
        "word_frequency_sha256": hashlib.sha256((ROOT / "word_freq.txt").read_bytes()).hexdigest(),
    }
    # A new export is a base retriever. Candidate weights require a separate fit.
    for filename in ["reranker.json", "reranker-verification.json", "character.bin", "character-verification.json"]:
        (args.output / filename).unlink(missing_ok=True)
    outputs = {"encoder.bin": bytes(encoded), "words.bin": bytes(bank),
               "verification.json": (json.dumps(verification, indent=2) + "\n").encode(),
               "manifest.json": (json.dumps(report, indent=2) + "\n").encode()}
    for filename, content in outputs.items():
        temporary = args.output / (filename + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(args.output / filename)
    print(f"Exported {len(words):,} word routes and {len(encoded):,} bytes of encoder weights")


if __name__ == "__main__":
    main()
