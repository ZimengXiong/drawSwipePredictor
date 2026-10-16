# Train the memory recognizer

The retrieval model learns to match a drawing to a word's keyboard route. Training combines real human recordings, synthetic drawings, and new distortions generated on the GPU. You do not need to record or label gestures yourself.

Run these commands from the repository root. Install Rust, Python, and [uv](https://docs.astral.sh/uv/) first.

## Get real recordings

[FUTO's swipe dataset](https://huggingface.co/datasets/futo-org/swipe.futo.org) contains donated human swipe recordings under the MIT license. The downloader pins a dataset revision and preserves the source's train, validation, and test splits. Those splits have separate donor sessions.

```sh
python3 training/fetch_real_swipes.py \
  --train 150000 --validation 10000 --test 5000 --vocab 30000 \
  --output-dir training/data/futo-large
```

The downloader reads bounded byte windows instead of downloading the full 5 GB training file. It caches those windows so an interrupted download can resume. This sampling method groups nearby records; it does not produce a uniform sample of the entire corpus.

The converted records contain a target word, ordered screen coordinates, relative timestamps, and an anonymous session identifier. Conversion restores the canvas's physical aspect ratio. Filters require a moving, finite path and a target in the selected vocabulary. The downloader does not filter records by how closely they match a word template.

These recordings use visible keyboards. They provide human movement data, but do not establish accuracy for freehand gestures drawn entirely from memory.

## Generate synthetic drawings

```sh
python3 training/generate_memory_data.py
```

This creates 18,000 training drawings, 3,000 validation drawings, and 3,000 test drawings in `training/data/memory/`. Target words and writer identities are disjoint across these splits. Each writer draws the same target at clean, loose, and memory difficulty levels within its split.

The generator models persistent errors in remembered key locations and row offsets. Individual attempts add rounded corners, occasional omitted waypoints, endpoint errors, drift, and uneven acquisition speed. Dataset manifests record seeds, target ranges, and file hashes.

## Train on the GPU

```sh
uv sync --directory training
uv run --directory training python train_retrieval.py \
  --vocab 60000 --epochs 40 --steps 100 --batch 512 \
  --output checkpoints/retrieval-v1
```

Apple Silicon uses PyTorch's MPS backend by default. Use `--device cuda` or `--device cpu` on another supported machine.

Each batch includes real recordings, generated drawings, and augmented canonical routes. The shared encoder learns a 128-dimensional vector. Contrastive training compares the intended word against other words, including hard negatives mined from the current model. Multiple examples of the same word share a positive label.

The GPU augmentation changes rotation, axis stretch, remembered key locations, corner rounding, and endpoints. It can replace a short section with a chord to simulate a missed part of the route. Arc-length resampling removes differences in acquisition speed afterward.

Validation uses separate donors and fresh synthetic drawings. Checkpoint selection averages first-choice accuracy across recorded real, distorted real, and synthetic validation groups. The trainer does not read test files.

Continue with more real recordings:

```sh
uv run --directory training python train_retrieval.py \
  --init-model checkpoints/retrieval-v1/best.pt \
  --vocab 60000 --epochs 120 --steps 150 --batch 512 \
  --real-fraction 0.5 --learning-rate 0.00015 \
  --output checkpoints/retrieval-v2
```

`--init-model` starts a new optimization schedule from learned weights. `--resume` restores the previous optimizer and schedule instead. Each run saves its arguments, source hashes, validation history, and best checkpoint.

## Calibrate and export

Calibration combines canonical word vectors with consensus vectors from real training drawings. It chooses the combination weight, word-frequency prior, and optional orientation ensemble using validation data.

```sh
uv run --directory training python calibrate_retrieval.py \
  --checkpoint checkpoints/retrieval-v2/best.pt \
  --output checkpoints/retrieval-v2/calibration.json

uv run --directory training python export_retrieval.py \
  --checkpoint checkpoints/retrieval-v2/best.pt \
  --calibration checkpoints/retrieval-v2/calibration.json \
  --output ../crates/swipe-engine/models/retrieval
```

Export folds batch normalization into convolution weights and caches word vectors. The Rust runtime needs no Python or ONNX installation. The encoder and word bank carry a shared model fingerprint to detect mismatched files.

Check native inference against PyTorch:

```sh
cargo run --release -p swipe-engine --example retrieval_cli -- \
  --model crates/swipe-engine/models/retrieval \
  --verify crates/swipe-engine/models/retrieval/verification.json
```

## Keep the original CTC experiment

The original letter-decoding model remains available:

```sh
uv run --directory training python train_ctc.py \
  --words ../word_freq.txt --output swipe_ctc.onnx
cargo run --release -p swipe-desktop --features neural --bin swipe-ctc
```

Its training generator and preprocessing differ from the retrieval pipeline. Existing CTC results do not measure the new retrieval model.
