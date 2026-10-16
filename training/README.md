# Train the memory recognizer

The retrieval model learns to match a drawing to a word's keyboard route. Training combines real human recordings, synthetic drawings, and new distortions generated on the GPU. You do not need to record or label gestures yourself.

Run these commands from the repository root. Install Rust, Python, and [uv](https://docs.astral.sh/uv/) first.

## Get real recordings

[FUTO's swipe dataset](https://huggingface.co/datasets/futo-org/swipe.futo.org) contains donated human swipe recordings under the MIT license. The downloader pins a dataset revision and preserves the source's train, validation, and test splits. Those splits have separate donor sessions.

```sh
python3 training/fetch_real_swipes.py \
  --train 50000 --validation 5000 --test 5000 --vocab 30000 \
  --output-dir training/data/futo-large
```

The downloader reads bounded byte windows instead of downloading the full 5 GB training file. It caches those windows so an interrupted download can resume. This sampling method groups nearby records; it does not produce a uniform sample of the entire corpus.

The converted records contain a target word, ordered screen coordinates, relative timestamps, and an anonymous session identifier. Conversion restores the canvas's physical aspect ratio. Filters require a moving, finite path and a target in the selected vocabulary. The downloader does not filter records by how closely they match a word template.

These recordings use visible keyboards. They provide human movement data, but do not establish accuracy for freehand gestures drawn entirely from memory.

## Generate synthetic drawings

```sh
python3 training/generate_memory_data.py
```

This creates 18,000 training drawings, 3,000 validation drawings, and 3,000 test drawings in `training/data/memory/`. Target words and writer identities are disjoint across these dataset files. Online canonical augmentation still covers the deployment vocabulary, so this does not constitute a word holdout for the complete neural training procedure. Each writer draws the same target at clean, loose, and memory difficulty levels within its split.

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

Expand the real dataset and train the final pass used by the bundled model:

```sh
python3 training/fetch_real_swipes.py \
  --train 150000 --validation 10000 --test 5000 --vocab 30000 \
  --output-dir training/data/futo-large

uv run --directory training python train_retrieval.py \
  --init-model checkpoints/retrieval-v2/best.pt \
  --vocab 60000 --epochs 80 --steps 150 --batch 768 \
  --validation-samples 2000 --real-fraction 0.5 --learning-rate 0.0001 \
  --output checkpoints/retrieval-v3
```

`--init-model` starts a new optimization schedule from learned weights. `--resume` restores the previous optimizer and schedule instead. Each run saves its arguments, source hashes, validation history, and best checkpoint.

## Calibrate and export

Calibration combines canonical word vectors with consensus vectors from real training drawings. It chooses the combination weight, word-frequency prior, and optional orientation ensemble using validation data.

```sh
uv run --directory training python calibrate_retrieval.py \
  --checkpoint checkpoints/retrieval-v3/best.pt --samples 2000 \
  --output checkpoints/retrieval-v3/calibration.json

uv run --directory training python export_retrieval.py \
  --checkpoint checkpoints/retrieval-v3/best.pt \
  --calibration checkpoints/retrieval-v3/calibration.json \
  --output ../crates/swipe-engine/models/retrieval
```

Export folds batch normalization into convolution weights and caches word vectors. The Rust runtime needs no Python or ONNX installation. The encoder and word bank carry a shared model fingerprint to detect mismatched files.

Check native inference against PyTorch:

```sh
cargo run --release -p swipe-engine --example retrieval_cli -- \
  --model crates/swipe-engine/models/retrieval \
  --verify crates/swipe-engine/models/retrieval/verification.json
```

## Train the geometry reranker

The second stage combines 32 neural candidates with 32 additional candidates from an independent shape search. It measures each candidate using elastic alignment at five rotations, sharp and rounded templates, endpoints, and route complexity. A small residual network learns how to combine these measurements with the neural distance and word-frequency prior.

Generate a wider pool of memory drawings, then extract features with the native runtime:

```sh
python3 training/generate_memory_data.py \
  --train 30000 --validation 2000 --test 2000 --vocab 60000 \
  --output-dir training/data/memory-wide
cargo build --release -p swipe-engine --example retrieval_cli

uv run --directory training python prepare_reranker.py \
  --split train --samples 24000 --workers 8 \
  --output data/reranker-union/train.npz
uv run --directory training python prepare_reranker.py \
  --split validation --samples 2000 --workers 2 \
  --output data/reranker-union/validation.npz

uv run --directory training python train_reranker.py \
  --train data/reranker-union/train.npz \
  --validation data/reranker-union/validation.npz \
  --output checkpoints/reranker-union-v2 --epochs 100
```

The training set contains 96,000 query lists: 24,000 recordings, distorted views of those recordings, 24,000 procedural drawings, and 24,000 augmented canonical routes. Each list has 64 candidate words. Training uses a listwise loss over lists that contain the target; candidate recall is reported separately. The validation set has 2,000 examples from each source. Checkpoint selection averages first-choice accuracy across those four validation groups. No test data enters fitting or selection.

Copy the selected `reranker.json` into the exported model directory. The file includes fingerprints of its encoder and word bank. Loading mismatched files fails. Exporting a new base retriever clears old reranker weights; fit the second stage again afterward.

Verify native candidate scoring against PyTorch:

```sh
uv run --directory training python verify_reranker.py \
  --model ../crates/swipe-engine/models/retrieval \
  --output ../crates/swipe-engine/models/retrieval/reranker-verification.json
```

## Freeze a new donor test

The second-stage experiment reserves donors absent from every file in the earlier real dataset, including its test file. The comparison uses the full 60,000-word vocabulary.

```sh
python3 training/fetch_real_swipes.py \
  --train 1 --validation 1 --test 8000 --vocab 60000 --seed 628319 \
  --output-dir training/data/futo-fresh
uv run --directory training python evaluate_reranker.py --freeze
```

Freeze the test before inspecting predictions. The freeze command takes up to ten recordings per new donor, shuffles them, and caps the set at 1,000 recordings. It adds fixed distorted versions of those traces and 6,000 synthetic test drawings. It refuses to overwrite an existing frozen test.

Keep a copy of the base model directory without `reranker.json`, then compare it with the directory containing the selected ranker:

```sh
uv run --directory training python evaluate_reranker.py \
  --baseline checkpoints/retrieval-v3-native \
  --challenger checkpoints/reranker-union-native \
  --output ../benchmarks/reranker-fresh-test.json
```

The evaluator reports paired accuracy differences and 95% intervals from resampling donor sessions or synthetic writers. Timing includes the complete native prediction path and varies with machine load.

## Keep the original CTC experiment

The original letter-decoding model remains available:

```sh
uv run --directory training python train_ctc.py \
  --words ../word_freq.txt --output swipe_ctc.onnx
cargo run --release -p swipe-desktop --features neural --bin swipe-ctc
```

Its training generator and preprocessing differ from the retrieval pipeline. Existing CTC results do not measure the new retrieval model.
