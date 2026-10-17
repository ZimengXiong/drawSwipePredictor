# Work toward 80% memory recognition

The target is at least 80% exact-word first-choice accuracy on the existing hard memory benchmark, with all 60,000 candidate words. The benchmark has not been rerun during these experiments. Development scores do not establish that this target has been reached.

The bundled app still uses the previously validated model. Experimental checkpoints stay in the ignored `training/checkpoints/` directory until selection, native verification, and final evaluation are complete.

## Keep evaluation separate

[The evaluation protocol](../benchmarks/target-80-protocol.json) fixes the benchmark hash, candidate vocabulary, scoring, and model-selection rules. [The data audit](../benchmarks/target-80-data-audit.json) records the unchanged generator and scoring sources.

Training and development use different synthetic writer ranges. Target words are sampled uniformly from all eligible words in the deployment vocabulary. An additional confirmation set stays unopened until a model and its ranking settings have been selected. Neither writer identities nor target ranks are model inputs.

An evaluation shortlist contains only words retrieved from the observed stroke. A missing target counts as a miss. Training can place a labeled positive beside hard negatives; evaluation cannot. Collapsing consecutive letters supplies character supervision but never grants credit for a different final word.

## Train stronger word models

`train_proxy.py` learns a full-vocabulary classifier. It combines donated human recordings, cached synthetic writers, canonical augmentation, and new motor errors sampled on the GPU. `--architecture large` uses six residual blocks. `intrinsic` adds a rotation-invariant view. `giant` uses sixteen wider residual blocks and 10,996,624 encoder parameters, compared with 405,744 in the original neural encoder.

The larger encoder starts with the smaller encoder's predictions intact. Added channels contribute zero at initialization, and added residual blocks start as identities. Learned word proxies remain separate from the encoder parameter count.

Generate an independent training shard:

```sh
cargo run --release -p swipe-engine --example memory_corpus -- \
  --output training/data/memory-proxy/train-700011.bin \
  --samples 1000000 --seed 700011
```

Use separate seed blocks for additional shards. Each block supports at most 1,000,002 writers. The loader rejects overlapping ranges and reserved evaluation blocks.

Stream several shards without keeping separate GPU copies of the entire corpus:

```sh
uv run --directory training python train_proxy.py \
  --init-model checkpoints/proxy-intrinsic-v1/best.pt \
  --architecture giant --stream-corpus \
  --extra-train data/memory-proxy/train-700011.bin \
  --fresh-fraction 0.3 --learning-rate 0.00015 \
  --batch 384 --epochs 240 --steps 240 \
  --output checkpoints/proxy-giant-v1
```

Repeat `--extra-train` for more shards. Each run records their hashes and writer ranges. A checkpoint retains the optimizer and schedule, but does not retain every random-number generator state. Resuming is not a bit-for-bit continuation.

## Compare candidate words with the stroke

Three experiments add evidence beyond a single word vector:

- `train_character.py` learns character probabilities along the stroke, then scores retrieved dictionary words with CTC. `--edit-channel` also marginalizes a forgotten interior key. Visited-key metadata supervises fresh training examples only; it never enters the decoder.
- `train_cross.py` uses cross attention between a candidate's letter sequence and the observed stroke. Training uses fresh writers and naturally retrieved hard negatives.
- `fit_development.py` fits bounded motor-error parameters for each retrieved candidate, including possible interior omissions. It measures this slower geometric method on the declared development corpus only.

These are experiments, not claims of improved benchmark accuracy. Run metadata and development histories remain beside each checkpoint. Unsuccessful pair-ranking and multi-center trials are recorded in `benchmarks/`.

## Verify without computer use

All checks use the command line. The native exporter supports the compact, large, intrinsic, and giant word encoders. Export to a separate directory, then compare native embeddings with PyTorch reference values:

```sh
uv run --directory training python export_retrieval.py \
  --checkpoint checkpoints/proxy-giant-v1/best.pt \
  --output checkpoints/proxy-giant-native
cargo run --release -p swipe-engine --example retrieval_cli -- \
  --model training/checkpoints/proxy-giant-native \
  --verify training/checkpoints/proxy-giant-native/verification.json
uv run --directory training python -m unittest test_proxy test_retrieval
cargo test --workspace --all-targets
```

Native word-encoder support does not yet export the experimental character or cross-attention decoders. A selected complete system must pass its own native parity checks before it replaces the app's model.
