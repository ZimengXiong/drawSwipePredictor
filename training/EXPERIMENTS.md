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

`wide512` instead expands the retrieval space from 128 to 512 coordinates. It retains the completed `large` encoder's old coordinates and pads word vectors with zeros, preserving initial candidate rankings. Small nonzero added query coordinates let the new word coordinates receive gradients immediately. Word vectors must remain trainable during expansion. Distillation compares the normalized old coordinates with the frozen teacher; it leaves the added coordinates free to learn. This architecture requires a native implementation before export.

```sh
uv run --directory training python train_proxy.py \
  --init-model checkpoints/proxy-giant-strong-init.pt \
  --architecture wide512 --balance-real --stream-corpus \
  --extra-train data/memory-proxy/train-700011.bin \
  --fresh-fraction 0.5 --learning-rate 0.00005 \
  --batch 384 --epochs 180 --steps 200 --seed 900449 \
  --output checkpoints/proxy-wide512-v1
```

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

`--architecture attention` adds four attention layers around a frozen `large` encoder. Each layer compares all 64 stroke positions and learns biases from spatial distance, tangent agreement, and stroke progress. A zero-initialized correction head preserves the starting predictions. The baseline stays in evaluation mode during training, and `--freeze-proxies` also preserves the word bank. This architecture is experimental and has no native exporter yet.

```sh
uv run --directory training python train_proxy.py \
  --init-model checkpoints/proxy-giant-strong-init.pt \
  --architecture attention --freeze-proxies --stream-corpus \
  --extra-train data/memory-proxy/train-700011.bin \
  --fresh-fraction 0.5 --learning-rate 0.00005 \
  --batch 256 --epochs 180 --steps 200 \
  --seed 900409 --output checkpoints/proxy-attention-v1
```

## Compare candidate words with the stroke

Four experiments add evidence beyond a single word vector:

- `train_character.py` learns character probabilities along the stroke, then scores retrieved dictionary words with CTC. `--edit-channel` also marginalizes a forgotten interior key. Visited-key metadata supervises fresh training examples only; it never enters the decoder.
- `train_joint.py` updates the encoder, word proxies, and character decoder together. Validation recomputes features and retrieves fresh candidate lists after each epoch.
- `train_cross.py` uses cross attention between a candidate's letter sequence and the observed stroke. Training uses fresh writers and naturally retrieved hard negatives.
- `fit_development.py` fits bounded motor-error parameters for each retrieved candidate, including possible interior omissions. It measures this slower geometric method on the declared development corpus only.

These are experiments, not claims of improved benchmark accuracy. Run metadata and development histories remain beside each checkpoint. Unsuccessful pair-ranking and multi-center trials are recorded in `benchmarks/`.

`train_autoregressive.py` trains a spelling decoder around a frozen word encoder. Each next-letter prediction can use the whole stroke and earlier candidate letters. Unlike the CTC targets, its targets retain repeated letters and include an end token. Training uses teacher forcing; evaluation scores only the 128 candidates retrieved naturally from the full dictionary. A retrieval-only option protects the baseline during development selection.

```sh
uv run --directory training python train_autoregressive.py \
  --checkpoint checkpoints/proxy-giant-strong-init.pt \
  --extra-train data/memory-proxy/train-700011.bin \
  --output checkpoints/autoregressive-v1 \
  --epochs 120 --steps 200 --batch 512 --seed 900419
```

The decoder receives stroke features and candidate letters. It receives no target label, writer identity, or word rank at evaluation. Its histories distinguish the fixed development subset from the full development set. It has no native exporter yet and cannot replace the app model until native verification is implemented.

After training, use `evaluate_spelling.py --checkpoint <word-checkpoint> --decoder <spelling-checkpoint> --output <report>` to select scores on all 12,000 development drawings. The evaluator checks both checkpoint bindings and the development hash. It rejects other corpora, including the reserved confirmation set.

For a second training stage, add `--init-decoder <saved-decoder> --ranking-candidates 16 --learning-rate 0.0001 --batch 32`. It combines candidate classification with one quarter of the letter loss. Training places the labeled positive beside naturally retrieved hard negatives. Evaluation never inserts it. `--validation-seed 900419` keeps the same development subset as the original spelling run while a separate training seed supplies fresh draws. The stage saves its initial decoder score before any update, so a regression cannot replace the starting model.

## Compare many independent drawings per word

`reference_density.py` tests whether a distribution of reference drawings improves on one learned word vector. It generates the same number of independent examples for every dictionary word, including words outside the usual training-length range. Every word receives the same fixed mixture of clean, loose, and hard errors. The bank contains no development drawings.

```sh
uv run --directory training python reference_density.py build \
  --checkpoint checkpoints/proxy-giant-strong-init.pt \
  --output checkpoints/reference-density-v1 --references 32 --seed 900433
uv run --directory training python reference_density.py evaluate \
  --checkpoint checkpoints/proxy-giant-strong-init.pt \
  --bank checkpoints/reference-density-v1 \
  --output checkpoints/reference-density-development-v1.json
```

Evaluation uses the existing full-vocabulary retriever's natural top 128 candidates. It blends learned word scores with reference-kernel scores and includes the unchanged baseline. Equal-count kernel averaging prevents reference count from acting as a word prior. Selection uses all development rows and checks encoder, dictionary, and bank hashes. The bank is experimental and has no native exporter yet.

`evaluate_training_exposure.py` separately checks whether uneven real-recording exposure affects ranking. It derives class offsets from the checkpoint's hashed training recordings and recorded sampling method, using the same procedural floor for every word. It never uses word popularity, word length, or vocabulary rank. A single correction strength is selected on full development, with zero correction included. This is a limited exposure hypothesis, not an exact correction for the full training distribution.

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

Export a joint character checkpoint after its base word model:

```sh
uv run --directory training python export_retrieval.py \
  --checkpoint checkpoints/joint-v1/best.pt --output checkpoints/joint-native
uv run --directory training python export_character.py \
  --retrieval-checkpoint checkpoints/joint-v1/best.pt --output checkpoints/joint-native
cargo run --release -p swipe-engine --example retrieval_cli -- \
  --model training/checkpoints/joint-native \
  --verify-character training/checkpoints/joint-native/character-verification.json
```

For a separately trained frozen decoder, also pass `--decoder-checkpoint` to `export_character.py`. The exporter and runtime bind the decoder to the exact encoder and word bank. Verification checks character probabilities and final word rankings. Native CTC also supports marginalizing one missing interior key.

Cross-attention models are not yet exported to native inference. A selected complete system must pass native parity checks before it replaces the app's model.

## Combine complementary word models

`average_word_models.py --first <checkpoint> --second <checkpoint> --output <directory>` compares two aligned models from the same training lineage. It includes both unchanged source models as controls, then tests parameter and word-vector averages. Batch-normalization statistics are refreshed using declared training paths only. The selected checkpoint must beat the source controls on full development; otherwise the utility preserves the better source file. These single-model averages complement the prediction ensembles below.

`evaluate_complete.py` compares a joint checkpoint with an auxiliary word model on the entire declared development corpus. It retrieves all candidates naturally from the full vocabulary, selects blend settings on development data, and retains exact-word scoring.

```sh
uv run --directory training python evaluate_complete.py \
  --primary checkpoints/complete-joint-frozen.pt \
  --auxiliary checkpoints/complete-intrinsic-frozen.pt \
  --output checkpoints/complete-development-v1.json
uv run --directory training python export_retrieval.py \
  --checkpoint checkpoints/complete-joint-frozen.pt --output checkpoints/complete-native
uv run --directory training python export_ensemble.py \
  --auxiliary-checkpoint checkpoints/complete-intrinsic-frozen.pt \
  --selection checkpoints/complete-development-v1.json --output checkpoints/complete-native
uv run --directory training python export_character.py \
  --retrieval-checkpoint checkpoints/complete-joint-frozen.pt \
  --auxiliary-checkpoint checkpoints/complete-intrinsic-frozen.pt \
  --output checkpoints/complete-native
cargo run --release -p swipe-engine --example retrieval_cli -- \
  --model training/checkpoints/complete-native \
  --verify-character training/checkpoints/complete-native/character-verification.json
```

The exported decoder binds to both encoders, both word banks, and the blend weight. Changing any of them requires a matching export. These directory exports work through the CLI; the bundled app still uses the previously validated model.

## Learn faster motor proposals

`train_amortized_motor.py` learns distortion parameters from fresh synthetic drawings and candidate geometry. Synthetic visited-key labels supervise the training reconstruction loss. Recognition enumerates candidate routes and possible interior omissions without those labels.

```sh
uv run --directory training python train_amortized_motor.py \
  --output checkpoints/amortized-motor-v1 --epochs 100 --steps 200 --batch 256
uv run --directory training python fit_development.py \
  --checkpoint checkpoints/fit-encoder-frozen.pt \
  --initializer checkpoints/amortized-motor-probe-v1.pt \
  --output checkpoints/amortized-refinement-probe-v1.json \
  --samples 128 --candidates 64 --steps 30 --batch 8
```

The trainer's conditional reconstruction error is measured with the correct word supplied. It is a fitting diagnostic, not word-recognition accuracy. The separate recognition evaluator uses naturally retrieved candidates and exact-word scoring. `--steps 0` tests the learned proposal directly; positive counts refine its parameters from the observed stroke.

Motor fitting and ranking can use different penalties. `fit_development.py --optimization-regularization <value>` controls the fitting objective. `--score-regularization <value>` fixes the ranking penalty instead of selecting it from the usual grid. Supplying the same value tests a consistent prior. Defaults preserve the earlier experiments. A machine-generated report can be passed directly to `--fixed-calibration` for a disjoint development slice; it freezes ranking settings and rejects changes to the fitting penalty, checkpoint, or initializer.

`combine_motor_evidence.py` checks whether completed fits add information beyond the fixed word-and-character system. It combines naturally retrieved lists and treats missing fit evidence as neutral. A saved `--fixed-combination` freezes the selected blend weight for a different development slice. Neither this blend nor the learned initializer is part of the bundled app yet.
