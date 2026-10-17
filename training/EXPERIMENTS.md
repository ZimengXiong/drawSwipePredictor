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

`wide512` instead expands the retrieval space from 128 to 512 coordinates. It retains the completed `large` encoder's old coordinates and pads word vectors with zeros, preserving initial candidate rankings. Small nonzero added query coordinates let the new word coordinates receive gradients immediately. Word vectors must remain trainable during expansion. Distillation compares the normalized old coordinates with the frozen teacher; it leaves the added coordinates free to learn. Native export retains all 512 coordinates. Reference checks verify embeddings and full-vocabulary word costs before a wider model can enter a complete native mixture.

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

Repeat `--extra-train` for more shards. Each run records their hashes and writer ranges. New checkpoints retain normalized inference vectors and raw word-vector parameters, together with the optimizer and schedule. Older files contain only normalized vectors, so their original raw training scale cannot be recovered. Every random-number generator state is not retained; full training resumes are still not bit-for-bit continuations.

`--proxy-learning-rate` controls word-vector updates separately from encoder updates. Its default remains four times `--learning-rate`. Run metadata records actual starting optimizer rates, raw-vector preservation, and legacy normalized resumes. The raw-state regression test checks that a serialized optimizer and its word vectors produce the same next update while preserving inference scores.

`--synthetic-only` removes the real-recording and augmented canonical portions of each training batch. Cached writers fill those positions instead. The selected fresh-writer fraction, if any, still applies. Generated routes retain their original errors without a second augmentation. The synthetic pool includes its unchanged clean, loose, and memory proportions; this option does not filter difficulty or target words. Resume rejects a changed mixture setting. The controlled synthetic-only stage keeps the smaller exact-data control's initialization, synthetic shards, learning rates, seed, batch size, and schedule. Its effect on accuracy remains unproven.

`--architecture massive` uses sixteen residual blocks with 2,048 hidden channels per block: 36,236,176 encoder parameters and a separate 7,680,000-parameter word bank. It can expand the completed `large` model while preserving its initial predictions. Added channels receive nonzero gradients in the regression test. A training stage must improve development accuracy before the larger size counts as a gain. Its native export passes canonical embedding, score, and top-five checks for the initializer; these checks do not establish a trained recognition gain. The exact-data experiment uses the same starting checkpoint and corpus as the smaller low-rate control, with smaller batches to limit activation memory. The architecture and batch size both differ.

`inspect_capacity.py` measures residual activations and gradients on 32 fixed training strokes. It checks the training corpus hash against the checkpoint and performs no parameter updates. At epoch 10, the massive model's ten added blocks produced residual signals about 0.1% of their input signals. Their gradients were finite. The original six blocks produced residual signals about 30–42% of their inputs. This suggests testing how quickly the added output gates learn; it does not establish why development accuracy stalled.

`--new-block-gate-learning-rate` sets a separate rate for the final normalization weights and biases in blocks 7–16. Other encoder parameters and word vectors retain their own rates. The controlled massive-gate run uses the original initialization, data, seed, batches, and schedule, with gate updates starting at 0.001 instead of 0.00001. The original massive run remains the control. Resume rejects a changed gate-rate setting. The diagnostic and the experiment use training and development writers only.

After the first gate-training epoch, the added residual signals measure 2.1–2.7% of their inputs. This confirms that they contribute more signal; the earlier control measurement used epoch 10, so the activation comparison has unequal training budgets. Hard development accuracy is 72.02%, below the preserved 72.32% initializer. The larger model has not established an accuracy gain.

`covariance_density.py build --references <reference-bank> --output <directory>` fits word-specific error covariance from the equal-count synthetic reference bank. A shared variance scale and every word parameter use references only. The factorization avoids storing a full inverse for each word. A regression test compares its Gaussian costs with direct matrix solves and determinants. `evaluate --checkpoint <word-model> --bank <directory> --output <report>` compares regularized covariance scores with the unchanged retrieval control on all development rows. Candidates come from the natural full-vocabulary top 128. This is an experimental ranking model; its costs are not calibrated probabilities and it is not exported to the app.

`diagnose_generalization.py` compares fixed hard examples from declared training pools with all hard development writers. Models remain in evaluation mode and retain their weights and normalization statistics. A training-pool score is diagnostic, not held-out accuracy. The first comparison found about an eight-point gap for the original one-million-writer model and a four-point gap after broader training.

`build_writer_shards.py --output-dir data/memory-proxy-exact-v2 --start-seed 700021 --shards 20 --workers 4` adds 20 million independent writers using the unchanged Rust generator. It checks existing, development, and confirmation intervals before generation, validates every completed shard, and records source and data hashes. Existing shards are reused only when their build receipts and contents match. Pending output requires inspection of the original builder before restarting. The new pool does not change the benchmark or existing corpus files.

The `proxy-exact-v1` stage starts from the completed balanced model and uses all 31 million cached writers, with `--fresh-fraction 0`. Real-recording and canonical augmentation remain in the training mixture. This tests a combined change in writer diversity and sampler fidelity; it does not isolate either factor. Its initial development score is preserved, and a gain from the data change remains unproven.

`fetch_additional_real_swipes.py` imports all eligible records from pinned FUTO collections 2–4, including the collection of confusable QWERTY words. It verifies each complete source file against its published SHA256, excludes known validation and test session IDs, and retains the original training file unchanged in a separate combined dataset. It applies finite-geometry, timestamp, and full-vocabulary checks; it does not filter by template distance or model score. The first import adds 73,550 recordings, bringing training to 223,550 recordings and 23,958 word types. Visible-keyboard recordings do not establish freehand-memory accuracy, and separate session IDs do not prove separate human identities. `train_proxy.py` and `train_autoregressive.py` accept `--real-train <combined-train.jsonl>` for an explicit data experiment; existing runs keep their original data.

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

The decoder receives stroke features and candidate letters. It receives no target label, writer identity, or word rank at evaluation. Its histories distinguish the fixed development subset from the full development set. Native spelling export supports one to four word-retrieval models, with a separately bound context encoder.

`--decoder-layers` expands a warm decoder to six, nine, or twelve blocks. Added blocks start with zero attention and feed-forward outputs, preserving the original spelling scores while allowing their output projections to learn immediately. Checkpoint loading infers the full depth from tensor names and rejects missing blocks or shrinking. The nine-block decoder has 5,422,109 parameters. Its initializer passes all five native spelling-cost and exact top-five references. Legacy three-block serialization remains byte-identical; deeper exports use `SWPSP002` with an explicit bounded depth. Initialization parity is not an accuracy gain.

The first trained nine-block checkpoint reaches 6,138/8,400, or 73.07%, on full hard development with the existing word mixture. Compared with the verified three-block system, it gains seven hard, two clean, and eight loose first choices, while losing two hard top-five hits. Its trained weights pass five native reference cases and seven corruption checks. A point-only native replay of all 12,000 development strokes reproduces every Python top-one and top-five count. Files and executable hashes remain unchanged throughout replay. Timing under concurrent training is 217 ms median and 416 ms p95. The preview retains the verified 72.99% model.

Later immutable wide/synthetic-only checkpoints reach 6,141/8,400 hard word-only first choices. Their nine-block complete system reaches 6,171/8,400 (73.46%), with length power 0.3 and proxy weight 4.0. It passes five native reference cases and four binding checks. Full point-only native replay reproduces every count across all 12,000 development rows. The benchmark and confirmation set stay closed.

`--decoder-width 384 --decoder-layers 12` expands the trained decoder to 28,560,797 parameters. Channel replication, rescaled attention, and identity added blocks retain the initialized likelihoods. Five full-size Python references differ by at most 0.000027, and native references by at most 0.000039. Native `SWPSP003` records bounded width and depth; legacy formats remain supported. These initialization probes do not establish trained accuracy. Initial native timing is about 700–800 ms per reference query under concurrent work.

The larger stage uses the frozen nine-block checkpoint, independent wide-model hard negatives, 31 million training writers, and candidate-ranking supervision. Causal prediction discards unused trailing padding while retaining every candidate. This was tested against complete padded word costs. The first stage was interrupted during initialization, before an epoch history or latest checkpoint existed, then restarted with this optimization. Its report preserves the reason and the replacement stage name.

For ranking training, `--retrieval-checkpoint` supplies an independent frozen word model for negative mining and natural development candidates. The context encoder remains separately bound. The independent model must have the same complete vocabulary, training corpus hash, and development corpus hash, with no test data used. Training positives remain labeled supervision; evaluation never inserts the target. This lets a deeper decoder learn against a stronger recognizer's confusions rather than the context encoder's older word predictions.

After training, use `evaluate_spelling.py --checkpoint <word-checkpoint> --decoder <spelling-checkpoint> --output <report>` to select scores on all 12,000 development drawings. The evaluator checks both checkpoint bindings and the development hash. It rejects other corpora, including the reserved confirmation set.

Add `--retrieval-checkpoint <word-checkpoint>` to test the decoder with candidates from a stronger word model. The spelling features still come from the decoder's bound encoder. The separate retrieval model must use the same vocabulary and declared development corpus. Its own natural top-128 list supplies every candidate and the retrieval-only control. The report records all three checkpoint hashes. No evaluation target is added to the list.

For an already selected pair of word models, also pass `--retrieval-auxiliary <checkpoint> --retrieval-selection <report>`. This preserves the report's fixed mixture instead of tuning another retrieval weight. Both source hashes, the complete vocabulary, and the development hash must match. The evaluator rejects selections that used test data or would drop a component. It then selects spelling scores around the pair's naturally retrieved candidates and retains the pair-only control.

For a larger fixed mixture, pass each active `--word-checkpoint` and its `--word-selection` report. The evaluator requires every selected source exactly once, checks all hashes and development splits, and supports independently sized word embeddings. It retains the complete word-only control while selecting spelling scores. `evaluate_proxy_ensemble.py` includes all quarter-weight mixtures, including three-component mixtures, plus the equal-weight mixture. A selected mixture must have a matching native implementation before promotion; a two-model export cannot silently discard extra components.

The sparse search can compare six candidate checkpoints while selecting at most four active components. `evaluate_uniform_words.py` separately tests the equal-weight average without that component restriction. The six-model control reaches 6,123/8,400 hard word-only first choices, compared with 6,122 for the sparse search's wide/synthetic-only pair. Both complete spelling combinations are evaluated before selection. A winning larger mixture requires native support for every active component. The multiplicative uniform formulation matches spelling selection; an earlier scalar-division control has identical aggregate hit counts and remains recorded.

`export_spelling.py` adds a selected decoder to a separate native word-model directory. Supply `--context-checkpoint`, `--retrieval-checkpoint`, `--decoder`, `--selection`, and `--output`; a word mixture also needs every auxiliary checkpoint through `--word-checkpoint`. The legacy pair export also accepts `--auxiliary-checkpoint`. The export retains the decoder's original encoder alongside the retrieval encoders. Its binary binds to that encoder, the complete word bank, all retrieval encoders, and their selected mixture weights. It rejects stale selections and directories containing other rankers. The bundled app is not changed by a directory export.

`verify_spelling.py --model <directory> --output <report>` checks artifact hashes, full spelling costs, and exact top-five rankings against Python references. It also tests rejection of missing decoder weights, changed decoder weights, a changed context encoder, and a changed retrieval mixture. References include repeated-letter words. Normal CLI requests contain points only. The native decoder uses causal attention, preserves the end token, and scores only the natural top 128 candidates. On macOS, dense layers use Accelerate; other platforms use the portable implementation. `evaluate_native_development.py` can then replay all 12,000 raw development drawings with the spelling selection report. Canonical parity does not establish full-development accuracy.

`python3 scripts/build_memory_app.py --model <native-directory> --output "build/Draw from Memory Preview.app" --bundle-id com.zimeng.draw-swipe-preview` packages a selected export into a separate macOS app. The build checks source and copied models through the native CLI, verifies declared hashes, and includes every auxiliary encoder and decoder. Training checkpoints are excluded. A receipt in `Contents/Resources/model-receipt.json` records all copied files. The app loads `Contents/Resources/model`; missing packaged models fail instead of falling back to the compiled recognizer. `DRAW_SWIPE_MODEL_DIR` remains an explicit override. Command-line binaries outside an app bundle retain the compiled default. Package verification can use `verify_spelling.py` against the copied resource directory without opening a window.

For a second training stage, add `--init-decoder <saved-decoder> --ranking-candidates 16 --learning-rate 0.0001 --batch 32`. It combines candidate classification with one quarter of the letter loss. Training places the labeled positive beside naturally retrieved hard negatives. Evaluation never inserts it. `--validation-seed 900419` keeps the same development subset as the original spelling run while a separate training seed supplies fresh draws. The stage saves its initial decoder score before any update, so a regression cannot replace the starting model.

Add `--fine-tune-encoder --encoder-learning-rate 0.00001` to update stroke features jointly with the initialized spelling decoder. A separate frozen copy of the original word model supplies training candidates and the retrieval control. Encoder running statistics stay fixed while gradients update its weights. Each checkpoint stores both learned components; evaluation and native export load the learned context from that checkpoint. The reported context-state hash identifies the joint file, while the original encoder hash identifies its initialization and retrieval binding. `--fresh-fraction 0` draws synthetic training examples from declared exact-generator shards; the default retains the original fresh-sampler mixture. Full-development selection and native checks remain required before promotion.

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

`prepare_complete_geometry.py` mines a second-pass geometry experiment around the complete spelling recognizer. Its 20 scalar inputs combine learned scores with independent route alignment. Word frequency is replaced with zero. Candidate identities and labels are stored only for loss and scoring; writer identities and difficulty never enter prediction requests. Every candidate comes from the observed stroke's natural top 128.

The separate `replay_memory_writers` data generator reconstructs raw strokes from training records. Twelve replay cases reproduce the original generator's points and corpus records exactly. Feature extraction ignores supplied label metadata and preserves baseline scores and candidate order. Mining checks every regenerated normalized curve, freezes private executable copies, and verifies all model-file hashes. Training excludes reserved and overlapping writer blocks. Development must keep every declared row and reproduce the existing native top-one and top-five counts before its cache is accepted.

The production experiment uses 50,000 training writers sampled from the 31-million-writer pool and all 12,000 development rows. `train_complete_geometry.py` trains a small shared residual network on those cached features. It derives a query-relative geometry margin from alignment costs, uses training-only normalization, and includes zero correction during selection. Missing targets remain misses in evaluation. CPU training is the default. The selected correction reaches 6,142/8,400 hard first choices (73.12%), eleven above the base system, with one fewer clean hit and ten more loose hits. Its native export passes five reference cases and fourteen corruption or binding checks. Full point-only native replay now reproduces every training count across all 12,000 development rows. Files remain unchanged throughout replay. Timing under concurrent training is 87 ms median and 105 ms p95. The later mature-pair spelling system scores higher, so this correction is retained as a verified comparison.

```sh
uv run --directory training python train_complete_geometry.py \
  --train data/complete-geometry-v1/train.npz \
  --development data/complete-geometry-v1/development.npz \
  --model checkpoints/exact-wide-complete-native-v1 \
  --output checkpoints/complete-geometry-ranker-v1
```

All checks use the command line. The native exporter supports the compact, large, intrinsic, giant, wide512, and massive word encoders. Word-bank widths must match their encoder outputs. `verify_word_model.py --checkpoint <checkpoint> --model <directory> --output <report>` also verifies exact top-five words and costs over the full dictionary. Export to a separate directory, then compare native embeddings with PyTorch reference values:

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

`average_word_models.py --first <checkpoint> --second <checkpoint> --output <directory>` compares two aligned models from the same training lineage. It includes both unchanged source models as controls, then tests parameter and word-vector averages. Optional `--bn-batches` refreshes normalization statistics using declared training paths only. This is off by default: the cached-only recalibration population reduced accuracy in the initial experiment. The selected checkpoint must beat the source controls on full development; otherwise the utility preserves the better source file.

`evaluate_proxy_ensemble.py` compares prediction mixtures on full development. When exactly two word models receive positive weights, its report can be passed directly to `export_ensemble.py`. The exporter requires both selected components and rejects a report that would silently discard a third model. `verify_word_ensemble.py --primary <checkpoint> --auxiliary <checkpoint> --model <export-directory> --output <report>` checks artifact hashes, exact top-five words, and word costs against the native CLI. Only points are sent to native inference.

`export_word_mixture.py` exports two to four active components, including mixtures of 128- and 512-coordinate models. Supply every selected component exactly once. Export the primary checkpoint first, then pass the remaining checkpoints with repeated `--component` arguments. Component order must match the selected Python calculation. The runtime retains every component's full word bank and rejects missing files, changed bindings, or different vocabulary order.

For the three-model development candidate:

```sh
uv run --directory training python export_word_mixture.py \
  --primary checkpoints/completed-intrinsic-v1.pt \
  --component checkpoints/exact-low-probe-v1.pt \
  --component checkpoints/wide512-probe-v1.pt \
  --selection checkpoints/exact-wide-word-ensemble-development-v2.json \
  --output checkpoints/exact-wide-complete-native-v1
```

When exporting its spelling decoder, pass every auxiliary source with repeated `--word-checkpoint` arguments to `export_spelling.py`. The context encoder has its own checkpoint binding. The decoder binds all word-model encoder hashes, bank hashes, and weights. `verify_spelling.py` checks the canonical costs and exact top-five rankings, then rejects missing or changed components and changed mixture weights. The three-model candidate passes five canonical cases and seven corruption checks.

The candidate also passes the full 12,000-row native replay. Hard development accuracy is 6,131/8,400, or 72.99%. Native top-one and top-five counts match Python in all three difficulty groups. Compared with the previous native candidate, it gains 22 hard first choices and loses six clean and seven loose first choices. Median native inference is 97.51 ms and p95 is 112.67 ms during concurrent training. The regenerated corpus hash matches the original. The executable, model manifest, and inference files match the canonical verification and post-replay checks. This run predates the evaluator's automatic start/end mutation guards; its additional checks are recorded separately in `benchmarks/exact-wide-native-full-development-v1.json`. Neither the benchmark nor the blind confirmation set was evaluated.

`evaluate_native_development.py --model <export-directory> --python-report <selection-report> --output <report>` replays every development drawing through the native CLI. It regenerates raw strokes from the declared development seed and requires the resulting binary corpus hash to match the original. `memory_corpus --verification-limit <count>` controls raw-reference export only; it changes neither generated drawings nor corpus bytes. Native requests contain points only. The evaluator keeps every row and requires exact agreement with Python's top-one and top-five hit counts in each difficulty group. The confirmation corpus is rejected by the development-hash guard.

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

## Learn the keys actually visited

`label_memory_routes` exports training targets for the unchanged generator's first draw from each independent writer. It reproduces consecutive-letter collapse and the optional interior omission. It retains duplicate visits created by deleting an intervening key. The exporter consumes the generator's fixed initialization draws without changing `synthetic.rs`.

`prepare_route_labels.py` samples 260,000 writers without replacement across the audited 31-million-writer training pool. It rejects reserved and overlapping writer intervals. Its output contains normalized stroke points and visited-key targets. Of these writers, 250,000 supply training updates and 10,000 form an internal holdout. No development or benchmark strokes enter this training stage.

`train_visited_routes.py` expands the existing spelling decoder to six blocks. It selects checkpoints using token loss on the internal training-writer holdout. This loss measures route prediction, not complete-word recognition. The query encoder receives points only; visited-key tokens supervise the autoregressive training objective.

`evaluate_visited_routes.py` retrieves 128 candidates from all 60,000 words, then scores each candidate's collapsed route and possible single interior omissions. It marginalizes route probabilities using a fixed omission prior of 0.2835 from the training difficulty mixture. Prediction receives no target word, visited route, true route count, writer identity, or difficulty. Exact complete-word scoring retains missing targets as misses. A 512-row development diagnostic must not be compared directly with the full 8,400-row hard results. This model has no native export yet.

## Learn which word expert to trust

`train_adaptive_words.py` freezes four complete word models and learns a small query-dependent gate. Shape features and per-expert score statistics determine convex mixture weights. No frequency, candidate identity, true word, writer identity, or difficulty enters the gate. Training uses full 60,000-class cross entropy on the audited training writers. Every development query is scored against every word.

The first five training rounds reach 6,146/8,400 hard first choices, compared with 6,141 for the fixed mature pair and 6,113 for the uniform four-model initializer. This is a word-only experiment; it trails the 6,171-hit complete system. It has no native export. The stage stopped after round 30, with its best at round 5; its saved best checkpoint is preserved. Candidate-permutation tests check that word-bank order cannot affect query weights.

The expanded-real-data stage stopped after 136 completed rounds, 81 beyond its best checkpoint at round 55. Its best 72.50% and latest weights remain preserved. Two larger encoder controls, the wider encoder, synthetic-only training, and the decoder experiments continue independently.

## Combine route and spelling evidence

`combine_route_spelling.py` adds visited-route costs to a previously selected spelling configuration. It binds both systems to the same context, word mixture, development corpus, and natural candidate lists. The zero-route setting must reproduce the existing spelling counts before a full-development comparison is accepted. Missing words remain misses. No target word or true visited route enters inference.

The first 512-row diagnostic improves six hard first choices over its spelling control: 270 versus 264 out of 349. This supports a full comparison; it does not establish 77.36% accuracy across all hard development rows. The full visited-route model reaches 6,152/8,400 hard first choices (73.24%). The complete fusion comparison retains zero route weight and the 6,171-hit spelling baseline. Every baseline count matches, with input hashes unchanged. The nonzero fusion proposal has no native exporter. Scored arrays are retained so reporting repairs do not require another model pass.

The initial wide encoder completed all 180 rounds, with its best at round 174 (73.04% hard development). Its weights are frozen before a follow-up stage using all 31 million exact-generator writers. That stage retains real recordings and augmented canonical routes, including positive examples outside the cached target-length range. It changes the data mixture as well as the schedule and is not a controlled width-only experiment.

## Test normalization during the data transition

`train_proxy.py --freeze-normalization` fixes BatchNorm running means and variances during fine-tuning. Affine parameters, the rest of the encoder, and word vectors remain trainable. Dropout remains active. Resume rejects a changed normalization mode. A regression test checks unchanged statistics and successful affine updates.

The paired wide-encoder runs use the same completed checkpoint, random seed, 31 training shards, real recordings, batch size, optimizer rates, and 200-round schedule. Their only configuration differences are the output directory and normalization flag. Both reproduce identical initial development counts. This tests an early regression in the normal-statistics control; it does not yet establish a trained gain.

The adaptive gate stopped after 25 rounds without improvement, and the nine-layer decoder after 22. Their saved checkpoints remain available. The gate trainer produced a best checkpoint only; the decoder also preserved its latest checkpoint. This frees compute for the larger decoder, structural-route evidence, and the paired normalization experiment.

## Recover unrounded controls

`label_memory_waypoints` exports latent control coordinates alongside the unchanged writer's actual stroke. It reproduces key errors, row shifts, omissions, endpoint errors, global pose, drift, and wave. Interior drift uses the fixed midpoint of each rounded corner. The labels exclude observation jitter. Tests check original-stroke identity, endpoint noise bounds, and clean keyboard geometry. `synthetic.rs` remains unchanged.

`prepare_waypoint_supervision.py` generates 300,000 fresh training writers in seed block 720051. It combines uniform sampling across all 60,000 words with 20% arbitrary keyboard routes. Every count from two to 32 has positive examples. It applies the observed stroke's exact centering and RMS scale to both points and targets. The first 10,000 writers form an internal holdout. Stored model inputs contain points; corner coordinates and counts are loss targets.

`WaypointDecoder` predicts unrounded locations and uncertainty. Its inference method enumerates every possible count from two to 32 and receives no true count, target word, writer identity, or difficulty. Training conditions its supervised location loss on the known training count and separately predicts that count from the stroke. These internal diagnostics are not complete-word accuracy.

The first five rounds improve internal count accuracy to 66.18%, while conditional location MSE stays near 0.567. Lower conditional loss therefore does not yet establish better corner reconstruction. This motivates checking alignment supervision and loss balance before using the model as word evidence. This first stage stopped after 21 rounds and has no native result.

## Isolate count evidence

`evaluate_count_evidence.py` tests the learned count head against the existing full spelling score cache. It computes all 31 count classes from points, removes the count prior estimated from training rows only, and integrates each public candidate's possible interior omission. Neither true query counts nor difficulty enter prediction. The full development comparison keeps zero count weight: 6,171/8,400 hard first choices. Nonzero count evidence is excluded from the selected recognizer.

A reporting-only check of the fixed model across development vocabulary ranges also gives no evidence that the current model meets 80% in the documented benchmark target range. It changes no model, selection, or candidate set and reads no benchmark rows. Rank metadata remains excluded from prediction.

## Separate widened-channel gradients

`--widen-symmetry-noise 0.01` adds paired, canceling projection-column perturbations after expansion. Replicated input activations cancel these perturbations at initialization. Their gradients differ, allowing the replicas to learn different features. A local random generator preserves the training random stream. Tests check unchanged predictions, unchanged global CPU random state, and distinct replicated-channel gradients with dropout disabled.

Five full-size GPU references differ by at most 0.000020 after this change. A matched 28.6-million-parameter run keeps the original context, decoder initialization, retrieval source, 31 training shards, rates, seed, and schedule. The output directory and symmetry setting are its only argument differences. It remains an experiment; initialization parity is not trained accuracy.

The first corner-recovery stage stopped after 21 rounds. Count learning improved, but location MSE stayed near its starting value. Best and latest saved checkpoints are preserved before adding stronger ordered-alignment supervision.


## Supervise corner alignment

The second corner dataset adds the chronological arc position of each latent control. It uses the same 300,000 writers as the first dataset. Preparation checks that the original points, corner coordinates, and counts remain array-identical. These are additional labels, not additional independent swipes. The generator remains unchanged.

`AlignedWaypointDecoder` predicts positive intervals between controls, then accumulates them into ordered arc positions. It samples the observed stroke at those positions and predicts a residual to recover the unrounded control. Explicit location and alignment losses prevent uncertainty or count learning from hiding a location failure. Checkpoint selection uses internal mean-query location MSE plus ten times arc MSE. Inference receives points and evaluates all 31 count hypotheses; it receives no true alignment or count.

The first three rounds reduce internal mean-query location MSE from 0.4406 to 0.1367. This is conditional geometry reconstruction, not word-recognition accuracy. A frozen round-two diagnostic gains one hard first choice on 512 development rows: 253/350 versus 252/350. A full-development comparison is required before drawing conclusions.

`evaluate_waypoint_evidence.py` compares recovered corners with each natural candidate's public keyboard route. It fits a regularized affine pose and integrates possible interior omissions. Count priors come from training rows only. The zero-corner baseline must reproduce every saved full-development spelling count. Candidate lists, inputs, source files, and model hashes are bound and checked. No native exporter exists for this proposal.

Four larger encoder controls stopped without a trained gain: the massive control after 46 rounds, its gate variant after 33, and both normalization variants after 41. Each best checkpoint remains its initializer. Best and latest files are preserved and hashed in `benchmarks/larger-encoder-controls-stopped-v1.json`. Stopping these runs frees unified memory for the larger decoder and aligned-corner work. The 28.6-million-parameter decoder's round-ten gain on its fixed subset is undergoing full-development evaluation; it is not yet a full result.


The frozen round-two corner model completes the full 12,000-row comparison. At weight 0.1, it reaches 6,172/8,400 hard first choices, one above the verified spelling baseline. It loses one clean first choice and gains two loose first choices. Every zero-corner baseline count matches, and inputs remain unchanged. This small development gain does not justify replacing the native preview or claiming progress to 80%. CPU and Metal geometry costs agree within 0.00000024 on 24 independent fit cases.

Later word checkpoints produce a stronger word-only mixture: 25% wider round 150, 25% wider round 174, and 50% synthetic-only round 158. Full-vocabulary development reaches 6,162/8,400 hard first choices, 21 above the previous word pair. The complete nine-layer spelling comparison is running. This result is separate from both the native complete model and the 28.6-million-parameter decoder comparison.
