# Recognition benchmarks

The model was frozen before these test runs. All recognizers receive the same points and candidate vocabulary within each comparison. Times are observed CPU inference times on an Apple M4 Max with 48 GB of RAM. Training used MPS; inference uses native Rust.

## Final reranker comparison

The final comparison uses 879 recordings from 107 donor sessions absent from every earlier collected dataset, plus 2,000 synthetic drawings at each difficulty. Both versions use the same encoder, word bank, 60,000-word vocabulary, and input points. The new version adds geometry candidates, a residual MLP, and a conservative fallback for clear neural matches.

| Input | Previous learned: first choice | New: first choice | New: top five |
|---|---:|---:|---:|
| Real swipe | 82.3% | 83.6% | 94.1% |
| Distorted real swipe | 71.3% | 73.0% | 88.9% |
| Clean synthetic | 96.5% | 96.5% | 99.8% |
| Loose synthetic | 66.9% | 70.5% | 84.2% |
| Hard synthetic memory | 33.9% | 42.8% | 62.2% |

Paired 95% intervals for first-choice gains, resampling donor sessions or synthetic writers:

| Condition | Gain | 95% interval |
|---|---:|---:|
| Recorded real | +1.37 points | +0.12 to +2.69 |
| Distorted real | +1.71 points | +0.12 to +3.29 |
| Loose synthetic | +3.60 points | +2.70 to +4.55 |
| Hard synthetic memory | +8.95 points | +7.50 to +10.35 |

The synthetic targets are sampled uniformly from eligible vocabulary ranks 54,000–60,000. They are rarer than the targets in the earlier 10,000-word test, so accuracy levels should not be compared across those tests. Canonical augmentation covers the deployment vocabulary during training; these are held-out drawings and writers, not wholly unseen words.

An initial test exposed a clean-route regression in the unprotected reranker. That report remains as `reranker-initial-test.json`. We added a fallback that preserves clear neural matches, calibrated it on validation data and 3,000 affine reference routes, then reserved a second cohort. The final cohort excludes every donor from both the original dataset and the initial test collection. Its synthetic seed is 246810. The selected weights and final protocol were committed before the final run.

The corrected model also repeats the earlier 10,000-word test: clean 99.0%, loose 81.0%, and hard memory 57.3% first-choice accuracy. This repeat is a regression comparison, not a fresh test. Its previous learned-model results were 99.0%, 78.7%, and 52.3%.

The reranker adds CPU work: a separate 1,500-query run measured 6.90 ms median and 7.36 ms p95, versus 3.52 ms and 3.80 ms for the previous learned version. Use `reranker-latency.json` for that timing run; accuracy-run timings also include concurrent build and verification load. These intervals describe the sampled cohorts, not the full population of freehand-memory users.

Reports: `reranker-final-test.json`, `reranker-final-test-protocol.json`, `reranker-synthetic-test-10k.json`, `reranker-calibration.json`, and `reranker-union-*`. Reproduction commands are in [the training guide](../training/README.md).

## Earlier learned-model comparisons

The following results describe the first learned model before candidate reranking.

### Default 60,000-word vocabulary

The test includes 500 real recordings from 136 held-out donor sessions. The transfer condition adds rotation, axis stretch, shear, and drift to those same recordings.

| Input | Original DTW top 1 | Geometry top 1 | Learned top 1 | Learned top 5 |
|---|---:|---:|---:|---:|
| Recorded human swipe | 44.4% | 45.8% | 86.2% | 96.2% |
| Human swipe + simulated memory errors | 24.6% | 43.0% | 78.8% | 93.0% |

Median transfer inference time: 132.4 ms for the original DTW recognizer and 3.31 ms for the learned recognizer. The measured ratio is 40×.

### 10,000-word comparison

This run includes 1000 eligible real recordings from 158 held-out donor sessions.

| Input | Original top 1 | Geometry top 1 | Learned top 1 | Learned top 5 |
|---|---:|---:|---:|---:|
| Recorded | 45.7% | 61.5% | 87.9% | 97.4% |
| Distorted real | 29.3% | 58.3% | 82.2% | 94.7% |

### Synthetic stress test

The 900 gestures use fresh seed 12345, with 300 drawings at each difficulty. Targets are sampled uniformly from eligible 4–12 letter words in the 10,000-word vocabulary. This differs from the frequency distribution of ordinary language.

| Difficulty | Original top 1 | Geometry top 1 | Learned top 1 | Learned top 5 |
|---|---:|---:|---:|---:|
| clean | 80.0% | 98.3% | 99.0% | 100.0% |
| loose | 49.0% | 77.0% | 78.7% | 88.7% |
| memory | 22.7% | 54.7% | 52.3% | 72.7% |

The hardest synthetic condition can remove a waypoint and distort the remembered keyboard. Both new recognizers still make substantial errors there. The geometry recognizer remains useful as an alternative; the learned model has the strongest results on real and distorted real recordings.

## Scope and provenance

The real data comes from the MIT-licensed [FUTO swipe corpus](https://huggingface.co/datasets/futo-org/swipe.futo.org), pinned to revision `d71bf5fd7f45b3e7c2ed2d76a21b0dbd3b4ba566`. Source train, validation, and test sessions are disjoint. Conversion verifies that the selected session sets do not overlap.

The recordings were collected over visible keyboards. The transfer condition is a simulation applied to real traces. Neither condition measures real freehand gestures drawn entirely from memory. Repeated-letter and translated-route ambiguities also remain. Top-five accuracy reports whether the intended word appears among alternatives, not whether the recognizer can select it without help.

Test targets are 4–12 ASCII letters and must be in the comparison vocabulary. No recognition-distance filter removes difficult recordings. The byte-window importer samples clusters of records, so the sample is not uniformly drawn from the entire corpus. Timing is descriptive and can vary with device load.

The synthetic dataset files separate target words and writer seeds. The neural trainer also generates augmented canonical routes over the entire deployment vocabulary. Its test results therefore measure new drawings and new donors, not a vocabulary held out from all training.

The original CTC model has no bundled trained checkpoint, so these reports compare against the original DTW implementation.

Earlier learned-model reports: `neural-real-test-10k.json`, `neural-real-test-60k.json`, and `neural-synthetic-test-10k.json`. Earlier `heldout-*` reports document development of the geometry recognizer. Dataset manifests and training histories accompany these files. Commands to reproduce the final reports are in the root README.
