# Recognition benchmarks

The model was frozen before these test runs. All recognizers receive the same points and candidate vocabulary within each comparison. Times are observed CPU inference times on an Apple M4 Max with 48 GB of RAM. Training used MPS; inference uses native Rust.

## Default 60,000-word vocabulary

The test includes 500 real recordings from 136 held-out donor sessions. The transfer condition adds rotation, axis stretch, shear, and drift to those same recordings.

| Input | Original DTW top 1 | Geometry top 1 | Learned top 1 | Learned top 5 |
|---|---:|---:|---:|---:|
| Recorded human swipe | 44.4% | 45.8% | 86.2% | 96.2% |
| Human swipe + simulated memory errors | 24.6% | 43.0% | 78.8% | 93.0% |

Median transfer inference time: 132.4 ms for the original DTW recognizer and 3.31 ms for the learned recognizer. The measured ratio is 40×.

## 10,000-word comparison

This run includes 1000 eligible real recordings from 158 held-out donor sessions.

| Input | Original top 1 | Geometry top 1 | Learned top 1 | Learned top 5 |
|---|---:|---:|---:|---:|
| Recorded | 45.7% | 61.5% | 87.9% | 97.4% |
| Distorted real | 29.3% | 58.3% | 82.2% | 94.7% |

## Synthetic stress test

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

Final reports: `neural-real-test-10k.json`, `neural-real-test-60k.json`, and `neural-synthetic-test-10k.json`. Earlier `heldout-*` reports document development of the geometry recognizer. Dataset manifests and training histories accompany these files. Commands to reproduce the final reports are in the root README.
