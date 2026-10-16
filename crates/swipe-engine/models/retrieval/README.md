# Bundled route recognizer

This model encodes 64-point gesture routes into 128-dimensional vectors. The word bank contains 60,000 English word routes, including `lunchbox` and `alpacas`.

Training used real recordings from [FUTO's MIT-licensed swipe dataset](https://huggingface.co/datasets/futo-org/swipe.futo.org), procedural drawings, and online memory-style distortions. See `DATA-LICENSE.txt` for the data attribution. No recordings from the source test split were used for training, checkpoint selection, or calibration.

`manifest.json` records model settings, source hashes, and validation results. `encoder.bin` contains fused convolution and dense weights. `words.bin` contains canonical and empirical word vectors. The native loader verifies their shared model fingerprint. `verification.json` contains expected encoder outputs for cross-runtime checks.

Regenerate the files with the training and export workflow in `training/README.md` at the repository root. Raw data and optimizer checkpoints are generated locally and are not bundled.
