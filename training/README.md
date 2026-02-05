# CTC Model Training

```bash
uv sync
uv run python train_ctc.py --words ../word_freq.txt --output training/swipe_ctc.onnx
cargo run -p swipe-desktop --bin swipe-ctc
```

## Key Parameters

- `--words` - Word frequency file (required)
- `--output` - ONNX model path
- `--max-words` - Vocabulary size (default: 10000)
- `--epochs` - Training epochs (default: 50)
- `--device` - `cpu`, `cuda`, or `mps` (default: mps)
- `--checkpoint-dir` - Checkpoint directory (default: `training/checkpoints`)

## Export from Checkpoint

```bash
uv run python train_ctc.py --resume training/checkpoints/checkpoint_epoch_50.pt --epochs 0
```

Gesture Path → BiLSTM(256, 3) → CTC Decoder → Letter Sequence

## Training Data Generation

**Synthetic data augmentation** simulates realistic blind swipe variations:

1. **Scale variation** (0.5x - 2.0x) - People draw at different sizes
2. **Rotation** (±25°) - Mental keyboard orientation varies
3. **Aspect distortion** - Horizontal/vertical stretch
4. **Point jitter** - Hand tremor and imprecision
5. **Speed warping** - Non-uniform drawing speed
6. **Endpoint noise** - Start/end point inaccuracy
7. **Path simplification** - Occasional letter skipping

Each training sample:

- Gets word path from QWERTY layout
- Applies random augmentations
- Resamples to 64 equidistant points
- Labels: letter sequence (a-z)
