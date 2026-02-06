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
- `--warmup` - Learning rate warmup epochs (default: 3)
- `--device` - `cpu`, `cuda`, or `mps` (default: mps)
- `--checkpoint-dir` - Checkpoint directory (default: `training/checkpoints`)

## Export from Checkpoint

```bash
uv run python train_ctc.py --resume training/checkpoints/checkpoint_epoch_50.pt --epochs 0
```

## Architecture

```
Gesture Path → Feature Extraction(7d) → Conv1D(128) → BiLSTM(256, 3) → CTC Decoder → Letter Sequence
```

### Input Features (7 dimensions per timestep)

| Feature    | Description                              |
|------------|------------------------------------------|
| x, y       | Normalized position (centroid-centered)  |
| dx, dy     | Delta to next point (local direction)    |
| angle      | Direction angle (radians/π, [-1, 1])     |
| speed      | Segment length / average segment length  |
| curvature  | Change in direction angle between steps  |

The richer feature set gives the model local directional context at each
timestep, significantly improving CTC alignment compared to raw (x, y) alone.

### Conv1D Front-End

Two Conv1D layers (kernel_size=3) with BatchNorm extract local n-gram-like
patterns before the BiLSTM, improving the model's ability to recognise
short letter sequences and transitions.

### Training Schedule

- **Warmup**: Linear LR warmup over the first few epochs for stable training
- **Cosine annealing**: Smooth LR decay after warmup

## Training Data Generation

**Synthetic data augmentation** simulates realistic blind swipe variations:

1. **Scale variation** (0.5x - 2.0x) - People draw at different sizes
2. **Rotation** (±25°) - Mental keyboard orientation varies
3. **Aspect distortion** - Horizontal/vertical stretch
4. **Point jitter** - Hand tremor and imprecision
5. **Speed warping** - Non-uniform drawing speed
6. **Endpoint noise** - Start/end point inaccuracy
7. **Path simplification** - Occasional letter skipping
8. **Horizontal drift** - Lateral mental keyboard shift
9. **Non-uniform speed warping** - Sinusoidal time warping

Each training sample:

- Gets word path from QWERTY layout
- Applies random augmentations
- Resamples to 64 equidistant points
- Extracts 7 features per point
- Labels: letter sequence (a-z)
