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

Uses a **two-strategy mixture** for maximum diversity:

### Strategy 1: Realistic Path Generation (~50%)

Simulates how a real finger moves across the keyboard:

| Technique              | Description                                                    |
|------------------------|----------------------------------------------------------------|
| Per-key positional noise | Gaussian noise around key centers (fingers miss targets)     |
| Key neighborhood confusion | Small probability of targeting adjacent key              |
| Letter skipping        | Fast swipers sometimes skip intermediate keys                  |
| Curved interpolation   | Bezier-like curves between keys with momentum/overshoot        |
| Per-key dwell variation | Some keys get lingered on (extra points near key position)    |
| Finger inertia         | Overshoot at sharp direction changes (momentum simulation)     |

### Strategy 2: Global Augmentation (~50%)

Applies broad geometric transforms to ideal straight-line paths:

| Technique              | Description                                                    |
|------------------------|----------------------------------------------------------------|
| Scale variation        | 0.5x–2.0x size (people draw at different sizes)                |
| Rotation               | ±25° (mental keyboard orientation varies)                      |
| Aspect distortion      | Independent X/Y stretch (horizontal/vertical bias)             |
| Vertical drift         | Smooth Y-axis drift (hand wanders)                             |
| Horizontal drift       | Smooth X-axis drift (lateral keyboard shift)                   |
| Corner rounding        | Gaussian smoothing (sharp turns become curves)                 |
| Local deformation      | Spatially-varying noise (some regions sloppier than others)    |
| Speed warping          | Sinusoidal time warping (non-uniform drawing speed)            |
| Point jitter           | Gaussian noise (hand tremor and imprecision)                   |
| Global translation     | Random offset (different drawing positions)                    |

### Pipeline

Each training sample:

- Selects a generation strategy (realistic or classic)
- Gets word path from QWERTY layout (with or without per-key noise)
- Interpolates between keys (curved or straight)
- Applies augmentation transforms
- Resamples to 64 equidistant points
- Normalizes: center at centroid, scale by path length
- Extracts 7 features per point
- Labels: letter sequence (a-z)
