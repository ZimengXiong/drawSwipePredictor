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
Gesture Path → Feature Extraction(7d) → Feature Masking → Conv1D(128) →
Relative Gesture Attention → BiLSTM(256, 3) → CTC Decoder → Letter Sequence
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

### Feature Masking (GestureFeatureMasking)

During training, randomly masks contiguous spans of the input sequence
(zeroing out 1–8 timesteps per span, up to 2 spans). This forces the model
to reconstruct missing path segments from surrounding context, making it
robust to the partially-recalled gestures common in blind drawing.
Disabled during inference.

### Conv1D Front-End

Two Conv1D layers (kernel_size=3) with BatchNorm extract local n-gram-like
patterns before the attention and BiLSTM layers.

### Relative Gesture Attention (RelativeGestureAttention)

A lightweight self-attention layer between Conv1D and BiLSTM that lets
the model see the global gesture shape while processing local patterns.
Uses learnable relative position biases so nearby points attend more
strongly to each other, which is natural for gesture paths. This helps
the model understand "where am I in the overall shape" — critical for
CTC alignment since blind drawings have distorted proportions but
correct overall topology.

### Training Schedule

- **Warmup**: Linear LR warmup over the first few epochs for stable training
- **Cosine annealing**: Smooth LR decay after warmup

## Training Data Generation

### The Problem: Drawing From Memory

Users draw swipe gestures **from memory on a blank trackpad** — there is no
keyboard visible. They recall the approximate shape of a word's swipe path
and reproduce it freehand. This creates very different distortions than
swiping on an actual keyboard:

- **No absolute reference frame**: the drawing can be any scale, rotation, position
- **Proportions are approximate**: some segments get compressed or stretched
- **Angles are remembered roughly**: ±20–30° direction errors are common
- **Sharp corners get rounded**: finger momentum smooths out details
- **Confidence varies along the path**: well-remembered parts are precise, fuzzy parts are sloppy
- **Overall shape is simplified**: fine directional details may be lost

### Augmentation Strategy

| Technique                  | Description                                                    |
|----------------------------|----------------------------------------------------------------|
| Arbitrary scale            | 0.3x–3.0x (no absolute size reference)                         |
| Arbitrary aspect ratio     | Independent X/Y scale 0.5x–2.0x (remembered proportions vary) |
| Rotation                   | ±8° (mental keyboard orientation is approximate)               |
| Segment stretching/compression | Smooth non-uniform warping along the path (proportions are approximate) |
| Shape exaggeration/flattening | 0.7x–1.5x radial distortion (angles remembered roughly)     |
| Vertical drift             | Smooth Y-axis wander during drawing                            |
| Horizontal drift           | Smooth X-axis wander during drawing                            |
| Corner rounding            | Gaussian smoothing (finger momentum rounds sharp turns)        |
| Spatially-varying noise    | Some regions sloppier than others (confidence varies)          |
| Hand tremor                | Gaussian jitter (trackpad drawing precision)                   |
| Arbitrary position         | Random global offset (no reference point)                      |

### Pipeline

Each training sample:

- Gets ideal word path from QWERTY layout
- Interpolates between key positions
- Applies blind-drawing augmentations (simulating memory recall)
- Resamples to 64 equidistant points
- Normalizes: center at centroid, scale by path length
- Extracts 7 features per point
- Labels: letter sequence (a-z)
