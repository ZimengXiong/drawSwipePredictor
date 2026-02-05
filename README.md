# Draw Swipe Predictor

Predict words purely from drawn keyboard key paths using either a DTW or CTC approach.

| `alpacas`                    | `lunchbox`                    |
| ---------------------------- | ----------------------------- |
| ![](screenshots/alpacas.png) | ![](screenshots/lunchbox.png) |

As you can see, DTW was able to predict `lunchbox` much better than `alpacas` because the pattern is much more distinct for the latter word (when drawing without referencing a keyboard, of course, when trying to draw carefully, DTW can correctly identify both words).

## DTW (Dynamic Time Warping)

We generalize and score paths on the folowing characteristics:
| Feature | Description | Type |
| ------------------- | -------------------------------------------------------------- | ------------------ |
| shape | Normalized resampled path (32 points, centered, length-scaled) | Position-invariant |
| angles | Direction angles per segment (radians) | Sequential |
| displacements | Normalized displacement vectors per segment | Sequential |
| turn_angles | Turn angles (signed: +CCW/-CW) | Sequential |
| start_angle | First segment direction angle | Sequential |
| end_vector | Start-to-end displacement (normalized) | Global |
| aspect_ratio | Bounding box width/height | Global |
| straightness | Direct distance / path length | Global |
| total_turning | Sum of turn angles (complexity) | Global |
| segment_ratios | Segment lengths / total length | Sequential |
| direction_histogram | 8-bin direction distribution | Distribution |
| location | Absolute position matching (SHARK2-style) | Positional |

## CTC (Connectionist Temporal Classification)

- Neural network (BiLSTM) trained on synthetic gesture-to-letter sequences
- Decodes variable-length gestures into letter sequences
- Generalizes to unseen words
- Worse performance compared to DTW as of now

## Usage

```bash
# DTW backend
cargo run -p swipe-desktop --bin swipe-dtw

# CTC neural backend
cargo run -p swipe-desktop --bin swipe-ctc
```

## Training

See [training/README.md](training/README.md) for model training.

```
crates/swipe-engine/  # DTW + CTC inference engines
apps/desktop/        # UI with backend abstraction
training/            # CTC model training
```
