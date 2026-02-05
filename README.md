# Swipe Predictor

**Blind swipe typing** - draw gestures on a virtual keyboard to predict words.

## Two Approaches

### 1. DTW-Based
```bash
cargo run -p swipe-desktop --bin swipe-dtw
```

### 2. Neural Network (CTC)
```bash
cargo run -p swipe-desktop --bin swipe-ctc
```

See `training/README.md` for model training.

## Architecture
```
crates/swipe-engine/  # Core engine
apps/desktop/        # Desktop app
training/            # Model training
```
