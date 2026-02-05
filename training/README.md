# CTC Model Training

```bash
uv sync
uv run python train_ctc.py --words ../word_freq.txt --output training/swipe_ctc.onnx
cargo run -p swipe-desktop --bin modeldraw --features neural
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

## Model Architecture

Gesture Path → BiLSTM(256, 3) → CTC Decoder → Letter Sequence
