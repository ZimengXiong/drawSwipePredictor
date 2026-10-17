#!/usr/bin/env python3
"""Export a widened initialization solely for native implementation checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

from autoregressive_model import SpellingDecoder
from export_retrieval import fingerprint
from export_spelling import spelling_binary
from train_joint import load_checkpoint
from widen_spelling import widen_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('use a new initialization-probe directory')
    manifest = json.loads((args.base / 'manifest.json').read_text())
    source, source_hash = load_checkpoint(args.decoder)
    old_bytes = (args.base / 'spelling.bin').read_bytes()
    spelling = manifest['spelling']
    if (source['metadata']['test_data_used'] or spelling['checkpoint_sha256'] != source_hash
            or spelling['sha256'] != hashlib.sha256(old_bytes).hexdigest()
            or 'geometry_ranker' in manifest):
        raise ValueError('base must contain the bound source decoder without a later correction')
    channels = struct.unpack_from('<I', old_bytes, 32)[0]
    model = SpellingDecoder(channels, width=384, layers=12).eval()
    widen_decoder(model, source['model'])
    context, encoder, bank = struct.unpack_from('<QQQ', old_bytes, 8)
    proxy_weight, length_power = struct.unpack_from('<ff', old_bytes, 36)
    binary = spelling_binary(model, context, encoder, bank, channels, proxy_weight, length_power)
    shutil.copytree(args.base, args.output)
    (args.output / 'spelling.bin').write_bytes(binary)
    spelling.pop('selection_sha256', None)
    spelling.pop('checkpoint_sha256', None)
    spelling.update(sha256=hashlib.sha256(binary).hexdigest(), tag=fingerprint(binary),
                    initialization_source_checkpoint_sha256=source_hash,
                    initialization_parity_only=True, decoder_width=384, decoder_layers=12)
    manifest['limits'] = 'Widened initialization probe only. No selected development accuracy or app promotion.'
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()),
                      'manifest_sha256': hashlib.sha256((args.output / 'manifest.json').read_bytes()).hexdigest(),
                      'test_data_used': False, 'initialization_parity_only': True}), flush=True)


if __name__ == '__main__':
    main()
