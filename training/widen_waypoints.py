#!/usr/bin/env python3
"""Expand an aligned corner model while preserving every count hypothesis."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from aligned_waypoint_model import AlignedWaypointDecoder
from autoregressive_model import SpellingDecoder
from prepare_waypoint_supervision import digest
from train_joint import load_checkpoint
from widen_spelling import widen_decoder, break_widened_symmetry


@torch.no_grad()
def widen_waypoints(target, original, seed=901049):
    old_width, width = original.slots.embedding_dim, target.slots.embedding_dim
    if width <= old_width or width % old_width or len(target.decoder.layers) < len(original.decoder.layers):
        raise ValueError('expand width by an integer multiple and retain every original block')
    if target.memory.in_channels != original.memory.in_channels:
        raise ValueError('corner context dimensions must remain unchanged')
    ratio = width//old_width
    channels = torch.arange(old_width).reshape(6, -1).repeat(1, ratio).flatten()
    # Reuse the tested attention expansion with temporary spelling adapters.
    # Their unused symbol/output tensors never become part of the corner model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapters = []
        for model in [original, target]:
            adapter = SpellingDecoder(model.memory.in_channels-256, model.slots.embedding_dim,
                                      len(model.decoder.layers))
            adapter.memory_position = nn.Parameter(model.memory_position.detach().clone())
            adapter.memory.load_state_dict(model.memory.state_dict())
            adapter.decoder.load_state_dict(model.decoder.state_dict())
            adapters.append(adapter)
        source, expanded = adapters
        widen_decoder(expanded, source.state_dict())
        break_widened_symmetry(expanded, source.state_dict(), .01, seed)
    target.memory.load_state_dict(expanded.memory.state_dict())
    target.memory_position.copy_(expanded.memory_position)
    target.decoder.load_state_dict(expanded.decoder.state_dict())
    for name in ['slots', 'counts']:
        getattr(target, name).weight.copy_(getattr(original, name).weight[:, channels])
    for name in ['base_projection', 'arc_encoding']:
        first, second = getattr(target, name), getattr(original, name)
        first.weight.copy_(second.weight[channels])
        if first.bias is not None:
            first.bias.copy_(second.bias[channels])
    for name in ['location', 'log_scale', 'gaps']:
        first, second = getattr(target, name), getattr(original, name)
        first.weight.copy_(second.weight[:, channels]/ratio)
        first.bias.copy_(second.bias)
    pooled = torch.cat([channels, channels+old_width])
    target.count_head[0].weight.copy_(original.count_head[0].weight[channels][:, pooled]/ratio)
    target.count_head[0].bias.copy_(original.count_head[0].bias[channels])
    target.count_head[2].weight.copy_(original.count_head[2].weight[:, channels]/ratio)
    target.count_head[2].bias.copy_(original.count_head[2].bias)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('use a new immutable initialization path')
    torch.set_num_threads(4)
    saved, source_hash = load_checkpoint(args.input)
    metadata = saved['metadata']
    config = metadata['arguments']
    if metadata['test_data_used'] or metadata.get('model_type') != 'aligned_waypoints_v2' or config['width'] != 192 or config['layers'] != 3:
        raise ValueError('expected the declared three-layer aligned corner model')
    raw_channels = saved['model']['memory.weight'].shape[1]-256
    original = AlignedWaypointDecoder(raw_channels, 192, 3).eval()
    original.load_state_dict(saved['model'])
    target = AlignedWaypointDecoder(raw_channels, 384, 6).eval()
    widen_waypoints(target, original)
    generator = torch.Generator().manual_seed(901051)
    sequence = torch.randn(4, 128, 32, generator=generator)
    raw = torch.randn(4, raw_channels, 64, generator=generator)
    embedding = torch.randn(4, 128, generator=generator)
    points = torch.randn(4, 64, 2, generator=generator).cumsum(1)/8
    with torch.no_grad():
        first = original.infer(original.encode_memory(sequence, raw, embedding), points)
        second = target.infer(target.encode_memory(sequence, raw, embedding), points)
    errors = {}
    for name in ['means', 'log_scales', 'count_log_probabilities']:
        errors[name] = float((first[name]-second[name]).abs().max())
        torch.testing.assert_close(first[name], second[name], atol=0.0001, rtol=0.0001)
    expanded_metadata = copy.deepcopy(metadata)
    expanded_metadata['arguments'].update(width=384, layers=6)
    expanded_metadata['parameters'] = sum(parameter.numel() for parameter in target.parameters())
    expanded_metadata['expansion'] = {'source_sha256': source_hash, 'source_epoch': saved['epoch'],
                                      'source_code_sha256': digest(Path(__file__)),
                                      'maximum_absolute_errors': errors, 'reference_queries': 4,
                                      'count_hypotheses_per_query': 31, 'symmetry_noise': .01,
                                      'limits': 'Synthetic tensor initialization parity only; no trained gain or word-recognition result.'}
    output = {**saved, 'model': target.state_dict(), 'metadata': expanded_metadata}
    for name in ['optimizer', 'scheduler']:
        output.pop(name, None)
    torch.save(output, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(expanded_metadata['expansion'], indent=2)+'\n')
    print(json.dumps({'parameters': expanded_metadata['parameters'], 'maximum_absolute_errors': errors}), flush=True)


if __name__ == '__main__':
    main()
