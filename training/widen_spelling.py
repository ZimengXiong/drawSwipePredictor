"""Widen spelling layers while retaining their initialized inference function."""
import torch

from autoregressive_model import decoder_dimensions


@torch.no_grad()
def widen_decoder(model, original):
    """Replicate channels and rescale attention for the decoder's six heads."""
    old_width, old_layers, positions = decoder_dimensions(original)
    new_width = model.symbols.embedding_dim
    if (new_width <= old_width or new_width % old_width or old_width % 6
            or positions != model.positions.num_embeddings or old_layers > len(model.decoder.layers)):
        raise ValueError('widening requires a larger multiple of width, fixed positions, and no removed layers')
    ratio = new_width // old_width
    device = model.symbols.weight.device
    source = {key: value.to(device) for key, value in original.items()}
    # Duplicate within each head so its attention sees repeated old coordinates.
    channels = torch.arange(old_width, device=device).reshape(6, -1).repeat(1, ratio).flatten()
    hidden = torch.arange(old_width * 4, device=device).repeat(ratio)
    state = model.state_dict()
    for name in ['memory.weight', 'memory.bias']:
        state[name].copy_(source[name][channels])
    for name in ['memory_position', 'symbols.weight', 'positions.weight']:
        state[name].copy_(source[name][..., channels])
    for name in ['decoder.norm.weight', 'decoder.norm.bias']:
        state[name].copy_(source[name][channels])
    state['output.weight'].copy_(source['output.weight'][:, channels] / ratio)
    state['output.bias'].copy_(source['output.bias'])
    for index in range(old_layers):
        prefix = f'decoder.layers.{index}.'
        for norm in ['norm1', 'norm2', 'norm3']:
            for suffix in ['weight', 'bias']:
                name = prefix + norm + '.' + suffix
                state[name].copy_(source[name][channels])
        for attention in ['self_attn', 'multihead_attn']:
            base = prefix + attention + '.'
            for block in range(3):
                rows = channels + block * old_width
                multiplier = ratio ** -0.25 if block < 2 else 1.0
                target = slice(block * new_width, (block+1) * new_width)
                state[base+'in_proj_weight'][target].copy_(source[base+'in_proj_weight'][rows][:, channels] * (multiplier / ratio))
                state[base+'in_proj_bias'][target].copy_(source[base+'in_proj_bias'][rows] * multiplier)
            state[base+'out_proj.weight'].copy_(source[base+'out_proj.weight'][channels][:, channels] / ratio)
            state[base+'out_proj.bias'].copy_(source[base+'out_proj.bias'][channels])
        for layer, output_ids, input_ids in [('linear1', hidden, channels), ('linear2', channels, hidden)]:
            name = prefix + layer
            state[name+'.weight'].copy_(source[name+'.weight'][output_ids][:, input_ids] / ratio)
            state[name+'.bias'].copy_(source[name+'.bias'][output_ids])
    model.load_state_dict(state)
    for block in model.decoder.layers[old_layers:]:
        for output in [block.self_attn.out_proj, block.multihead_attn.out_proj, block.linear2]:
            output.weight.zero_()
            output.bias.zero_()


@torch.no_grad()
def break_widened_symmetry(model, original, scale=0.01, seed=900971):
    """Change replicated-channel gradients while canceling their initial outputs."""
    old_width, old_layers, _ = decoder_dimensions(original)
    new_width = model.symbols.embedding_dim
    if new_width <= old_width or new_width % old_width or not 0 < scale <= 0.05:
        raise ValueError('symmetry breaking requires an expanded width and a small positive scale')
    ratio = new_width//old_width
    channels = torch.arange(old_width).reshape(6,-1).repeat(1,ratio).flatten()
    hidden = torch.arange(old_width*4).repeat(ratio)
    generator = torch.Generator(device='cpu').manual_seed(seed)

    def perturb(layer, mapping):
        groups = torch.stack([(mapping==index).nonzero().flatten() for index in range(int(mapping.max())+1)])
        if groups.numel()!=layer.weight.shape[1] or groups.shape[1]!=ratio:
            raise ValueError('projection does not have the expected replicated inputs')
        noise = torch.randn(layer.weight.shape[0],len(groups),ratio-1,generator=generator)*scale
        noise = torch.cat([noise,-noise.sum(2,keepdim=True)],dim=2)
        columns = groups.flatten().to(layer.weight.device)
        layer.weight[:,columns] += noise.reshape(layer.weight.shape[0],-1).to(layer.weight.device)

    for block in model.decoder.layers[:old_layers]:
        for attention in [block.self_attn,block.multihead_attn]:
            # MultiheadAttention stores its combined Q/K/V matrix directly.
            class Projection:
                pass
            projection=Projection();projection.weight=attention.in_proj_weight
            perturb(projection,channels)
            perturb(attention.out_proj,channels)
        perturb(block.linear1,channels)
        perturb(block.linear2,hidden)
    perturb(model.output,channels)
