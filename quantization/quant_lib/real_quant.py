"""
Export/import REAL (not "fake") low-bit quantized checkpoints.

`quant_ops.fake_quantize` (used by `ModelQuantizer`/`run_quantization.py`)
quantizes a weight then immediately dequantizes it back to fp32, so the
*value* is snapped onto a low-bit grid but the *storage* stays 4 bytes/
element -- that's what let the unmodified EvT model code run unchanged and is
enough to measure the PTQ accuracy impact, but it means the saved `.pt`
checkpoints are (almost) the same size as fp32.

This module does what an actual deployed quantized model repo (bitsandbytes,
GPTQ, GGUF, HF `safetensors` with a quant config, ...) does: pack weights
into real low-bit buffers (`torch.int8` for 8-bit, two 4-bit values per byte
for 4-bit) plus a small fp32 per-channel/per-tensor scale, and dequantize
back to fp32 on load, right before the tensor is used -- same numerical
result as `fake_quantize`, but the file on disk is genuinely smaller.

Only `nn.Linear.weight` and `nn.MultiheadAttention.in_proj_weight` are
touched here, mirroring exactly what `ModelQuantizer.quantize_weights()`
targets (see `quant_wrapper.py`); everything else (biases, LayerNorm affine
params, positional encoding, latent memory vectors) is copied through as-is
so the exported checkpoint is fully self-contained (no need to also ship the
original fp32 file to reconstruct the model).
"""

import torch
import torch.nn as nn

from . import quant_ops


def _quant_targets(model):
    """Yields (state_dict_key, module, attr_name) for every weight tensor that
    `ModelQuantizer` quantizes -- same walk as `quant_wrapper.py:52-60`."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            yield f"{name}.weight", m, 'weight'
        elif isinstance(m, nn.MultiheadAttention):
            yield f"{name}.in_proj_weight", m, 'in_proj_weight'


def pack_int4(q_int8):
    """Pack a tensor with integer values in [-8, 7] two-per-byte into a
    `torch.uint8` buffer (half the bytes of a plain int8 tensor)."""
    flat = q_int8.detach().flatten().to(torch.int16)
    n = flat.numel()
    if n % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int16)])
    nibble = (flat + 8).clamp(0, 15).to(torch.uint8)   # map [-8, 7] -> [0, 15]
    lo, hi = nibble[0::2], nibble[1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed, n


def unpack_int4(packed, numel):
    """Inverse of `pack_int4`: returns a float32 tensor of the original
    (unscaled) integer codes, truncated back to `numel` elements."""
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(packed.numel() * 2, dtype=torch.int16)
    out[0::2] = lo
    out[1::2] = hi
    return out[:numel].to(torch.float32)


def export_real_quantized(model, weight_bits, per_channel, out_path, extra=None):
    """Write a compact, self-contained checkpoint for `model`: every
    quantizable weight is stored as a real low-bit buffer + scale; everything
    else is copied through verbatim (fp32). Returns the payload dict that was
    saved (for immediate inspection without re-reading the file).
    """
    quantized = {}
    with torch.no_grad():
        for key, module, attr in _quant_targets(model):
            w = getattr(module, attr).data
            scale = quant_ops.compute_qparams(w, weight_bits, per_channel=per_channel, channel_dim=0)
            q = quant_ops.quantize_to_int(w, scale, weight_bits)
            if weight_bits == 4:
                packed, numel = pack_int4(q)
                entry = dict(packed=packed, numel=numel, bits=4)
            else:
                entry = dict(packed=q.clone(), numel=q.numel(), bits=weight_bits)
            entry['scale'] = scale.clone().float()
            entry['shape'] = tuple(w.shape)
            quantized[key] = entry

    fp32_extra = {k: v.clone() for k, v in model.state_dict().items() if k not in quantized}

    payload = dict(quantized=quantized, fp32_extra=fp32_extra,
                   weight_bits=weight_bits, per_channel=per_channel)
    if extra:
        payload.update(extra)
    torch.save(payload, out_path)
    return payload


def load_real_quantized(path, model, map_location='cpu'):
    """Reconstruct a runnable fp32 model in-place from a real-quantized
    checkpoint written by `export_real_quantized`. Returns (model, payload)."""
    payload = torch.load(path, map_location=map_location)
    sd = model.state_dict()

    for k, v in payload['fp32_extra'].items():
        sd[k] = v

    for key, entry in payload['quantized'].items():
        shape = torch.Size(entry['shape'])
        if entry['bits'] == 4:
            flat = unpack_int4(entry['packed'], entry['numel'])
        else:
            flat = entry['packed'].to(torch.float32)
        w_int = flat[:shape.numel()].reshape(shape)
        sd[key] = w_int * entry['scale']

    model.load_state_dict(sd)
    return model, payload


def real_quantized_size_bytes(payload):
    """Bytes actually used by a payload dict, for a direct sanity check
    against `torch.save`'s on-disk size (which adds some pickle/zip overhead)."""
    total = 0
    for entry in payload['quantized'].values():
        total += entry['packed'].numel() * entry['packed'].element_size()
        total += entry['scale'].numel() * entry['scale'].element_size()
    for v in payload['fp32_extra'].values():
        total += v.numel() * v.element_size()
    return total
