"""
Fake-quantization primitives (quantize -> dequantize round trip).

These functions never actually change the storage dtype of a tensor: they take a
float32 tensor, simulate what would happen if it were stored with `num_bits`
using a symmetric linear quantizer, and immediately dequantize it back to
float32. The returned tensor therefore has exactly the numerical precision an
N-bit fixed-point FPGA datapath would produce, while remaining a regular
float32 tensor so the rest of the (unmodified) EvT model code keeps working.

Only symmetric, zero-centered quantization is implemented (zero_point == 0),
which is the standard choice for weight quantization targeting FPGA/ASIC
fixed-point multipliers (signed N-bit weight x signed/unsigned activation).
"""

import torch


def compute_qparams(tensor, num_bits, per_channel=False, channel_dim=0, eps=1e-8):
    """Compute the symmetric quantization scale for `tensor`.

    Args:
        tensor: float tensor to be quantized.
        num_bits: target bit-width (signed range is [-2^(b-1), 2^(b-1)-1]).
        per_channel: if True, compute one scale per slice along `channel_dim`
            (row-wise for a Linear/MHA weight, i.e. per output neuron).
        channel_dim: dimension kept when computing per-channel statistics.
        eps: numerical floor to avoid division by zero for all-zero channels.

    Returns:
        scale: tensor broadcastable to `tensor`'s shape.
    """
    qmax = 2 ** (num_bits - 1) - 1
    t = tensor.detach()
    if per_channel and t.dim() > 1:
        reduce_dims = [d for d in range(t.dim()) if d != channel_dim]
        max_abs = t.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        max_abs = t.abs().max()
    scale = max_abs.clamp(min=eps) / qmax
    return scale


def fake_quantize(tensor, scale, num_bits):
    """Quantize `tensor` to `num_bits` (signed, symmetric) and dequantize back."""
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax - 1
    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q * scale


def quantize_to_int(tensor, scale, num_bits):
    """Return the actual low-bit integer tensor (for size accounting / export)."""
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax - 1
    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    storage_dtype = torch.int8 if num_bits <= 8 else torch.int16
    return q.to(storage_dtype)


def fake_quantize_tensor(tensor, num_bits, per_channel=False, channel_dim=0):
    """Convenience wrapper: compute qparams from `tensor` itself and fake-quantize it."""
    scale = compute_qparams(tensor, num_bits, per_channel=per_channel, channel_dim=channel_dim)
    return fake_quantize(tensor, scale, num_bits)
