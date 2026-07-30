"""Model size / compute complexity accounting for the quantization comparison.

Two independent things are estimated per quantization case:

  1. Model size (bytes): the *ideal* packed size if weights were actually
     stored at `weight_bits`, i.e. what would be written to FPGA BRAM/flash.
     Quantizable weights (Linear + MultiheadAttention in_proj) are counted at
     `weight_bits` plus a small fp32 scale-factor overhead (per output
     channel, or per tensor for per-tensor quantization); everything else
     (biases, LayerNorm, positional encoding, latent memory vectors) stays
     fp32 since it is never quantized (see quant_wrapper.py).

  2. Compute: MACs are measured once (via ptflops, fp32 baseline model) and
     reused for every case, since fake-quantization does not change the
     number of multiply-accumulates -- only their bit-width. From that we
     derive BOPs (bit-operations = MACs * weight_bits * act_bits), the
     standard proxy for FPGA DSP/LUT cost of a given precision combination.
     For weight-only cases activations stay fp32 (act_bits=32) since this
     PTQ simulation dequantizes weights back to fp32 before the matmul; a
     real weight-only FPGA design would still need a narrower activation
     datapath to realize compute savings, which is exactly what the W8A8 /
     W4A8 cases are meant to demonstrate.
"""

import numpy as np
import torch.nn as nn

FP32_BITS = 32
SCALE_BITS = 32  # fp32 scale factor stored per (channel|tensor)


def count_quantizable_params(model):
    """Returns (quantizable_params, other_params, num_tensors, num_channels)."""
    quantizable, num_tensors, num_channels = 0, 0, 0
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            quantizable += m.weight.numel()
            num_tensors += 1
            num_channels += m.weight.shape[0]
        elif isinstance(m, nn.MultiheadAttention):
            quantizable += m.in_proj_weight.numel()
            num_tensors += 1
            num_channels += m.in_proj_weight.shape[0]
    total = sum(p.numel() for p in model.parameters())
    other = total - quantizable
    return quantizable, other, num_tensors, num_channels


def estimate_model_size_bytes(model, weight_bits=None, per_channel=True):
    """Ideal packed model size (bytes) for a given weight bit-width."""
    quant_params, other_params, num_tensors, num_channels = count_quantizable_params(model)
    other_bytes = other_params * FP32_BITS / 8

    if weight_bits is None or weight_bits >= FP32_BITS:
        return quant_params * FP32_BITS / 8 + other_bytes

    weight_bytes = quant_params * weight_bits / 8
    overhead_units = num_channels if per_channel else num_tensors
    overhead_bytes = overhead_units * SCALE_BITS / 8
    return weight_bytes + overhead_bytes + other_bytes


def estimate_macs(model, dataloader, device, max_batches=8, max_timesteps_per_batch=4):
    """Average MACs per sample, measured with ptflops on a few real batches.

    Only needs to be computed once per model (fp32, unquantized): quantization
    changes weight/activation precision, not the number of MACs.
    """
    from ptflops import get_model_complexity_info
    import torch

    model.eval()
    macs_list = []
    seen_batches = 0
    with torch.no_grad():
        for polarity, pixels, _labels in dataloader:
            if polarity is None:
                continue
            polarity = polarity.to(device)
            pixels = pixels.to(device)
            num_timesteps = polarity.shape[0]
            step = max(1, num_timesteps // max_timesteps_per_batch)
            for ts in range(0, num_timesteps, step):
                mask = polarity[ts:ts + 1].sum(-1).sum(0).sum(0) != 0
                if mask.sum() == 0:
                    continue
                pol_t = polarity[ts:ts + 1][:, :, mask, :]
                pix_t = pixels[ts:ts + 1][:, :, mask, :]
                macs, _ = get_model_complexity_info(
                    model.backbone, ({'kv': pol_t, 'pixels': pix_t},),
                    input_constructor=lambda x: x[0],
                    as_strings=False, print_per_layer_stat=False, verbose=False,
                )
                macs_list.append(macs)
            seen_batches += 1
            if seen_batches >= max_batches:
                break
    return float(np.mean(macs_list)) if macs_list else 0.0


def compute_bops(macs, weight_bits, act_bits):
    w = weight_bits if weight_bits else FP32_BITS
    a = act_bits if act_bits else FP32_BITS
    return macs * w * a


def bops_reduction_ratio(macs, weight_bits, act_bits):
    baseline_bops = compute_bops(macs, FP32_BITS, FP32_BITS)
    case_bops = compute_bops(macs, weight_bits, act_bits)
    return baseline_bops / case_bops if case_bops > 0 else float('inf')
