"""16-bit fixed-point (Qm.n) primitives shared by the hardware-oriented
quantization path (`hw_quant.py`).

Everything the FPGA datapath touches has to be an integer: INT8 operands,
INT32 accumulators, and -- for the scales and for the values entering the
non-linear units (GELU / LayerNorm / softmax) -- 16-bit *fixed point*.

A Qm.n number is a 16-bit two's-complement integer `c` interpreted as
`c * 2^-n`; `m = 15 - n` is the number of integer bits (the sign bit is
counted separately, so `Q5.10` here means 1 sign + 5 integer + 10 fractional
= 16 bits, representable range [-32, +32) with a step of 2^-10).

`n` is *not* fixed globally: it is chosen per tensor (or per requantization
site) from the calibrated dynamic range, which is exactly what a real design
does -- each layer's rescale is an int16 multiplier plus a per-layer shift
amount. `choose_frac_bits` picks the largest `n` that keeps the tensor's
maximum magnitude inside the int16 range, i.e. the format that wastes no
headroom. `analyze_*` below reports what those formats come out to be so the
choice can be inspected rather than taken on faith.
"""

import math

import torch

TOTAL_BITS = 16
FX_QMAX = 2 ** (TOTAL_BITS - 1) - 1     # +32767
FX_QMIN = -2 ** (TOTAL_BITS - 1)        # -32768
FRAC_CAP = 30                           # max shift amount a sane HW rescale unit would use


def choose_frac_bits(max_abs, total_bits=TOTAL_BITS, frac_cap=FRAC_CAP):
    """Largest number of fractional bits `n` such that `max_abs` still fits in
    a signed `total_bits` integer after scaling by 2^n (no saturation)."""
    qmax = 2 ** (total_bits - 1) - 1
    max_abs = float(max_abs) if max_abs is not None else 0.0
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        return frac_cap
    n = int(math.floor(math.log2(qmax / max_abs)))
    while max_abs * (2.0 ** n) > qmax:      # guard against log2 rounding at the boundary
        n -= 1
    while max_abs * (2.0 ** (n + 1)) <= qmax:
        n += 1
    return int(min(max(n, -frac_cap), frac_cap))


def q_format_name(frac_bits, total_bits=TOTAL_BITS):
    """'Q5.10'-style name (sign bit implicit, so m + n = total_bits - 1)."""
    return f"Q{total_bits - 1 - frac_bits}.{frac_bits}"


def fx_codes(x, frac_bits, total_bits=TOTAL_BITS):
    """Fixed-point integer codes of `x`, returned as a float tensor holding
    exact integers. float32 only represents integers exactly below 2**24, so
    anything wider than 24 bits is computed in float64 -- a simulation detail;
    the hardware register is an ordinary two's-complement integer."""
    qmax = 2 ** (total_bits - 1) - 1
    if total_bits > 24:
        return torch.clamp(torch.round(x.double() * (2.0 ** frac_bits)), -qmax - 1, qmax)
    return torch.clamp(torch.round(x * (2.0 ** frac_bits)), -qmax - 1, qmax)


def fx_round(x, frac_bits, total_bits=TOTAL_BITS):
    """`x` snapped onto the Qm.n grid (quantize -> dequantize round trip)."""
    out = fx_codes(x, frac_bits, total_bits) * (2.0 ** -frac_bits)
    return out.to(x.dtype) if out.dtype != x.dtype else out


def fx_round_scalar(x, frac_bits=None, total_bits=TOTAL_BITS):
    """Scalar version; picks the format from the value itself when
    `frac_bits` is None. Returns (rounded_value, frac_bits)."""
    if frac_bits is None:
        frac_bits = choose_frac_bits(abs(float(x)), total_bits)
    qmax = 2 ** (total_bits - 1) - 1
    code = max(-qmax - 1, min(qmax, int(round(float(x) * (2.0 ** frac_bits)))))
    return code * (2.0 ** -frac_bits), frac_bits


def pack_fx16(t, frac_bits=None):
    """Tensor -> (int16 codes, frac_bits) with a per-tensor shared exponent."""
    if frac_bits is None:
        frac_bits = choose_frac_bits(float(t.detach().abs().max()))
    return fx_codes(t.detach(), frac_bits).to(torch.int16), int(frac_bits)


def unpack_fx16(codes, frac_bits):
    return codes.to(torch.float32) * (2.0 ** -int(frac_bits))


def pack_int8(t, scale):
    """Symmetric INT8 codes of `t` under `scale` (tensor or scalar)."""
    return torch.clamp(torch.round(t.detach() / scale), -128, 127).to(torch.int8)


def relative_error(original, reconstructed):
    """max and mean relative error, for reporting how much a Q-format costs."""
    o = original.detach().float().flatten()
    r = reconstructed.detach().float().flatten()
    denom = o.abs().clamp(min=1e-12)
    rel = (r - o).abs() / denom
    return float(rel.max()), float(rel.mean())


def analyze_tensor(t, name, total_bits=TOTAL_BITS):
    """Distribution summary + the Qm.n format that would be picked for `t`."""
    t = t.detach().float().flatten()
    a = t.abs()
    nz = a[a > 0]
    max_abs = float(a.max()) if a.numel() else 0.0
    min_abs = float(nz.min()) if nz.numel() else 0.0
    frac_bits = choose_frac_bits(max_abs, total_bits)
    rec = fx_round(t, frac_bits, total_bits)
    rel_max, rel_mean = relative_error(t, rec)
    return dict(
        name=name, numel=int(t.numel()), max_abs=max_abs, min_abs_nonzero=min_abs,
        dynamic_range=(max_abs / min_abs) if min_abs > 0 else float('inf'),
        frac_bits=frac_bits, q_format=q_format_name(frac_bits, total_bits),
        step=2.0 ** -frac_bits, rel_err_max=rel_max, rel_err_mean=rel_mean,
    )
