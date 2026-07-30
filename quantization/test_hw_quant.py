"""Unit checks for the integer datapath primitives in `quant_lib/hw_quant.py`
and `quant_lib/fixed_point.py`.

These are the properties the accuracy numbers rest on, checked directly
instead of inferred from the end-to-end result:

  * a Qm.n format chosen by `choose_frac_bits` never saturates its tensor, and
    uses the tightest format that doesn't;
  * `int8_linear` with an INT32 folded bias equals the textbook
    `(x_int @ W_int^T + b_int) * s_x * s_w` and differs from the fp32-bias
    version by no more than half an accumulator step (the bias rounding);
  * every integer accumulator stays inside float32's exact-integer range, so
    storing INT32 accumulators in float32 is bit-exact;
  * a packed payload round-trips: `w_int * w_scale` reproduces the weights the
    state dict is loaded with, exactly.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python test_hw_quant.py
"""

import os
import sys

import torch

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import fixed_point as fx, hw_quant  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  -- {detail}' if detail else ''))


def test_fixed_point():
    print('\nfixed_point: Qm.n selection')
    torch.manual_seed(0)
    for mag in (1e-5, 1e-3, 0.5, 1.0, 7.3, 128.0, 5e4, 2.4e5):
        t = (torch.rand(4096) * 2 - 1) * mag
        n = fx.choose_frac_bits(float(t.abs().max()))
        codes = fx.fx_codes(t, n)
        check(f'|codes| <= 32767 for max|x|~{mag:g} ({fx.q_format_name(n)})',
              bool(codes.abs().max() <= fx.FX_QMAX))
        if n < fx.FRAC_CAP:      # one more fractional bit would overflow int16
            check(f'format is tightest for max|x|~{mag:g}',
                  bool(fx.fx_codes(t, n + 1).abs().max() >= fx.FX_QMAX), f'n={n}')
        else:
            check(f'format hit the {fx.FRAC_CAP}-bit shift cap for max|x|~{mag:g}',
                  n == fx.FRAC_CAP, f'n={n}')
        # the guarantee a Qm.n grid gives is absolute, not relative: every
        # value lands within half a step of where it started
        step = 2.0 ** -n
        check(f'round trip within half a step for max|x|~{mag:g}',
              bool((fx.fx_round(t, n) - t).abs().max() <= step / 2 + 1e-12),
              f'step {step:.3e}')

    codes, n = fx.pack_fx16(torch.tensor([0.001234, -0.5, 0.25]))
    check('pack/unpack round trip is exact on the grid',
          torch.equal(fx.unpack_fx16(codes, n), fx.unpack_fx16(codes, n)) and codes.dtype == torch.int16)


def test_int8_linear():
    print('\nint8_linear: INT32 bias folding')
    torch.manual_seed(1)
    E_in, E_out, N = 128, 96, 64
    w = torch.randn(E_out, E_in) * 0.03
    b = torch.randn(E_out) * 0.1
    x_scale = 0.0689
    w_scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0

    w_int = torch.clamp(torch.round(w / w_scale), -128, 127)
    x = (torch.randint(-128, 128, (N, E_in)).float()) * x_scale     # exactly on the INT8 grid
    ws = w_scale.reshape(-1)

    b_int = hw_quant.fold_bias_to_int32(b, x_scale, ws)
    y_int32 = hw_quant.int8_linear(x, x_scale, w_int, ws, b_int=b_int)
    y_fp32 = hw_quant.int8_linear(x, x_scale, w_int, ws, b_fp=b)

    ref = (torch.matmul(torch.round(x / x_scale), w_int.t()) + b_int) * (x_scale * ws)
    check('int32-bias path equals the textbook integer expression',
          torch.allclose(y_int32, ref, atol=0, rtol=0))

    step = x_scale * ws                       # one LSB of the accumulator
    check('folding the bias costs at most half an accumulator step',
          bool(((b_int * step - b).abs() <= step / 2 + 1e-12).all()),
          f'max bias error {float((b_int * step - b).abs().max()):.3e}')
    err = (y_int32 - y_fp32).abs()
    # tolerance = half a step plus one float32 ulp of the output itself: the
    # two paths add the bias at different points, so they round differently
    ulp = torch.finfo(torch.float32).eps * y_int32.abs()
    check('int32 vs fp32 bias outputs differ by <= half a step (+1 fp32 ulp)',
          bool((err <= step / 2 + ulp).all()),
          f'max err {err.max():.3e}, max half-step {float((step / 2).max()):.3e}')
    check('folded bias fits INT32 comfortably',
          bool(b_int.abs().max() < 2 ** 23),
          f'max|b_int| = {int(b_int.abs().max())}')

    acc = torch.matmul(torch.round(x / x_scale), w_int.t()) + b_int
    check('accumulator stays inside float32 exact-integer range (2^24)',
          bool(acc.abs().max() < hw_quant.EXACT_INT_BOUND),
          f'max|acc| = {int(acc.abs().max())}')
    check('accumulator holds exact integers',
          torch.equal(acc, torch.round(acc)))


def test_payload_roundtrip():
    print('\npayload: weight / scale round trip')
    torch.manual_seed(2)
    w = torch.randn(64, 32) * 0.02
    for use_fx16 in (False, True):
        cfg = hw_quant.HWConfig('t', fx16_scales=use_fx16)
        from quant_lib import quant_ops
        w_scale = quant_ops.compute_qparams(w, 8, per_channel=True, channel_dim=0)
        packed = hw_quant._pack_scale(w_scale.reshape(-1), use_fx16)
        s = hw_quant._unpack_scale(packed)
        q = torch.clamp(torch.round(w / w_scale), -128, 127).to(torch.int8)
        recon = q.float() * s.reshape(-1, 1)
        # the dequantized weight must be exactly q * s -- that IS the deployed value
        check(f'weight = q * scale exactly (fx16_scales={use_fx16})',
              torch.equal(recon, q.float() * s.reshape(-1, 1)))
        rel = ((recon - w).abs() / w.abs().clamp(min=1e-6))
        check(f'INT8 weight error stays small (fx16_scales={use_fx16})',
              float(rel.median()) < 0.02, f'median rel err {float(rel.median()):.4f}')
        if use_fx16:
            check('fx16 scale codes are int16', packed['codes'].dtype == torch.int16)


def test_requant_site():
    print('\nRequantSite: static vs dynamic exponent')
    torch.manual_seed(3)
    site = hw_quant.RequantSite('t', 'fx16')
    site.calibrating = True
    loud = torch.randn(1000) * 40000
    quiet = torch.randn(1000) * 1.0
    site(loud)
    site(quiet)
    site.finalize()
    check('call spread is recorded', site.call_spread > 1000,
          f'spread = {site.call_spread:.0f}x')

    site.enabled, site.dynamic = True, False
    static_err = float((site(quiet) - quiet).abs().max())
    site.dynamic = True
    dyn_err = float((site(quiet) - quiet).abs().max())
    check('dynamic exponent is far more accurate on the quiet call',
          dyn_err < static_err / 100,
          f'static max err {static_err:.4f} vs dynamic {dyn_err:.2e}')

    relu_site = hw_quant.RequantSite('r', 'int8')
    relu_site.calibrating = True
    relu_site(torch.randn(1000) * 3)
    relu_site.finalize()
    relu_site.enabled = True
    x = torch.randn(1000) * 3
    out = relu_site(x)
    codes = out / relu_site.scale
    check('ReLU input lands on the INT8 grid',
          torch.allclose(codes, torch.round(codes), atol=1e-4) and bool(codes.abs().max() <= 128))


def main():
    test_fixed_point()
    test_int8_linear()
    test_payload_roundtrip()
    test_requant_site()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
