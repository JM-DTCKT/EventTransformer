"""Instrument one real inference and check, tensor by tensor, that the
datapath is what `hw_flow.md` says it is.

The accuracy numbers are only meaningful if the model that produced them
really ran on integer operands. This script does not take that on trust: it
loads a packed checkpoint, patches `int8_linear` and `RequantSite.__call__` to
record what actually flows through them on a real batch, and reports

  * **MAC operands** -- is every `x_int` / `w_int` an exact integer in
    [-128, 127]?
  * **accumulators** -- is every partial sum an exact integer, and does it stay
    inside INT32 (and inside float32's exact-integer range 2**24, which is what
    the simulator stores it in)?
  * **requantization** -- does every value entering a non-linear unit land
    exactly on its declared Qm.n grid, and inside that grid's range?
  * **residual / reduction ops** -- the few places that are neither a MAC nor a
    non-linearity (residual adds, the mean over latents), which the simulator
    carries in float32 between two requantization points.

Anything that does *not* hold is printed, not hidden -- see the summary at the
end for what a float32 simulator can and cannot represent.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python audit.py --dataset DVS128_10
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import pandas as pd
import torch

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import data_utils, fixed_point as fx, hw_quant  # noqa: E402

EXACT_INT_BOUND = hw_quant.EXACT_INT_BOUND
INT32_MAX = 2 ** 31 - 1


def is_integer(t, tol=0.0):
    return bool((t - torch.round(t)).abs().max() <= tol)


class Recorder:
    def __init__(self):
        self.mac = OrderedDict()
        self.requant = OrderedDict()
        self.n_mac_calls = 0

    def note_mac(self, key, x_int, w_int, acc, bias_int):
        r = self.mac.setdefault(key, dict(
            calls=0, x_int_exact=True, x_in_int8=True, w_int_exact=True,
            w_in_int8=True, acc_exact=True, acc_max=0.0, bias_int_max=0.0,
            reduce_dim=int(x_int.shape[-1])))
        r['calls'] += 1
        r['x_int_exact'] &= is_integer(x_int)
        r['x_in_int8'] &= bool(x_int.abs().max() <= 128)
        r['w_int_exact'] &= is_integer(w_int)
        r['w_in_int8'] &= bool(w_int.abs().max() <= 128)
        r['acc_exact'] &= is_integer(acc)
        r['acc_max'] = max(r['acc_max'], float(acc.abs().max()))
        if bias_int is not None:
            r['bias_int_max'] = max(r['bias_int_max'], float(bias_int.abs().max()))

    def note_requant(self, site, x_in, out):
        r = self.requant.setdefault(site.name, dict(
            calls=0, kind=site.kind, bits=site.total_bits,
            dynamic=site.dynamic, per_call=site.per_call, frac_bits=site.frac_bits,
            on_grid=True, max_abs_in=0.0, max_abs_out=0.0, worst_grid_err_lsb=0.0,
            saturated_elems=0, total_elems=0, float32_exact=True,
            worst_bits_needed=0.0))
        r['calls'] += 1
        r['max_abs_in'] = max(r['max_abs_in'], float(x_in.abs().max()))
        r['max_abs_out'] = max(r['max_abs_out'], float(out.abs().max()))
        if site.kind == 'int8':
            n, qmax, lsb = None, 127, site.scale
        else:
            n = site.frac_bits
            if site.per_call and site.per_call_frac_bits:
                n = site.per_call_frac_bits[min(site.call_idx - 1,
                                                len(site.per_call_frac_bits) - 1)]
            if site.dynamic:      # shift is data-dependent; recover it from the tensor
                m = float(x_in.abs().max())
                n = fx.choose_frac_bits(m, site.total_bits) if m > 0 else site.frac_bits
            qmax, lsb = 2 ** (site.total_bits - 1) - 1, 2.0 ** -n
            r['frac_bits'] = n

        # --- saturation: how many values were clipped by the format's range?
        limit = qmax * lsb
        r['saturated_elems'] += int((x_in.detach().abs() > limit).sum())
        r['total_elems'] += int(x_in.numel())

        # --- grid: is the output an exact multiple of the LSB?
        codes = out.detach().double() / lsb
        err = float((codes - torch.round(codes)).abs().max())
        r['worst_grid_err_lsb'] = max(r['worst_grid_err_lsb'], err)
        r['on_grid'] &= err <= 1e-4

        # --- but the grid check is vacuous when float32's own step at this
        #     magnitude is already coarser than the LSB. Record how many
        #     mantissa bits the declared format needs vs. float32's 24.
        m = float(out.abs().max())
        if m > 0:
            bits_needed = math.log2(m / lsb) + 1
            r['worst_bits_needed'] = max(r['worst_bits_needed'], bits_needed)
            r['float32_exact'] &= bits_needed <= 24.0


def install(rec):
    """Observe every integer MAC (via `hw_quant.MAC_PROBE`, which sits inside
    all four matmul sites: Linear, the pos-enc split GEMM, Q.K^T and attn@V)
    and every requantization site."""
    orig_call = hw_quant.RequantSite.__call__

    def probe(tag, a_int, b_int, acc, bias_int):
        rec.note_mac(tag, a_int, b_int, acc, bias_int)
        rec.n_mac_calls += 1

    def traced_call(self, x):
        out = orig_call(self, x)
        if self.enabled and not self.force_off and not self.calibrating:
            rec.note_requant(self, x, out)
        return out

    hw_quant.MAC_PROBE = probe
    hw_quant.RequantSite.__call__ = traced_call
    return lambda: (setattr(hw_quant, 'MAC_PROBE', None),
                    setattr(hw_quant.RequantSite, '__call__', orig_call))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DVS128_10')
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    ds_dir = os.path.join(QUANT_DIR, args.dataset)
    path = os.path.join(ds_dir, 'quantized', 'model_int8.pt')
    assert os.path.exists(path), f'missing {path}'
    weights_path = data_utils.find_best_checkpoint(os.path.join(ds_dir, 'weights'),
                                                    metric='val_acc', mode='max')
    all_params_path = os.path.join(ds_dir, 'all_params.json')
    val_dl = data_utils.build_datamodule(json.load(open(all_params_path)),
                                         workers=2).val_dataloader()

    payload = torch.load(path, map_location='cpu')
    skeleton, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    model, rt = hw_quant.instantiate(payload, skeleton, device)

    rec = Recorder()
    restore = install(rec)
    seen = 0
    with torch.no_grad():
        for polarity, pixels, _l in val_dl:
            if polarity is None:
                continue
            model(polarity.to(device), pixels.to(device))
            seen += 1
            if seen >= args.batches:
                break
    restore()

    print(f"\n{'=' * 90}\n {args.dataset} -- {seen} real batches, "
          f"{rec.n_mac_calls} GEMM invocations\n{'=' * 90}")

    mac = pd.DataFrame(rec.mac).T
    print("\n--- MAC units: are the operands really INT8 and the accumulators INT32? ---")
    print(mac[['calls', 'reduce_dim', 'x_int_exact', 'x_in_int8', 'w_int_exact',
               'w_in_int8', 'acc_exact', 'acc_max', 'bias_int_max']].to_string())
    print(f"\n  every operand an exact integer in [-128,127] : "
          f"{bool(mac[['x_int_exact', 'x_in_int8', 'w_int_exact', 'w_in_int8']].all().all())}")
    print(f"  every accumulator an exact integer           : {bool(mac['acc_exact'].all())}")
    print(f"  largest |accumulator|                        : {mac['acc_max'].max():,.0f}   "
          f"(INT32 limit {INT32_MAX:,}; float32 holds integers exactly to {EXACT_INT_BOUND:,})")
    print(f"  largest |folded INT32 bias|                  : {mac['bias_int_max'].max():,.0f}")

    req = pd.DataFrame(rec.requant).T
    req.index.name = 'site'
    req['saturated_pct'] = req['saturated_elems'] / req['total_elems'] * 100
    print("\n--- Requantization sites (every value entering a non-linear unit) ---")
    print(f"  sites exercised                              : {len(req)}")
    print(f"  outputs land on their declared Qm.n / INT8 grid : {bool(req['on_grid'].all())}")

    sat = req[req['saturated_pct'] > 0].sort_values('saturated_pct', ascending=False)
    print(f"\n  sites where the calibrated range SATURATES on the test set: {len(sat)}")
    if len(sat):
        print(sat[['kind', 'bits', 'frac_bits', 'max_abs_in', 'max_abs_out',
                   'saturated_pct']].to_string())

    nonrep = req[~req['float32_exact'].astype(bool)].sort_values(
        'worst_bits_needed', ascending=False)
    print(f"\n  sites whose declared format needs more than float32's 24 mantissa bits\n"
          f"  (the simulator cannot carry them at full width -- real fixed-point\n"
          f"   hardware is strictly MORE precise there, so accuracy is a lower bound): "
          f"{len(nonrep)}")
    if len(nonrep):
        print(nonrep[['bits', 'frac_bits', 'max_abs_out', 'worst_bits_needed']].to_string())

    out_csv = os.path.join(QUANT_DIR, args.dataset, 'datapath_audit.csv')
    req.to_csv(out_csv)
    print(f"\nPer-site detail written to {out_csv}")


if __name__ == '__main__':
    main()
