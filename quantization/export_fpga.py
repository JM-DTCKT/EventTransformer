"""Turn a packed checkpoint into the exact integer constants an RTL block needs,
and *verify* that the integer-only formula reproduces what the simulator did.

The `.pt` files store scales as `(int16 code, frac_bits)` pairs and weights as
INT8 + per-channel scale. A GEMM block on the board does not want any of that
at run time -- it wants, per output channel, one multiplier and one shift. This
script folds the stored pairs into that form:

    acc[c]      = SUM_k x_int[k] * w_int[c,k] + b_int[c]          INT32
    out_code[c] = (acc[c] * M[c] + (1 << (k-1))) >> k             requantize
                  saturated to the consumer's width

where `M[c]` and `k` come from `s_x * s_w[c] / lsb_out` -- the layer's own
scales divided by the LSB of whatever consumes its output (a Qm.n grid in front
of a non-linear unit, or an INT8 step in front of the next GEMM).

`--verify` runs a real batch and checks `out_code` against the simulator's
output for every layer, so the constants in `fpga_export/` are known to be the
ones that produced the reported accuracy, not a re-derivation that drifted.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python export_fpga.py --dataset DVS128_10
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import data_utils, hw_quant  # noqa: E402
from quant_lib.hw_quant import _unpack_scale  # noqa: E402

M_BITS = 32          # width of the requantization multiplier (see --m_bits)


# =============================================================================
# Who consumes each GEMM's output, and on what grid
# =============================================================================
def build_consumer_map(model, payload, rt):
    """gemm state-dict key -> {'lsb': float, 'width': int, 'signed_min/max',
    'name': str} describing the format its output has to land on.

    Every GEMM output goes to exactly one of:
      * a Qm.n grid in front of a non-linear unit (LayerNorm / GELU / softmax),
      * an INT8 step in front of the next GEMM (ReLU inputs, and the attention
        Q/K/V/out_proj operands),
      * the residual stream, which is the LayerNorm-input Qm.n grid.
    """
    sites = rt.sites

    def q_target(site_name):
        s = sites[site_name]
        assert s.kind == 'fx16', site_name
        return dict(name=site_name, kind='fixed', lsb=2.0 ** -s.frac_bits,
                    width=s.total_bits, frac_bits=s.frac_bits,
                    qmax=2 ** (s.total_bits - 1) - 1)

    def bf16_target(site_name):
        """LayerNorm's datapath is bfloat16, so its input register is bf16 --
        the producing GEMM converts its INT32 accumulator and rounds to bf16.
        It does NOT snap to a Qm.n grid, and there is no M/shift for it."""
        return dict(name=site_name, kind='bfloat16', width=16,
                    lsb=None, frac_bits=None, qmax=None)

    def int8_target(name, step):
        return dict(name=name, kind='int8', lsb=float(step), width=8,
                    frac_bits=None, qmax=127)

    cons = {}
    from models.EvT import AttentionBlock

    cons['backbone.event_projection.seq_init.0.weight'] = \
        q_target('backbone.event_projection.seq_init.1.in')
    cons['backbone.preproc_block_events.seq_init.0.weight'] = \
        q_target('backbone.preproc_block_events.seq_init.1.in')
    for i, j in ((1, 2), (4, 5)):
        cons[f'backbone.proc_event_blocks.0.seq_init.{i}.weight'] = int8_target(
            f'backbone.proc_event_blocks.0.seq_init.{j}.in',
            sites[f'backbone.proc_event_blocks.0.seq_init.{j}.in'].scale)

    blocks = [n for n, m in model.named_modules() if isinstance(m, AttentionBlock)]
    for blk in blocks:
        # in_proj: the three row bands are requantized straight to INT8 with the
        # post-projection Q / K / V scales
        cons[f'{blk}.attention.in_proj_weight'] = dict(
            name=f'{blk}.attention.in_proj', kind='int8_banded', width=8, qmax=127,
            bands=[('Q', rt.attn_scale('qk_scales', blk, 'q_scale')),
                   ('K', rt.attn_scale('qk_scales', blk, 'k_scale')),
                   ('V', rt.attn_scale('v_scale', blk))])
        # out_proj and linear3 feed the residual stream
        cons[f'{blk}.attention.out_proj.weight'] = bf16_target(f'{blk}.layer_norm_att.in')
        cons[f'{blk}.linear1.weight'] = q_target(f'{blk}.gelu1.in')
        cons[f'{blk}.linear2.weight'] = q_target(f'{blk}.gelu2.in')
        cons[f'{blk}.linear3.weight'] = bf16_target(f'{blk}.layer_norm_att.in')

    cons['backbone.proc_embs_block.linear1.weight'] = int8_target(
        'backbone.proc_embs_block.relu.in', sites['backbone.proc_embs_block.relu.in'].scale)
    cons['models_clf.0.linear_1.weight'] = int8_target(
        'models_clf.0.relu.in', sites['models_clf.0.relu.in'].scale)
    # the classifier head ends at argmax: no output grid, no shift, no
    # saturation -- but M[c] still has to be applied, because the per-class
    # weight scale would otherwise reorder the classes
    cons['models_clf.0.linear_2.weight'] = dict(
        name='models_clf.0.argmax', kind='argmax', width=None,
        lsb=1.0, frac_bits=None, qmax=None)
    return cons


# =============================================================================
# Fold (scale pair, consumer LSB) into one multiplier + one shift
# =============================================================================
def make_requant(ratio, m_bits=None):
    """ratio[c] = s_x * s_w[c] / lsb_out  ->  (M[c] int, shift k) with
    M[c] / 2**k ~= ratio[c] and max|M| just under 2**(m_bits-1).

    `m_bits` defaults to the module-level `M_BITS` *at call time*, so
    `--m_bits` actually takes effect (a default argument would be frozen at
    def time)."""
    m_bits = M_BITS if m_bits is None else m_bits
    ratio = np.asarray(ratio, dtype=np.float64).reshape(-1)
    mx = float(np.abs(ratio).max())
    assert mx > 0
    k = int(math.floor(math.log2((2 ** (m_bits - 1) - 1) / mx)))
    M = np.round(ratio * (2.0 ** k)).astype(np.int64)
    while np.abs(M).max() > 2 ** (m_bits - 1) - 1:
        k -= 1
        M = np.round(ratio * (2.0 ** k)).astype(np.int64)
    return M, k


def requantize(acc, M, k, qmax):
    """The RTL operation: multiply, round-to-nearest by adding half an LSB,
    arithmetic shift right, saturate."""
    acc = acc.astype(np.int64)
    prod = acc * M
    out = (prod + (1 << (k - 1))) >> k if k > 0 else prod << (-k)
    return np.clip(out, -qmax - 1, qmax)


# =============================================================================
def export(payload, model, rt, out_dir):
    cons = build_consumer_map(model, payload, rt)
    if os.path.isdir(out_dir):      # never leave blobs from a previous config
        for f in os.listdir(out_dir):
            if f.endswith(('.bin', '.json', '.csv')):
                os.remove(os.path.join(out_dir, f))
    os.makedirs(out_dir, exist_ok=True)
    blobs, manifest = {}, dict(
        format=payload['format'], config=payload['config'],
        note='out_code[c] = sat((acc[c]*M[c] + 2**(k-1)) >> k); '
             'acc[c] = SUM x_int*w_int + b_int',
        layers=[])

    for key, entry in payload['layers'].items():
        base = key.rsplit('.', 1)[0]
        w_scale = _unpack_scale(entry['scale']).numpy().astype(np.float64)
        rec = dict(name=base, kind=entry['kind'], shape=list(entry['shape']),
                   weight_file=f'{base}.W.int8.bin',
                   weight_dtype='int8', weight_layout='[E_out][E_in] row-major')
        blobs[rec['weight_file']] = entry['q'].numpy()

        if entry.get('bias'):
            b = entry['bias']
            rec['bias_file'] = f'{base}.B.int32.bin'
            rec['bias_dtype'] = 'int32'
            blobs[rec['bias_file']] = b['q'].numpy()

        c = cons.get(key)
        if c is None:
            manifest['layers'].append(rec)
            continue

        if entry['kind'] == 'linear':
            x_scale = float(_unpack_scale(entry['x_scale'])[0])
            rec['input'] = dict(dtype='int8', step=x_scale,
                                note='the producing stage already emits these codes')
            if c['kind'] == 'argmax':
                M, k = make_requant(x_scale * w_scale)
                rec['requant'] = dict(mult_file=f'{base}.M.int32.bin', shift=k,
                                      note='argmax(acc[c]*M[c]) -- the shift is '
                                           'order-preserving and may be skipped')
                blobs[f'{base}.M.int32.bin'] = M.astype(np.int32)
            elif c['kind'] == 'bfloat16':
                # int32 acc -> multiply by (s_x * s_w[c]) -> round to bf16.
                # One bf16 constant per output channel, no shift.
                sc = torch.tensor(x_scale * w_scale, dtype=torch.float32)
                rec['requant'] = dict(
                    kind='bf16', scale_file=f'{base}.S.bf16.bin',
                    note='out = bf16(acc * scale[c]); no integer shift')
                blobs[f'{base}.S.bf16.bin'] = (
                    sc.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16))
            else:
                M, k = make_requant(x_scale * w_scale / c['lsb'])
                rec['requant'] = dict(mult_file=f'{base}.M.int32.bin', shift=k)
                blobs[f'{base}.M.int32.bin'] = M.astype(np.int32)
        else:                              # in_proj: three row bands
            E = entry['embed_dim']
            x_rows = np.concatenate([
                np.full(E, rt.attn_scale('q_in_scales', base[: -len('.attention')])),
                np.full(E, rt.attn_scale('k_in_scales', base[: -len('.attention')])),
                np.full(E, rt.attn_scale('v_in_scales', base[: -len('.attention')]))])
            lsb_rows = np.concatenate([np.full(E, s) for _, s in c['bands']])
            M, k = make_requant(x_rows * w_scale / lsb_rows)
            # The three bands read *two* different tensors (Q from layer_norm_1,
            # K and V both from layer_norm_x) and each has its own input step.
            # Whatever produces those tensors has to requantize onto this grid,
            # so the step belongs in the manifest -- it is not recoverable from
            # M[c] alone (w_scale is folded in).
            rec['input'] = dict(dtype='int8', per_band=True,
                                bands=[dict(name=n, step=float(x_rows[i * E]))
                                       for i, (n, _s) in enumerate(c['bands'])])
            rec['requant'] = dict(mult_file=f'{base}.M.int32.bin', shift=k,
                                  bands=[dict(name=n, rows=[i * E, (i + 1) * E],
                                              input_step=float(np.full(1, s)[0]))
                                         for i, (n, s) in enumerate(c['bands'])])
            blobs[f'{base}.M.int32.bin'] = M.astype(np.int32)
            for bk in ('bias_k', 'bias_v'):
                if entry.get(bk):
                    rec[f'{bk}_file'] = f'{base}.{bk}.int32.bin'
                    blobs[rec[f'{bk}_file']] = entry[bk]['q'].numpy()
            # bias_k / bias_v are extra *tokens* appended to K and V, not biases
            # on an accumulator: `attention_forward_hw` concatenates them after
            # the projection and then quantizes K/V onto the QK/AV operand grid.
            # So what an accelerator needs is their INT8 code on that grid -- the
            # INT32 form above is the accumulator-domain value and cannot be used
            # directly. `mha.bias_k` already holds the reconstructed value.
            mha = model.get_submodule(base)
            for bk, bi in (('bias_k', 1), ('bias_v', 2)):
                t = getattr(mha, bk, None)
                if t is None:
                    continue
                grid = c['bands'][bi][1]
                code = torch.clamp(torch.round(t.detach().reshape(-1) / grid),
                                   -128, 127)
                rec[f'{bk}_int8_file'] = f'{base}.{bk}.int8.bin'
                rec.setdefault('bias_kv_note',
                               'int8 code on the QK/AV operand grid: '
                               'round(bias / bands[K|V].step)')
                blobs[rec[f'{bk}_int8_file']] = (
                    code.cpu().numpy().astype(np.int8))

        rec['output'] = dict(consumer=c['name'], kind=c['kind'], width=c['width'],
                             lsb=c.get('lsb'), frac_bits=c.get('frac_bits'),
                             qmax=c['qmax'])
        manifest['layers'].append(rec)

    # ---- attention matmuls, the non-linear formats, and the tables ----
    manifest['attention'] = attention_spec(payload, rt, model)
    manifest['nonlinear_formats'] = [
        dict(site=n, kind=s.kind, width=s.total_bits,
             frac_bits=s.frac_bits, q_format=(f'Q{s.total_bits - 1 - s.frac_bits}.{s.frac_bits}'
                                              if s.kind == 'fx16' else 'INT8'),
             lsb=(2.0 ** -s.frac_bits) if s.kind == 'fx16' else s.scale)
        for n, s in sorted(rt.sites.items()) if s.frac_bits is not None or s.scale is not None]

    pe = payload.get('pos_encoding')
    if pe:
        blobs['pos_encoding.int8.bin'] = pe['q'].numpy()
        manifest['pos_encoding'] = dict(file='pos_encoding.int8.bin', dtype='int8',
                                        shape=list(pe['shape']),
                                        step=float(_unpack_scale(pe['scale'])[0]),
                                        note='indexed [y//6][x//6][d]; requantized to the consumer Linear\'s input step')
    manifest['fx_params'] = []
    for k, e in payload['fx_params'].items():
        f = f'{k}.int16.bin'
        blobs[f] = e['codes'].numpy()
        manifest['fx_params'].append(dict(name=k, file=f,
                                          dtype='int16',
                                          frac_bits=e.get('frac_bits'),
                                          shape=list(e.get('shape', ()))))

    total = 0
    for fname, arr in blobs.items():
        p = os.path.join(out_dir, fname.replace('/', '_'))
        arr.tofile(p)
        total += arr.nbytes
    manifest['total_blob_bytes'] = total
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    return manifest, cons, total


def attention_spec(payload, rt, model):
    from models.EvT import AttentionBlock
    out = []
    for blk, m in model.named_modules():
        if not isinstance(m, AttentionBlock):
            continue
        H = m.attention.num_heads
        hd = m.attention.embed_dim // H
        s_q = rt.attn_scale('qk_scales', blk, 'q_scale')
        s_k = rt.attn_scale('qk_scales', blk, 'k_scale')
        s_v = rt.attn_scale('v_scale', blk)
        s_sm = rt.softmax_scale
        inv = rt.const(1.0 / math.sqrt(hd))
        site_scores = rt.sites[f'{blk}.softmax.in']
        lsb_scores = 2.0 ** -site_scores.frac_bits
        M_qk, k_qk = make_requant([s_q * s_k * inv / lsb_scores])
        lsb_out = rt.attn_scale('out_proj_scales', blk)
        M_av, k_av = make_requant([s_sm * s_v / lsb_out])
        out.append(dict(
            block=blk, heads=H, head_dim=hd,
            QK=dict(a='Q_int (int8)', b='K_int (int8)', reduce=hd,
                    mult=int(M_qk[0]), shift=k_qk,
                    note='multiplier already includes 1/sqrt(head_dim)',
                    output=dict(consumer=f'{blk}.softmax.in',
                                frac_bits=site_scores.frac_bits, lsb=lsb_scores)),
            softmax=dict(input_frac_bits=site_scores.frac_bits,
                         output_frac_bits=rt.sites[f'{blk}.softmax.out'].frac_bits,
                         to_int8_step=s_sm, unsigned=True),
            AV=dict(a='attn_int (uint8, 0..127)', b='V_int (int8)',
                    mult=int(M_av[0]), shift=k_av,
                    output=dict(consumer=f'{blk}.attention.out_proj input',
                                kind='int8', step=lsb_out))))
    return out


# =============================================================================
@torch.no_grad()
def verify(payload, model, rt, cons, dataloader, device, batches=1):
    """Run a real batch; for every Linear, recompute the output with the
    integer-only formula and compare against the simulator's."""
    rows = []
    grabs = {}
    taps = {}
    hw_quant.ATTN_TAP = taps          # capture in_proj / out_proj I/O too
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and f'{name}.weight' in cons:
            def pre(mod, args, _n=name):
                grabs[_n] = args[0].detach()
            handles.append(m.register_forward_pre_hook(pre))

            def post(mod, args, output, _n=name):
                grabs[_n + '::out'] = output.detach()
            handles.append(m.register_forward_hook(post))

    seen = 0
    for polarity, pixels, _l in dataloader:
        if polarity is None:
            continue
        model(polarity.to(device), pixels.to(device))
        seen += 1
        if seen >= batches:
            break
    for h in handles:
        h.remove()
    hw_quant.ATTN_TAP = None

    for key, entry in payload['layers'].items():
        if entry['kind'] != 'linear':
            continue
        name = key.rsplit('.', 1)[0]
        if name not in grabs or key not in cons:
            continue
        c = cons[key]
        x = grabs[name].cpu().double().numpy()
        y_sim = grabs[name + '::out'].cpu().double().numpy()
        w_int = entry['q'].numpy().astype(np.int64)
        w_scale = _unpack_scale(entry['scale']).numpy().astype(np.float64)
        b_int = (entry['bias']['q'].numpy().astype(np.int64)
                 if entry.get('bias') is not None else None)
        x_scale = float(_unpack_scale(entry['x_scale'])[0])

        x_int = np.clip(np.round(x / x_scale), -128, 127).astype(np.int64)
        acc = x_int @ w_int.T + (b_int if b_int is not None else 0)
        if c['kind'] == 'argmax':
            M, k = make_requant(x_scale * w_scale)
            pred = (acc * M).argmax(axis=-1)
            ref = y_sim.argmax(axis=-1)
            same = float((pred == ref).mean() * 100)
            rows.append(dict(layer=name, consumer_kind='argmax', out_width=0,
                             exact_pct=same, within_1lsb_pct=same,
                             max_diff_lsb=0.0, n=int(pred.size)))
            continue
        if c['kind'] == 'bfloat16':
            # the hardware value is bf16(acc * s_x * s_w); compare in units of
            # the bf16 ULP at each element rather than a fixed LSB
            sc = torch.tensor(x_scale * w_scale, dtype=torch.float32)
            out = (torch.tensor(acc, dtype=torch.float32) * sc).to(torch.bfloat16).float().numpy()
            ref = torch.tensor(y_sim, dtype=torch.float32).to(torch.bfloat16).float().numpy()
            ulp = np.maximum(np.abs(ref), 1e-30) * 2.0 ** -8
            diff = np.abs(out - ref) / ulp
            rows.append(dict(layer=name, consumer_kind='bfloat16', out_width=16,
                             exact_pct=float((diff == 0).mean() * 100),
                             within_1lsb_pct=float((diff <= 1).mean() * 100),
                             max_diff_lsb=float(diff.max()), n=int(diff.size)))
            continue
        M, k = make_requant(x_scale * w_scale / c['lsb'])
        out_int = requantize(acc, M, k, c['qmax'])

        ref_int = np.clip(np.round(y_sim / c['lsb']), -c['qmax'] - 1, c['qmax'])
        diff = np.abs(out_int - ref_int)
        rows.append(dict(layer=name, consumer_kind=c['kind'], out_width=c['width'],
                         exact_pct=float((diff == 0).mean() * 100),
                         within_1lsb_pct=float((diff <= 1).mean() * 100),
                         max_diff_lsb=int(diff.max()), n=int(diff.size)))

    rows += _verify_attention(payload, rt, cons, taps)
    return rows


def _verify_attention(payload, rt, cons, taps):
    """The six GEMMs inside the attention blocks. `attention_forward_hw` calls
    `int8_linear` directly, so no module hook sees them -- they come from
    `hw_quant.ATTN_TAP` instead."""
    rows = []
    for key, entry in payload['layers'].items():
        base = key.rsplit('.', 1)[0]
        blk = base[: -len('.attention')] if base.endswith('.attention') else None
        if entry['kind'] == 'in_proj' and blk in taps:
            t = taps[blk]
            E = entry['embed_dim']
            w_int = entry['q'].numpy().astype(np.int64)
            w_scale = _unpack_scale(entry['scale']).numpy().astype(np.float64)
            b_int = entry['bias']['q'].numpy().astype(np.int64)
            bands = cons[key]['bands']
            diffs = []
            for i, (nm, out_step) in enumerate(bands):
                lo, hi = i * E, (i + 1) * E
                src = {'Q': 'q_in', 'K': 'k_in', 'V': 'v_in'}[nm]
                in_step = rt.attn_scale({'Q': 'q_in_scales', 'K': 'k_in_scales',
                                         'V': 'v_in_scales'}[nm], blk)
                x = t[src].cpu().double().numpy()
                y_sim = t[{'Q': 'q', 'K': 'k', 'V': 'v'}[nm]].cpu().double().numpy()
                x_int = np.clip(np.round(x / in_step), -128, 127).astype(np.int64)
                acc = x_int @ w_int[lo:hi].T + b_int[lo:hi]
                # the whole matrix shares one shift; recompute it the same way
                # `export()` does so the verification uses the exported constant
                x_rows = np.concatenate([
                    np.full(E, rt.attn_scale('q_in_scales', blk)),
                    np.full(E, rt.attn_scale('k_in_scales', blk)),
                    np.full(E, rt.attn_scale('v_in_scales', blk))])
                lsb_rows = np.concatenate([np.full(E, s2) for _, s2 in bands])
                M_all, k_all = make_requant(x_rows * w_scale / lsb_rows)
                out_int = requantize(acc, M_all[lo:hi], k_all, 127)
                ref_int = np.clip(np.round(y_sim / out_step), -128, 127)
                diffs.append(np.abs(out_int - ref_int))
            d = np.concatenate([x.reshape(-1) for x in diffs])
            rows.append(dict(layer=f'{base} (Q|K|V)', consumer_kind='int8_banded',
                             out_width=8, exact_pct=float((d == 0).mean() * 100),
                             within_1lsb_pct=float((d <= 1).mean() * 100),
                             max_diff_lsb=int(d.max()), n=int(d.size)))
        elif base.endswith('.attention.out_proj'):
            blk = base[: -len('.attention.out_proj')]
            if blk not in taps or 'out_proj_in' not in taps[blk]:
                continue
            t = taps[blk]
            w_int = entry['q'].numpy().astype(np.int64)
            w_scale = _unpack_scale(entry['scale']).numpy().astype(np.float64)
            b_int = entry['bias']['q'].numpy().astype(np.int64)
            x_scale = float(_unpack_scale(entry['x_scale'])[0])
            x = t['out_proj_in'].cpu().double().numpy()
            y_sim = t['out_proj_out'].cpu().double().numpy()
            x_int = np.clip(np.round(x / x_scale), -128, 127).astype(np.int64)
            acc = x_int @ w_int.T + b_int
            sc = torch.tensor(x_scale * w_scale, dtype=torch.float32)
            out = (torch.tensor(acc, dtype=torch.float32) * sc).to(torch.bfloat16).float().numpy()
            ref = torch.tensor(y_sim, dtype=torch.float32).to(torch.bfloat16).float().numpy()
            ulp = np.maximum(np.abs(ref), 1e-30) * 2.0 ** -8
            d = np.abs(out - ref) / ulp
            rows.append(dict(layer=base, consumer_kind='bfloat16', out_width=16,
                             exact_pct=float((d == 0).mean() * 100),
                             within_1lsb_pct=float((d <= 1).mean() * 100),
                             max_diff_lsb=float(d.max()), n=int(d.size)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DVS128_10')
    parser.add_argument('--device', default=None)
    parser.add_argument('--no_verify', action='store_true')
    parser.add_argument('--m_bits', type=int, default=M_BITS,
                        help='width of the requantization multiplier M[c]')
    args = parser.parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    globals()['M_BITS'] = args.m_bits

    ds_dir = os.path.join(QUANT_DIR, args.dataset)
    path = os.path.join(ds_dir, 'quantized', 'model_int8.pt')
    weights_path = data_utils.find_best_checkpoint(os.path.join(ds_dir, 'weights'),
                                                    metric='val_acc', mode='max')
    all_params_path = os.path.join(ds_dir, 'all_params.json')
    payload = torch.load(path, map_location='cpu')
    skeleton, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    model, rt = hw_quant.instantiate(payload, skeleton, device)

    suffix = '' if args.m_bits == 32 else f'__m{args.m_bits}'
    out_dir = os.path.join(QUANT_DIR, 'fpga_export', f'{args.dataset}{suffix}')
    manifest, cons, total = export(payload, model, rt, out_dir)
    print(f"\nWrote {len(os.listdir(out_dir))} files to {out_dir}")
    print(f"  raw blob bytes: {total:,} ({total / 1024:.1f} KiB)")
    print(f"  layers described: {len(manifest['layers'])}, "
          f"attention blocks: {len(manifest['attention'])}, "
          f"non-linear formats: {len(manifest['nonlinear_formats'])}")

    if not args.no_verify:
        dl = data_utils.build_datamodule(json.load(open(all_params_path)),
                                         workers=2).val_dataloader()
        rows = verify(payload, model, rt, cons, dl, device)
        import pandas as pd
        df = pd.DataFrame(rows)
        print("\n--- integer-only formula vs. the simulator, per layer ---")
        print(df.to_string(index=False))
        print(f"\n  layers reproduced exactly        : "
              f"{int((df.exact_pct == 100).sum())} / {len(df)}")
        print(f"  worst-case disagreement          : {int(df.max_diff_lsb.max())} LSB "
              f"of the consumer's format")
        print(f"  min 'within 1 LSB' across layers : {df.within_1lsb_pct.min():.4f}%")
        df.to_csv(os.path.join(out_dir, 'verification.csv'), index=False)


if __name__ == '__main__':
    main()
