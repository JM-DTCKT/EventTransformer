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
    python export_fpga_constants.py --dataset DVS128_10 --config all_ln24_guard2
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
        cons[f'{blk}.attention.out_proj.weight'] = q_target(f'{blk}.layer_norm_att.in')
        cons[f'{blk}.linear1.weight'] = q_target(f'{blk}.gelu1.in')
        cons[f'{blk}.linear2.weight'] = q_target(f'{blk}.gelu2.in')
        cons[f'{blk}.linear3.weight'] = q_target(f'{blk}.layer_norm_att.in')

    cons['backbone.proc_embs_block.linear1.weight'] = int8_target(
        'backbone.proc_embs_block.relu.in', sites['backbone.proc_embs_block.relu.in'].scale)
    cons['models_clf.0.linear_1.weight'] = int8_target(
        'models_clf.0.relu.in', sites['models_clf.0.relu.in'].scale)
    cons['models_clf.0.linear_2.weight'] = q_target('models_clf.0.log_softmax.in')
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
            rec['bias_dtype'] = 'int32' if b['mode'] == 'int32' else 'float32'
            blobs[rec['bias_file']] = (b['q'] if b['mode'] == 'int32' else b['values']).numpy()

        c = cons.get(key)
        if c is None:
            manifest['layers'].append(rec)
            continue

        if entry['kind'] == 'linear':
            x_scale = float(_unpack_scale(entry['x_scale'])[0])
            rec['input'] = dict(dtype='int8', step=x_scale,
                                note='the producing stage already emits these codes')
            if 'x_scale_b' in entry:      # pos-enc K-split GEMM
                x_scale_b = float(_unpack_scale(entry['x_scale_b'])[0])
                k_split = int(entry['split_at'])
                Ma, ka = make_requant(x_scale * w_scale / c['lsb'])
                Mb, kb = make_requant(x_scale_b * w_scale / c['lsb'])
                rec['k_split'] = dict(
                    split_at=k_split,
                    note='two INT32 accumulators: A over inputs [0,split), '
                         'B over [split,E_in); bias folded into A',
                    A=dict(input_step=x_scale, mult_file=f'{base}.M_A.int32.bin', shift=ka),
                    B=dict(input_step=x_scale_b, mult_file=f'{base}.M_B.int32.bin', shift=kb))
                blobs[f'{base}.M_A.int32.bin'] = Ma.astype(np.int32)
                blobs[f'{base}.M_B.int32.bin'] = Mb.astype(np.int32)
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
            rec['requant'] = dict(mult_file=f'{base}.M.int32.bin', shift=k,
                                  bands=[dict(name=n, rows=[i * E, (i + 1) * E],
                                              input_step=float(np.full(1, s)[0]))
                                         for i, (n, s) in enumerate(c['bands'])])
            blobs[f'{base}.M.int32.bin'] = M.astype(np.int32)
            for bk in ('bias_k', 'bias_v'):
                if entry.get(bk):
                    rec[f'{bk}_file'] = f'{base}.{bk}.int32.bin'
                    blobs[rec[f'{bk}_file']] = entry[bk]['q'].numpy()

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
    if pe and pe['mode'] == 'int8':
        blobs['pos_encoding.int8.bin'] = pe['q'].numpy()
        manifest['pos_encoding'] = dict(file='pos_encoding.int8.bin', dtype='int8',
                                        shape=list(pe['shape']),
                                        step=float(_unpack_scale(pe['scale'])[0]),
                                        note='indexed [y][x][d]; feeds the K-split GEMM B side')
    manifest['fx_params'] = []
    for k, e in payload['fx_params'].items():
        f = f'{k}.int16.bin'
        blobs[f] = (e['codes'] if e['mode'] == 'fx16' else e['values']).numpy()
        manifest['fx_params'].append(dict(name=k, file=f,
                                          dtype='int16' if e['mode'] == 'fx16' else 'float32',
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
                 if entry.get('bias', {}).get('mode') == 'int32' else None)
        x_scale = float(_unpack_scale(entry['x_scale'])[0])

        if 'x_scale_b' in entry:
            ks = int(entry['split_at'])
            xb_scale = float(_unpack_scale(entry['x_scale_b'])[0])
            xa = np.clip(np.round(x[..., :ks] / x_scale), -128, 127).astype(np.int64)
            xb = np.clip(np.round(x[..., ks:] / xb_scale), -128, 127).astype(np.int64)
            acc_a = xa @ w_int[:, :ks].T + (b_int if b_int is not None else 0)
            acc_b = xb @ w_int[:, ks:].T
            Ma, ka = make_requant(x_scale * w_scale / c['lsb'])
            Mb, kb = make_requant(xb_scale * w_scale / c['lsb'])
            out_int = requantize(acc_a, Ma, ka, c['qmax']) + requantize(acc_b, Mb, kb, c['qmax'])
            out_int = np.clip(out_int, -c['qmax'] - 1, c['qmax'])
        else:
            x_int = np.clip(np.round(x / x_scale), -128, 127).astype(np.int64)
            acc = x_int @ w_int.T + (b_int if b_int is not None else 0)
            M, k = make_requant(x_scale * w_scale / c['lsb'])
            out_int = requantize(acc, M, k, c['qmax'])

        ref_int = np.clip(np.round(y_sim / c['lsb']), -c['qmax'] - 1, c['qmax'])
        diff = np.abs(out_int - ref_int)
        rows.append(dict(layer=name, consumer_kind=c['kind'], out_width=c['width'],
                         exact_pct=float((diff == 0).mean() * 100),
                         within_1lsb_pct=float((diff <= 1).mean() * 100),
                         max_diff_lsb=int(diff.max()), n=int(diff.size)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DVS128_10')
    parser.add_argument('--config', default='all_ln24_guard2')
    parser.add_argument('--device', default=None)
    parser.add_argument('--no_verify', action='store_true')
    parser.add_argument('--m_bits', type=int, default=M_BITS,
                        help='width of the requantization multiplier M[c]')
    args = parser.parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    globals()['M_BITS'] = args.m_bits

    ds_dir = os.path.join(QUANT_DIR, args.dataset)
    path = os.path.join(ds_dir, 'hw_quantized_models', f'hw__{args.config}.pt')
    weights_path = data_utils.find_best_checkpoint(os.path.join(ds_dir, 'weights'),
                                                    metric='val_acc', mode='max')
    all_params_path = os.path.join(ds_dir, 'all_params.json')
    payload = torch.load(path, map_location='cpu')
    skeleton, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    model, rt = hw_quant.instantiate(payload, skeleton, device)

    suffix = '' if args.m_bits == 32 else f'__m{args.m_bits}'
    out_dir = os.path.join(QUANT_DIR, 'fpga_export', f'{args.dataset}__{args.config}{suffix}')
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
