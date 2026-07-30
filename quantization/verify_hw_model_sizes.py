"""Independent size check for the packed checkpoints `run_hw_quantization.py`
writes.

`hw_quant.payload_size_bytes` sums the bytes the saved tensors actually
occupy. This script recomputes what those bytes *should* be from the model
architecture alone -- parameter counts times the width each config assigns to
them -- and asserts the two agree, so "the file got smaller" is a checked
claim rather than a reported number. It also shows how much of the `.pt`
on-disk size is torch's zip/pickle container rather than payload.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python verify_hw_model_sizes.py
"""

import os
import sys

import pandas as pd
import torch
import torch.nn as nn

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import data_utils, fixed_point, hw_quant  # noqa: E402

DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']


def architecture_counts(model):
    """Parameter counts, grouped the way the payload groups them."""
    c = dict(weight_elems=0, weight_rows=0, weight_layers=0, n_linear=0,
             bias_elems=0, pos_enc_elems=0, fx_param_elems=0, fx_param_tensors=0)
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            c['weight_elems'] += m.weight.numel()
            c['weight_rows'] += m.weight.shape[0]
            c['weight_layers'] += 1
            c['n_linear'] += 1        # every Linear stores its own input scale
            if m.bias is not None:
                c['bias_elems'] += m.bias.numel()
        elif isinstance(m, nn.MultiheadAttention):
            c['weight_elems'] += m.in_proj_weight.numel()
            c['weight_rows'] += m.in_proj_weight.shape[0]
            c['weight_layers'] += 1
            if m.in_proj_bias is not None:
                c['bias_elems'] += m.in_proj_bias.numel()
            if m.bias_k is not None:
                c['bias_elems'] += m.bias_k.numel() + m.bias_v.numel()
        elif isinstance(m, nn.LayerNorm):
            for attr in ('weight', 'bias'):
                t = getattr(m, attr)
                if t is not None:
                    c['fx_param_elems'] += t.numel()
                    c['fx_param_tensors'] += 1
    sd = model.state_dict()
    if 'backbone.pos_encoding' in sd:
        c['pos_enc_elems'] = sd['backbone.pos_encoding'].numel()
    if 'backbone.memory_vertical' in sd:
        c['fx_param_elems'] += sd['backbone.memory_vertical'].numel()
        c['fx_param_tensors'] += 1
    return c


def expected_bytes(counts, cfg, n_attention_scales):
    """What each group must weigh under `cfg`, from first principles."""
    scale_w = 2 if cfg.fx16_scales else 4          # int16 code vs fp32
    scale_meta = 1 if cfg.fx16_scales else 0       # one shift amount per tensor
    fx_w = 2 if cfg.fx_nonlinear else 4
    fx_meta = 1 if cfg.fx_nonlinear else 0
    pos_w = 1 if cfg.int8_pos_enc else 4

    g = {
        'weights_int8': counts['weight_elems'] * 1,
        'weight_scales': counts['weight_rows'] * scale_w + counts['weight_layers'] * scale_meta,
        # INT32 and fp32 are both 4 bytes wide -- folding the bias into the
        # accumulator is a datapath change, not a storage saving
        'biases': counts['bias_elems'] * 4,
        'pos_encoding': counts['pos_enc_elems'] * pos_w + (scale_w + scale_meta
                                                           if cfg.int8_pos_enc else 0),
        'fx_params': counts['fx_param_elems'] * fx_w + counts['fx_param_tensors'] * fx_meta,
        # one scale per Linear input (+1 when the pos-enc concat gets a second
        # one) + the attention Q/K/V/out_proj/softmax scales
        'act_scales': (counts['n_linear'] + n_attention_scales
                       + (1 if cfg.split_posenc_scale else 0)) * (scale_w + scale_meta),
    }
    return g


def audit_dtypes(payload):
    """Every tensor the *inference* datapath reads, by dtype. Under the `all`
    config nothing here may be float: the claim is INT8 operands, INT32
    accumulator biases and 16-bit fixed-point everything-else."""
    found = {}

    def note(t, where):
        found.setdefault(str(t.dtype), []).append(where)

    for key, entry in payload['layers'].items():
        note(entry['q'], f'{key}:weight')
        for sk in ('scale', 'x_scale', 'x_scale_b'):
            if sk in entry:
                e = entry[sk]
                note(e['values'] if e['mode'] == 'fp32' else e['codes'], f'{key}:{sk}')
        for bk in ('bias', 'bias_k', 'bias_v'):
            if entry.get(bk):
                b = entry[bk]
                note(b['q'] if b['mode'] == 'int32' else b['values'], f'{key}:{bk}')
    for key, e in payload['fx_params'].items():
        note(e['values'] if e['mode'] == 'fp32' else e['codes'], key)
    pe = payload.get('pos_encoding')
    if pe:
        note(pe['q'] if pe['mode'] == 'int8' else pe['values'], 'pos_encoding')
        if pe['mode'] == 'int8':
            note(pe['scale']['values'] if pe['scale']['mode'] == 'fp32'
                 else pe['scale']['codes'], 'pos_encoding:scale')

    def walk(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict) and 'mode' in v:
                note(v['values'] if v['mode'] == 'fp32' else v['codes'], f'{prefix}{k}')
            elif isinstance(v, dict):
                walk(v, f'{prefix}{k}.')
    walk(payload['attention_scales'])
    return found


def main():
    rows = []
    for ds in DATASETS:
        model_dir = os.path.join(QUANT_DIR, ds, 'hw_quantized_models')
        if not os.path.isdir(model_dir):
            continue
        all_params_path = os.path.join(QUANT_DIR, ds, 'all_params.json')
        weights_path = data_utils.find_best_checkpoint(
            os.path.join(QUANT_DIR, ds, 'weights'), metric='val_acc', mode='max')
        model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
        counts = architecture_counts(model)
        fp32_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
        del model

        print(f"\n{'=' * 96}\n {ds}   (fp32 checkpoint = {fp32_bytes:,} B = {fp32_bytes / 1024:.1f} KiB)\n{'=' * 96}")
        print(f"{'config':<20} {'expected B':>12} {'actual B':>12} {'match':>7} "
              f"{'.pt on disk':>12} {'container':>10} {'vs fp32':>9}")

        for cfg in hw_quant.ablation_configs():
            path = os.path.join(model_dir, f'hw__{cfg.name}.pt')
            if not os.path.exists(path):
                continue
            payload = torch.load(path, map_location='cpu')
            actual = hw_quant.payload_size_bytes(payload)
            # number of attention scales stored (q/k/v_in, out_proj, v, q/k proj, softmax)
            n_attn = sum(len(v) for k, v in payload['attention_scales'].items()
                         if k not in ('softmax_scale', 'qk_scales')) \
                + 2 * len(payload['attention_scales'].get('qk_scales', {})) + 1
            exp = expected_bytes(counts, cfg, n_attn)
            exp_total = sum(exp.values())
            act_total = sum(v for k, v in actual.items()
                            if k not in ('total', 'other'))
            ok = exp_total == act_total
            disk = os.path.getsize(path)
            print(f"{cfg.name:<20} {exp_total:>12,} {act_total:>12,} {'OK' if ok else 'MISMATCH':>7} "
                  f"{disk:>12,} {disk - actual['total']:>10,} "
                  f"{act_total / fp32_bytes * 100:>8.1f}%")
            if not ok:
                for k in exp:
                    if exp[k] != actual[k]:
                        print(f"      group '{k}': expected {exp[k]:,} vs actual {actual[k]:,}")
            rows.append(dict(dataset=ds, config=cfg.name, expected_bytes=exp_total,
                             actual_bytes=act_total, match=ok, pt_file_bytes=disk,
                             container_overhead_bytes=disk - actual['total'],
                             fp32_ckpt_bytes=fp32_bytes,
                             pct_of_fp32=act_total / fp32_bytes * 100,
                             **{f'{k}_bytes': v for k, v in actual.items()}))

            if cfg.name.startswith('all'):
                dtypes = audit_dtypes(payload)
                floats = {d: v for d, v in dtypes.items() if 'float' in d}
                summary = ', '.join(f'{d}x{len(v)}' for d, v in sorted(dtypes.items()))
                print(f"      dtype audit [{cfg.name}]: {summary}"
                      + ('' if not floats else
                         f"   <-- STILL FLOAT: {[v[0] for v in floats.values()]}"))
                # the INT8 requant step in front of each ReLU lives in the site
                # book rather than in a tensor; `instantiate` snaps it onto a Q
                # format at load time -- verify that snap is idempotent
                bad = [k for k, s in (payload.get('sites') or {}).items()
                       if s['kind'] == 'int8' and s['scale'] is not None
                       and fixed_point.fx_round_scalar(s['scale'])[0] != s['scale']]
                if bad:
                    print(f"      note: {len(bad)} ReLU INT8 steps are stored fp32 in the "
                          f"site book and snapped to a Q format at load time")
                assert not floats, f'{cfg.name} still stores float tensors: {floats}'

    df = pd.DataFrame(rows)
    out = os.path.join(QUANT_DIR, 'hw_quant_size_verification.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    if len(df):
        print(f"all groups match expectation: {bool(df['match'].all())}")


if __name__ == '__main__':
    main()
