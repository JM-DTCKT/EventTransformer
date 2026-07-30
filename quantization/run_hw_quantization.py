"""ZCU102-oriented quantization sweep: INT8 operands, INT32 accumulators/biases,
16-bit fixed-point scales and non-linear I/O -- and nothing else.

Starting point is the best real-quantized case from
`export_attention_mac_real_quantized.py` (`int8_w8a8_static+qkv_mac`: INT8
per-channel weights, INT8 activations, every attention matmul a real
INT8xINT8->INT32 GEMM). On top of that this script measures four changes,
each on its own and all together:

  1. `int32_bias`    -- biases folded into the INT32 accumulator
                        (`b_int = round(b / (s_x * s_w))`) instead of fp32
  2. `int8_pos_enc`  -- the learned positional-encoding table stored as INT8
  3. `fx16_scales`   -- every scale a 16-bit Qm.n fixed-point code
  4. `fx_nonlinear`  -- GELU / LayerNorm / softmax I/O snapped onto 16-bit
                        fixed-point grids (ReLU inputs: INT8, since only the
                        sign matters); LayerNorm affine params + the latent
                        memory table become Q-format int16 too
  5. `all`           -- all four at once

For every (dataset, config) it writes a genuinely packed checkpoint, reloads
the model *from that file*, evaluates test accuracy, and reports the byte
breakdown. It also dumps the value distributions the Q-format choices were
made from (`hw_quant_distributions.csv`).

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python run_hw_quantization.py                        # all 3 datasets
    python run_hw_quantization.py --datasets DVS128_10
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import data_utils, fixed_point as fx, hw_quant, quant_ops  # noqa: E402

DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']
SOURCE_CASE = 'int8_w8a8_static+qkv_mac'


# =============================================================================
# Distribution analysis -- what the Q formats are actually chosen from
# =============================================================================
def analyze_distributions(model, scales, site_book, dataset_name):
    """Every group of numbers that has to become fixed point, with the Qm.n
    format `fixed_point.choose_frac_bits` picks for it and the relative error
    that costs."""
    rows = []

    def add(group, name, tensor):
        r = fx.analyze_tensor(tensor, name)
        r['group'] = group
        r['dataset'] = dataset_name
        rows.append(r)

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            s = quant_ops.compute_qparams(m.weight.data, 8, per_channel=True, channel_dim=0)
            add('weight_scale (per-channel, per-layer exponent)', name, s.reshape(-1))
        elif isinstance(m, nn.MultiheadAttention):
            s = quant_ops.compute_qparams(m.in_proj_weight.data, 8, per_channel=True, channel_dim=0)
            add('weight_scale (per-channel, per-layer exponent)', f'{name}.in_proj', s.reshape(-1))

    act = torch.tensor([v / 127.0 for v in scales['linear_act_scales'].values()])
    add('activation_scale (per-tensor)', 'all nn.Linear inputs', act)
    attn_vals = []
    for key in ('q_in_scales', 'k_in_scales', 'v_in_scales', 'out_proj_scales', 'v_scale'):
        attn_vals += [float(v) for v in scales[key].values()]
    for v in scales['qk_scales'].values():
        attn_vals += [float(v['q_scale']), float(v['k_scale'])]
    add('activation_scale (per-tensor)', 'attention Q/K/V/out_proj', torch.tensor(attn_vals))

    sd = model.state_dict()
    if 'backbone.pos_encoding' in sd:
        # stored as INT8 (not Qm.n) -- report the INT8 round trip instead
        p = sd['backbone.pos_encoding']
        pscale = float(p.abs().max()) / 127.0
        rel_max, rel_mean = fx.relative_error(p, fx.pack_int8(p, pscale).float() * pscale)
        nz = p.abs()[p.abs() > 0]
        rows.append(dict(
            dataset=dataset_name, group='positional encoding (-> INT8)',
            name='backbone.pos_encoding', numel=int(p.numel()), max_abs=float(p.abs().max()),
            min_abs_nonzero=float(nz.min()), dynamic_range=float(p.abs().max() / nz.min()),
            frac_bits=None, q_format='INT8', step=pscale,
            rel_err_max=rel_max, rel_err_mean=rel_mean))
    if 'backbone.memory_vertical' in sd:
        add('latent memory table (-> Q16)', 'backbone.memory_vertical', sd['backbone.memory_vertical'])
    ln = [v.reshape(-1) for k, v in sd.items() if 'layer_norm' in k]
    if ln:
        add('LayerNorm affine (-> Q16)', 'all LayerNorm weight/bias', torch.cat(ln))

    for name, site in sorted(site_book.items()):
        kind = 'fx16 requant (non-linear I/O)' if site.kind == 'fx16' else 'int8 requant (ReLU input)'
        rows.append(dict(
            dataset=dataset_name, group=kind, name=name, numel=0,
            max_abs=site.max_abs, min_abs_nonzero=site.call_max_min,
            dynamic_range=site.call_spread,
            frac_bits=site.frac_bits if site.kind == 'fx16' else None,
            q_format=fx.q_format_name(site.frac_bits) if site.kind == 'fx16' else 'INT8',
            step=(2.0 ** -site.frac_bits) if site.kind == 'fx16' else site.scale,
            rel_err_max=float('nan'), rel_err_mean=float('nan')))

    # Would ONE global Q format do, instead of a per-tensor shared exponent?
    every_scale = torch.cat([
        quant_ops.compute_qparams(m.weight.data if isinstance(m, nn.Linear) else m.in_proj_weight.data,
                                  8, per_channel=True, channel_dim=0).reshape(-1)
        for _, m in model.named_modules() if isinstance(m, (nn.Linear, nn.MultiheadAttention))]
        + [act, torch.tensor(attn_vals)])
    add('ALL scales under ONE global Q format', 'every scale in the model', every_scale)
    return pd.DataFrame(rows)


# =============================================================================
# Per-dataset run
# =============================================================================
def run_dataset(dataset_name, args, device):
    print(f"\n{'=' * 74}\n Dataset: {dataset_name}\n{'=' * 74}")
    ds_dir = os.path.join(QUANT_DIR, dataset_name)
    all_params_path = os.path.join(ds_dir, 'all_params.json')
    weights_path = data_utils.find_best_checkpoint(os.path.join(ds_dir, 'weights'), metric='val_acc', mode='max')
    out_dir = os.path.join(ds_dir, 'hw_quantized_models')
    os.makedirs(out_dir, exist_ok=True)
    print(f" - checkpoint: {os.path.basename(weights_path)}")

    scales_path = os.path.join(ds_dir, 'attention_mac', 'attention_mac_scales.json')
    assert os.path.exists(scales_path), f"missing {scales_path}: run run_attention_mac.py first"
    scales = json.load(open(scales_path))[SOURCE_CASE]

    all_params = json.load(open(all_params_path))
    val_dataloader = data_utils.build_datamodule(all_params, workers=args.workers).val_dataloader()
    train_dataloader = data_utils.build_datamodule(all_params, workers=args.workers).train_dataloader()

    fp32_bytes = _fp32_checkpoint_bytes(weights_path, all_params_path)

    # ---- 1) calibrate the non-linear requantization sites (once, on the
    #        base integer datapath; the ranges do not depend on the config) ----
    base_cfg = hw_quant.HWConfig('base')
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    calib_payload = hw_quant.build_payload(model, scales, base_cfg, site_book=None)
    del model
    model, rt = hw_quant.instantiate(calib_payload, *_fresh_model(weights_path, all_params_path, device))
    n_batches = args.site_calib_batches or len(train_dataloader)
    n_batches = min(n_batches, len(train_dataloader))
    t0 = time.time()
    seen = hw_quant.calibrate_sites(model, rt, train_dataloader, device, n_batches)
    print(f" - calibrated {len(rt.sites)} requantization sites on {seen} train batches "
          f"({time.time() - t0:.0f}s)")
    site_book = rt.sites
    del model

    # ---- 2) distribution report ----
    ref_model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    dist_df = analyze_distributions(ref_model, scales, site_book, dataset_name)
    dist_df.to_csv(os.path.join(out_dir, 'hw_quant_distributions.csv'), index=False)
    del ref_model

    # ---- 3) every ablation config: pack -> save -> reload -> evaluate ----
    rows = []
    for cfg in hw_quant.ablation_configs():
        if args.configs and cfg.name not in args.configs:
            continue
        print(f"\n--- [{dataset_name}] config: {cfg} ---")
        src_model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
        payload = hw_quant.build_payload(src_model, scales, cfg, site_book=site_book)
        del src_model

        out_path = os.path.join(out_dir, f'hw__{cfg.name}.pt')
        torch.save(payload, out_path)

        loaded = torch.load(out_path, map_location='cpu')
        model, rt = hw_quant.instantiate(loaded, *_fresh_model(weights_path, all_params_path, device))

        t0 = time.time()
        acc, n_eval = data_utils.evaluate_accuracy(
            model, val_dataloader, device, dtype=torch.float32,
            max_samples=args.max_samples, desc=f'{dataset_name}:{cfg.name}')
        eval_s = time.time() - t0
        del model

        sizes = hw_quant.payload_size_bytes(loaded)
        row = dict(dataset=dataset_name, config=cfg.name, **cfg.as_dict())
        row.pop('name')
        row.update(accuracy=acc, n_eval_samples=n_eval, eval_time_s=eval_s,
                   file_bytes=os.path.getsize(out_path), fp32_ckpt_bytes=fp32_bytes,
                   max_abs_bias_int=payload['diagnostics']['max_abs_bias_int'],
                   **{f'bytes_{k}': v for k, v in sizes.items()})
        rows.append(row)
        print(f"    accuracy = {acc * 100:.3f}%   (n={n_eval}, {eval_s:.0f}s)")
        print(f"    packed   = {sizes['total'] / 1024:.1f} KiB tensors "
              f"({os.path.getsize(out_path) / 1024:.1f} KiB on disk) | "
              + ' '.join(f"{k}={v / 1024:.1f}K" for k, v in sizes.items() if k != 'total' and v))

    df = pd.DataFrame(rows)
    # keep any config that was not re-run this time, so `--configs` can extend
    # an earlier sweep instead of discarding it
    results_csv = os.path.join(out_dir, 'hw_quant_results.csv')
    if os.path.exists(results_csv):
        old = pd.read_csv(results_csv)
        old = old[~old['config'].isin(df['config'])]
        df = pd.concat([old, df], ignore_index=True, sort=False)
    order = {c.name: i for i, c in enumerate(hw_quant.ablation_configs())}
    df = df.sort_values('config', key=lambda s: s.map(order)).reset_index(drop=True)
    df.to_csv(results_csv, index=False)
    print(f"\nSaved {dataset_name} results to {out_dir}")
    return df, dist_df


def _fresh_model(weights_path, all_params_path, device):
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    return model, device


def _fp32_checkpoint_bytes(weights_path, all_params_path):
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    total = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    del model
    return total


# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--configs', nargs='+', default=None,
                        help='subset of: base int32_bias int8_pos_enc fx16_scales fx_nonlinear all')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--site_calib_batches', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    dfs, dists = [], []
    for ds in args.datasets:
        df, dist = run_dataset(ds, args, device)
        dfs.append(df)
        dists.append(dist)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(os.path.join(QUANT_DIR, 'hw_quant_summary.csv'), index=False)
    pd.concat(dists, ignore_index=True).to_csv(
        os.path.join(QUANT_DIR, 'hw_quant_distributions.csv'), index=False)

    print(f"\n{'=' * 74}\n Summary\n{'=' * 74}")
    print(df_all[['dataset', 'config', 'accuracy', 'bytes_total', 'file_bytes']].to_string(index=False))
    print(f"\nSaved to {os.path.join(QUANT_DIR, 'hw_quant_summary.csv')}")


if __name__ == '__main__':
    main()
