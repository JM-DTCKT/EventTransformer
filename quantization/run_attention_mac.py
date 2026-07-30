"""
End-to-end accuracy sweep for real INT8xINT8->INT32 attention MACs
(`quant_lib/attention_mac.py`), replacing the deleted
`quant_lib/int8_score_attention.py` / `run_qk_int8_score.py` /
`run_full_attention_int8.py` with a single, from-scratch-verified script.

Three cumulative versions are run end-to-end on ASL_DVS / DVS128_10 /
DVS128_11 (weights always INT8 per-channel, same as `int8_w8a8_static`):

  1. int8_w8a8_static+outproj  -- reproduces the existing `int8_w8a8_static`
     case (pre-projection Q/K/V activations fake-quantized; `Q.K^T` and
     `softmax(.)@V` computed in fp32 on the unquantized post-projection
     Q/K/V) *plus* the one gap that case always had: `out_proj`'s input
     activation is now also quantized, with a real int8 GEMM.
  2. int8_w8a8_static+qk_mac   -- (1) + Q and K are *also* quantized right
     after their projection, and `Q.K^T` is computed as a real INT8xINT8
     ->INT32 GEMM instead of fp32.
  3. int8_w8a8_static+qkv_mac  -- (2) + V is also quantized after its
     projection and `softmax(scores)` is quantized (analytically bounded in
     [0, 1], no calibration needed), so `softmax(.)@V` is *also* a real int8
     GEMM. This is the "every attention matmul really is int8xint8" case.

Before trusting any of the numbers below, `quant_lib.attention_mac.
sanity_check_against_reference` is run once per dataset: it patches
`AttentionBlock` to call the from-scratch `attention_forward` with *zero*
quantization and checks its logits against PyTorch's own, untouched
`nn.MultiheadAttention` on real batches -- if the reimplementation had a bug,
this would fail loudly before any quantized case runs.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python run_attention_mac.py                       # all 3 datasets
    python run_attention_mac.py --datasets ASL_DVS
    python run_attention_mac.py --calib_batches 200    # quick smoke test
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import attention_mac as am, data_utils, quant_wrapper  # noqa: E402

DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']
REFERENCE_CASES = ['fp32_baseline', 'int8_weightonly_perchannel', 'int8_w8a8_static']

# name -> (stage, description)
VERSIONS = [
    dict(name='int8_w8a8_static+outproj', stage='baseline',
         desc='existing int8_w8a8_static + out_proj activation now quantized (real int8 GEMM)'),
    dict(name='int8_w8a8_static+qk_mac', stage='qk_mac',
         desc='+ Q,K quantized post-projection, Q.K^T is a real INT8xINT8->INT32 GEMM'),
    dict(name='int8_w8a8_static+qkv_mac', stage='qkv_mac',
         desc='+ V quantized post-projection, softmax(scores) quantized, softmax(.)@V is also a real GEMM'),
]
_CASE_ORDER = REFERENCE_CASES + [v['name'] for v in VERSIONS]
_CMAP = plt.get_cmap('tab10')
CASE_COLORS = {name: _CMAP(i % 10) for i, name in enumerate(_CASE_ORDER)}


@torch.no_grad()
def calibrate_all(model, calibrators, act_quantizer, dataloader, device, num_batches):
    """Single pass over `dataloader` that simultaneously calibrates every
    `attention_mac` calibrator (pre/post-projection Q/K/V, out_proj) across
    all `AttentionBlock`s, plus `act_quantizer`'s hooks on every standalone
    `nn.Linear` (event_projection, proc_event_blocks, linear1-3, ...)."""
    model.eval()
    am.start_calibration(calibrators)
    act_quantizer.start_calibration()
    seen = 0
    for polarity, pixels, _labels in dataloader:
        if polarity is None:
            continue
        polarity = polarity.to(device)
        pixels = pixels.to(device)
        model(polarity, pixels)
        seen += 1
        if seen >= num_batches:
            break
    am.finalize_calibration(calibrators)
    act_quantizer.stop_calibration()
    act_quantizer.enable_quantized_activations()
    return seen


def run_one_version(version, weights_path, all_params_path, val_dataloader, train_dataloader,
                     device, max_samples, calib_batches):
    model, _ = data_utils.load_model(weights_path, all_params_path, device=device)

    # Weights: INT8 per-channel everywhere (every nn.Linear, incl. out_proj,
    # + nn.MultiheadAttention.in_proj_weight) -- identical to int8_w8a8_static.
    weight_quantizer = quant_wrapper.ModelQuantizer(model, weight_bits=8, weight_per_channel=True, act_bits=None)
    weight_quantizer.quantize_weights()

    # Attention: patch every AttentionBlock to run attention_mac.attention_forward
    # instead of the fused nn.MultiheadAttention kernel.
    calibrators = am.patch_attention_blocks(model, stage=version['stage'], num_bits=8)

    # Every other standalone nn.Linear's activation (event_projection,
    # proc_event_blocks, linear1-3 inside each AttentionBlock,
    # LatentEmbsCompressor, CLFBlock): identical mechanism to int8_w8a8_static.
    # weight_bits=None here so this instance never re-quantizes weights (already done above).
    act_quantizer = quant_wrapper.ModelQuantizer(model, weight_bits=None, weight_per_channel=True, act_bits=8)

    n_calib_batches = calibrate_all(model, calibrators, act_quantizer, train_dataloader, device, calib_batches)

    t0 = time.time()
    acc, n_eval = data_utils.evaluate_accuracy(
        model, val_dataloader, device, dtype=torch.float32, max_samples=max_samples, desc=version['name'],
    )
    eval_time_s = time.time() - t0

    # Pre-projection Q/K/V scales (exist for every stage -- these are what
    # `int8_w8a8_static` itself always calibrated) and every standalone
    # nn.Linear's activation scale, saved so `export_attention_mac_real_
    # quantized.py` can reconstruct a fully-quantized model without
    # recalibrating from scratch.
    q_in_scales = {name: c.q_in.scale.item() for name, c in calibrators.items()}
    k_in_scales = {name: c.k_in.scale.item() for name, c in calibrators.items()}
    v_in_scales = {name: c.v_in.scale.item() for name, c in calibrators.items()}
    linear_act_scales = act_quantizer.act_scales_snapshot()

    out_proj_scales = {name: c.out_proj.scale.item() for name, c in calibrators.items()}
    qk_scales = None
    if version['stage'] in ('qk_mac', 'qkv_mac'):
        qk_scales = {name: dict(q_scale=c.q_proj.scale.item(), k_scale=c.k_proj.scale.item())
                     for name, c in calibrators.items()}
    v_scale = None
    if version['stage'] == 'qkv_mac':
        v_scale = {name: c.v_proj.scale.item() for name, c in calibrators.items()}

    result = dict(
        case=version['name'], stage=version['stage'], accuracy=acc, n_eval_samples=n_eval,
        eval_time_s=eval_time_s, n_calib_batches=n_calib_batches,
    )
    scales = dict(q_in_scales=q_in_scales, k_in_scales=k_in_scales, v_in_scales=v_in_scales,
                   linear_act_scales=linear_act_scales,
                   out_proj_scales=out_proj_scales, qk_scales=qk_scales, v_scale=v_scale)
    return result, scales


def run_dataset(dataset_name, args, device):
    print(f"\n{'=' * 70}\n Dataset: {dataset_name}\n{'=' * 70}")
    dataset_dir = os.path.join(QUANT_DIR, dataset_name)
    weights_dir = os.path.join(dataset_dir, 'weights')
    all_params_path = os.path.join(dataset_dir, 'all_params.json')
    out_dir = os.path.join(dataset_dir, 'attention_mac')
    os.makedirs(out_dir, exist_ok=True)

    weights_path = data_utils.find_best_checkpoint(weights_dir, metric='val_acc', mode='max')
    print(f" - checkpoint: {os.path.basename(weights_path)}")

    all_params = json.load(open(all_params_path))
    val_dm = data_utils.build_datamodule(all_params, workers=args.workers)
    val_dataloader = val_dm.val_dataloader()
    train_dm = data_utils.build_datamodule(all_params, workers=args.workers)
    train_dataloader = train_dm.train_dataloader()

    # --- Trust check: does attention_forward (zero quantization) reproduce
    # PyTorch's own nn.MultiheadAttention? Must pass before anything below is
    # meaningful. ---
    sanity_model, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    max_diff = am.sanity_check_against_reference(sanity_model, val_dataloader, device, num_batches=3, atol=1e-4)
    del sanity_model
    print(f" - sanity check: attention_forward (no quant) vs nn.MultiheadAttention max logit diff = {max_diff:.2e} (PASSED)")

    calib_batches = args.calib_batches if args.calib_batches is not None else len(train_dataloader)
    print(f" - calibration: {calib_batches} train batches (mask-excluded for K/V max)")

    rows = []
    all_scales = {}
    for version in VERSIONS:
        print(f"\n--- [{dataset_name}] version: {version['name']} ({version['desc']}) ---")
        result, scales = run_one_version(
            version, weights_path, all_params_path, val_dataloader, train_dataloader,
            device, args.max_samples, calib_batches,
        )
        rows.append(result)
        all_scales[version['name']] = scales
        print(f"    accuracy={result['accuracy']*100:.3f}%  eval_time={result['eval_time_s']:.1f}s "
              f"(n={result['n_eval_samples']}, calib_batches={result['n_calib_batches']})")

    df = pd.DataFrame(rows)
    df.insert(0, 'dataset', dataset_name)
    df.insert(2, 'sanity_check_max_logit_diff', max_diff)

    ref_csv = os.path.join(dataset_dir, 'quantization_results.csv')
    if os.path.exists(ref_csv):
        ref_df = pd.read_csv(ref_csv)
        ref_df = ref_df[ref_df['case'].isin(REFERENCE_CASES)][['case', 'accuracy']].copy()
        ref_df.insert(0, 'dataset', dataset_name)
        ref_df['stage'] = 'n/a (reference, from run_quantization.py; int8_w8a8_static here still has the out_proj gap)'
        df = pd.concat([ref_df, df], ignore_index=True, sort=False)

    df.to_csv(os.path.join(out_dir, 'attention_mac_results.csv'), index=False)
    with open(os.path.join(out_dir, 'attention_mac_scales.json'), 'w') as f:
        json.dump(all_scales, f, indent=2)

    plot_case_comparison(df, dataset_name, os.path.join(out_dir, 'attention_mac_comparison.png'))
    print(f"\nSaved results to {out_dir}")
    return df


def plot_case_comparison(df, dataset_name, out_path):
    plot_df = df.copy()
    plot_df['_order'] = plot_df['case'].map({c: i for i, c in enumerate(_CASE_ORDER)})
    plot_df = plot_df.dropna(subset=['_order']).sort_values('_order')
    colors = [CASE_COLORS.get(c, '#607D8B') for c in plot_df['case']]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    bars = ax.bar(plot_df['case'], plot_df['accuracy'] * 100, color=colors, edgecolor='black', linewidth=0.6)
    baseline_acc = df.loc[df['case'] == 'fp32_baseline', 'accuracy']
    if len(baseline_acc):
        ax.axhline(baseline_acc.iloc[0] * 100, color='red', linestyle='--', lw=1.2, label='fp32 baseline')
    for b, v in zip(bars, plot_df['accuracy']):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.1, f"{v*100:.3f}", ha='center', fontsize=8)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'{dataset_name}: real INT8 attention MACs (out_proj always quantized)\n'
                 f'(gray = reference from run_quantization.py; colored = new, cumulative versions)')
    ax.tick_params(axis='x', rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_cross_dataset_summary(datasets, out_path):
    dfs = []
    for ds in datasets:
        csv_path = os.path.join(QUANT_DIR, ds, 'attention_mac', 'attention_mac_results.csv')
        if not os.path.exists(csv_path):
            continue
        d = pd.read_csv(csv_path)
        d = d[d['case'].isin(_CASE_ORDER)].set_index('case').loc[
            [c for c in _CASE_ORDER if c in d['case'].values]].reset_index()
        d['dataset'] = ds
        dfs.append(d)
    if not dfs:
        return
    full = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, len(dfs), figsize=(7.6 * len(dfs), 6.0), dpi=150)
    if len(dfs) == 1:
        axes = [axes]
    for ax, ds in zip(axes, [d['dataset'].iloc[0] for d in dfs]):
        sub = full[full['dataset'] == ds]
        bar_colors = [CASE_COLORS.get(c, '#607D8B') for c in sub['case']]
        bars = ax.bar(range(len(sub)), sub['accuracy'] * 100, color=bar_colors, edgecolor='black', linewidth=0.6)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['case'], fontsize=7.5)
        ax.tick_params(axis='x', rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        ymin, ymax = sub['accuracy'].min() * 100, sub['accuracy'].max() * 100
        pad = max((ymax - ymin) * 0.6, 0.15)
        ax.set_ylim(ymin - pad, ymax + pad * 1.6)
        for b, v in zip(bars, sub['accuracy']):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + pad * 0.15, f'{v*100:.3f}', ha='center', fontsize=7.5)
        base_acc = sub.loc[sub['case'] == 'fp32_baseline', 'accuracy']
        if len(base_acc):
            ax.axhline(base_acc.iloc[0] * 100, color='red', ls='--', lw=1.1)
        ax.set_title(ds, fontsize=12, fontweight='bold')
        ax.set_ylabel('Test accuracy (%)')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Real INT8xINT8->INT32 attention MACs (Q.K\u1d40, softmax(.)\u00b7V, out_proj) vs. existing cases\n'
        '(gray = reference; colored = cumulative new versions, all with the out_proj fix)',
        fontsize=11, y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--calib_batches', type=int, default=None,
                         help='default: one full pass over the training set')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    all_dfs = [run_dataset(ds, args, device) for ds in args.datasets]
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(QUANT_DIR, 'attention_mac_summary.csv'), index=False)
    plot_cross_dataset_summary(args.datasets, os.path.join(QUANT_DIR, 'attention_mac_cross_dataset.png'))
    print(f"\nSaved combined summary to {os.path.join(QUANT_DIR, 'attention_mac_summary.csv')}")


if __name__ == '__main__':
    main()
