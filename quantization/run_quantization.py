"""
Post-training quantization sweep for the three pretrained EvT models
(ASL_DVS, DVS128_10, DVS128_11), targeting FPGA deployment.

For every model this script:
  1. Loads the best checkpoint + its training config (`all_params.json`).
  2. Measures the fp32 baseline: test accuracy, model size, MACs.
  3. For each of QUANT_CASES: reloads a fresh model, fake-quantizes its
     weights (and, for W*A* cases, calibrates + fake-quantizes activations),
     evaluates test accuracy on the *real* test split, and estimates the
     resulting model size / BOPs.
  4. Saves the quantized weights + a metrics table + comparison plots under
     `quantization/<dataset>/`.
  5. After all datasets are done, writes a combined summary table/plot.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python run_quantization.py                                    # full run, all datasets
    python run_quantization.py --datasets DVS128_10                # single dataset
    python run_quantization.py --max_samples 100                   # quick smoke test
    python run_quantization.py --cases int4_w4a4_static             # add/refresh one case only,
                                                                     # merged into existing results
"""

import argparse
import copy
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

from quant_lib import calibration, complexity, data_utils, quant_wrapper  # noqa: E402


DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']

# name -> (weight_bits, weight_per_channel, act_bits, dtype)
#   dtype 'fp16' is handled as a plain torch.half() cast (no fake-quant hooks);
#   every other case runs in fp32 with fake-quantized weights/activations.
QUANT_CASES = [
    dict(name='fp32_baseline',                weight_bits=None, per_channel=True,  act_bits=None, dtype='fp32'),
    dict(name='fp16',                          weight_bits=None, per_channel=True,  act_bits=None, dtype='fp16'),
    dict(name='int8_weightonly_pertensor',     weight_bits=8,    per_channel=False, act_bits=None, dtype='fp32'),
    dict(name='int8_weightonly_perchannel',    weight_bits=8,    per_channel=True,  act_bits=None, dtype='fp32'),
    dict(name='int8_w8a8_static',              weight_bits=8,    per_channel=True,  act_bits=8,    dtype='fp32'),
    dict(name='int4_weightonly_perchannel',    weight_bits=4,    per_channel=True,  act_bits=None, dtype='fp32'),
    dict(name='int4w_int8a',                   weight_bits=4,    per_channel=True,  act_bits=8,    dtype='fp32'),
]


_CASE_NAMES = [c['name'] for c in QUANT_CASES]
_CMAP = plt.get_cmap('tab10')
# Fixed name -> color mapping, shared by every plot so the same quantization
# scheme always gets the same color across figures (accuracy/size/BOPs panels
# and across datasets).
CASE_COLORS = {name: _CMAP(i % 10) for i, name in enumerate(_CASE_NAMES)}


def effective_bits(bits):
    return bits if bits else 32


def run_one_case(case, weights_path, all_params_path, val_dataloader, train_dataloader,
                  device, max_samples, calib_batches):
    dtype_torch = torch.float16 if case['dtype'] == 'fp16' else torch.float32

    model, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    if case['dtype'] == 'fp16':
        model = model.half()

    quantizer = None
    n_calib_batches = 0
    if case['weight_bits'] is not None:
        quantizer = quant_wrapper.ModelQuantizer(
            model, weight_bits=case['weight_bits'],
            weight_per_channel=case['per_channel'], act_bits=case['act_bits'],
        )
        quantizer.quantize_weights()
        if case['act_bits'] is not None:
            n_calib_batches = calibration.calibrate_activations(
                model, quantizer, train_dataloader, device, num_batches=calib_batches
            ) or 0
            quantizer.enable_quantized_activations()

    t0 = time.time()
    acc, n_eval = data_utils.evaluate_accuracy(
        model, val_dataloader, device, dtype=dtype_torch, max_samples=max_samples, desc=case['name']
    )
    eval_time_s = time.time() - t0

    size_bytes = complexity.estimate_model_size_bytes(
        model, weight_bits=case['weight_bits'], per_channel=case['per_channel']
    )
    is_fp16 = case['dtype'] == 'fp16'
    if is_fp16:  # halves *everything* (not just quantizable weights); override.
        size_bytes = sum(p.numel() for p in model.parameters()) * 16 / 8

    result = dict(
        case=case['name'],
        weight_bits=16 if is_fp16 else effective_bits(case['weight_bits']),
        act_bits=16 if is_fp16 else effective_bits(case['act_bits']),
        weight_scheme='per_channel' if case['per_channel'] else 'per_tensor',
        accuracy=acc,
        n_eval_samples=n_eval,
        eval_time_s=eval_time_s,
        model_size_mb=size_bytes / (1024 ** 2),
        n_calib_batches=n_calib_batches,
    )

    quantized_state_dict = model.state_dict()
    act_scales = quantizer.act_scales_snapshot() if quantizer is not None else {}
    return result, quantized_state_dict, act_scales


def run_dataset(dataset_name, args, device):
    print(f"\n{'=' * 70}\n Dataset: {dataset_name}\n{'=' * 70}")
    dataset_dir = os.path.join(QUANT_DIR, dataset_name)
    weights_dir = os.path.join(dataset_dir, 'weights')
    all_params_path = os.path.join(dataset_dir, 'all_params.json')
    quant_models_dir = os.path.join(dataset_dir, 'quantized_models')
    os.makedirs(quant_models_dir, exist_ok=True)

    weights_path = data_utils.find_best_checkpoint(weights_dir, metric='val_acc', mode='max')
    print(f" - checkpoint: {os.path.basename(weights_path)}")

    all_params = json.load(open(all_params_path))
    val_dm = data_utils.build_datamodule(all_params, workers=args.workers)
    val_dataloader = val_dm.val_dataloader()
    train_dm = data_utils.build_datamodule(all_params, workers=args.workers)
    train_dataloader = train_dm.train_dataloader()

    # Calibrate activation quantizers over one full pass of the training set by
    # default (the CustomBatchSampler cycles through classes randomly, so this
    # is one full-epoch-equivalent of representative, class-balanced batches).
    calib_batches = args.calib_batches if args.calib_batches is not None else len(train_dataloader)
    print(f" - activation calibration: {calib_batches} train batches "
          f"({'user override' if args.calib_batches is not None else 'full training epoch'})")

    # MACs are independent of quantization case -> measure once on the fp32 model.
    base_model, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    macs = complexity.estimate_macs(base_model, val_dataloader, device, max_batches=args.macs_batches)
    del base_model
    print(f" - measured MACs/sample (avg over ~{args.macs_batches} batches): {macs / 1e6:.3f} M")

    cases_to_run = [c for c in QUANT_CASES if args.cases is None or c['name'] in args.cases]
    if args.cases is not None:
        print(f" - running only cases: {[c['name'] for c in cases_to_run]} (merging into existing results)")

    rows = []
    for case in cases_to_run:
        print(f"\n--- [{dataset_name}] case: {case['name']} ---")
        result, state_dict, act_scales = run_one_case(
            case, weights_path, all_params_path, val_dataloader, train_dataloader,
            device, args.max_samples, calib_batches,
        )
        bops = complexity.compute_bops(macs, result['weight_bits'], result['act_bits'])
        result['macs_M'] = macs / 1e6
        result['bops_G'] = bops / 1e9
        result['bops_reduction_x'] = complexity.bops_reduction_ratio(
            macs, result['weight_bits'], result['act_bits']
        )
        rows.append(result)

        torch.save(
            {'state_dict': state_dict, 'case': case, 'act_scales': act_scales, 'macs': macs},
            os.path.join(quant_models_dir, f"{case['name']}.pt"),
        )
        print(f"    accuracy={result['accuracy']*100:.2f}%  "
              f"size={result['model_size_mb']:.3f} MB  "
              f"BOPs_reduction={result['bops_reduction_x']:.2f}x  "
              f"eval_time={result['eval_time_s']:.1f}s  (n={result['n_eval_samples']})")

    new_df = pd.DataFrame(rows)
    new_df.insert(0, 'dataset', dataset_name)

    results_csv = os.path.join(dataset_dir, 'quantization_results.csv')
    if os.path.exists(results_csv):
        # Merge: keep every previously-computed case not re-run this time, drop
        # stale rows for any case we just (re)ran, then append the fresh ones.
        old_df = pd.read_csv(results_csv)
        old_df = old_df[~old_df['case'].isin(new_df['case'])]
        df = pd.concat([old_df.drop(columns=['accuracy_drop_pp', 'size_reduction_x'], errors='ignore'),
                         new_df], ignore_index=True)
    else:
        df = new_df

    case_order = [c['name'] for c in QUANT_CASES]
    df['_order'] = df['case'].map({name: i for i, name in enumerate(case_order)})
    df = df.sort_values('_order').drop(columns='_order').reset_index(drop=True)

    if not (df['case'] == 'fp32_baseline').any():
        raise RuntimeError(
            f"No fp32_baseline row found for {dataset_name} (need it to compute accuracy_drop_pp / "
            f"size_reduction_x) -- run with --cases fp32_baseline <your case> at least once."
        )
    baseline_acc = df.loc[df['case'] == 'fp32_baseline', 'accuracy'].iloc[0]
    baseline_size = df.loc[df['case'] == 'fp32_baseline', 'model_size_mb'].iloc[0]
    df['accuracy_drop_pp'] = (baseline_acc - df['accuracy']) * 100
    df['size_reduction_x'] = baseline_size / df['model_size_mb']

    df.to_csv(results_csv, index=False)
    df.to_json(os.path.join(dataset_dir, 'quantization_results.json'), orient='records', indent=2)
    plot_dataset_comparison(df, dataset_name, os.path.join(dataset_dir, 'quantization_comparison.png'))
    print(f"\nSaved results to {dataset_dir}")
    return df


def plot_dataset_comparison(df, dataset_name, out_path):
    """2x2 figure: accuracy / size bars (colored per-case) plus two trade-off
    scatter plots (accuracy vs. size, accuracy vs. BOPs reduction) that use
    the *same* per-case color + a legend instead of overlapping text labels,
    so points that land close together are still distinguishable."""
    # df rows are already ordered by QUANT_CASES (see run_dataset).
    cases = df['case'].tolist()
    colors = [CASE_COLORS[c] for c in cases]
    baseline_acc = df.loc[df['case'] == 'fp32_baseline', 'accuracy'].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    ax.bar(cases, df['accuracy'] * 100, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(baseline_acc * 100, color='red', linestyle='--', label='fp32 baseline')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'{dataset_name}: accuracy by case')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    ax = axes[0, 1]
    ax.bar(cases, df['bops_reduction_x'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('BOPs reduction vs. fp32 (x, log scale)')
    ax.set_yscale('log')
    ax.set_title('Compute (BOPs) reduction by case')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, which='both')

    ax = axes[1, 0]
    for case, color in zip(cases, colors):
        row = df.loc[df['case'] == case].iloc[0]
        ax.scatter(row['model_size_mb'], row['accuracy'] * 100, color=color, s=140,
                   label=case, edgecolor='black', linewidth=0.7, zorder=3)
    ax.set_xlabel('Model size (MB)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Accuracy vs. size trade-off')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)

    ax = axes[1, 1]
    for case, color in zip(cases, colors):
        row = df.loc[df['case'] == case].iloc[0]
        ax.scatter(row['bops_reduction_x'], row['accuracy'] * 100, color=color, s=140,
                   label=case, edgecolor='black', linewidth=0.7, zorder=3)
    ax.set_xlabel('BOPs reduction vs. fp32 (x, log scale)')
    ax.set_xscale('log')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Accuracy vs. compute (BOPs) trade-off')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)

    fig.suptitle(f'EvT quantization comparison -- {dataset_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_summary(df_all, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    case_order = [c['name'] for c in QUANT_CASES]
    bar_colors = [CASE_COLORS[c] for c in case_order]

    pivot_drop = df_all.pivot(index='case', columns='dataset', values='accuracy_drop_pp').loc[case_order]
    pivot_drop.plot(kind='bar', ax=axes[0])
    axes[0].set_ylabel('Accuracy drop vs fp32 (pp)')
    axes[0].set_title('Accuracy drop by case')
    axes[0].tick_params(axis='x', rotation=45)
    for tick, color in zip(axes[0].get_xticklabels(), bar_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    pivot_size = df_all.pivot(index='case', columns='dataset', values='size_reduction_x').loc[case_order]
    pivot_size.plot(kind='bar', ax=axes[1])
    axes[1].set_ylabel('Model size reduction (x)')
    axes[1].set_title('Size reduction by case')
    axes[1].tick_params(axis='x', rotation=45)
    for tick, color in zip(axes[1].get_xticklabels(), bar_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    pivot_bops = df_all.pivot(index='case', columns='dataset', values='bops_reduction_x').loc[case_order]
    pivot_bops.plot(kind='bar', ax=axes[2])
    axes[2].set_ylabel('BOPs (compute) reduction (x, log scale)')
    axes[2].set_yscale('log')
    axes[2].set_title('Compute reduction by case')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3, which='both')
    for tick, color in zip(axes[2].get_xticklabels(), bar_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    fig.suptitle('EvT quantization summary across datasets', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--cases', nargs='+', default=None,
                         choices=[c['name'] for c in QUANT_CASES],
                         help='only (re)run these cases and merge the results into the existing '
                              'quantization_results.csv for each dataset (default: run all cases)')
    parser.add_argument('--max_samples', type=int, default=None,
                         help='cap #test samples evaluated per case (default: full test set)')
    parser.add_argument('--calib_batches', type=int, default=None,
                         help='#train batches used for activation calibration '
                              '(default: one full pass over the training set, i.e. len(train_dataloader))')
    parser.add_argument('--macs_batches', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    all_dfs = []
    for dataset_name in args.datasets:
        df = run_dataset(dataset_name, args, device)
        all_dfs.append(df)

    if len(all_dfs) > 0:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv(os.path.join(QUANT_DIR, 'quantization_summary.csv'), index=False)
        if set(args.datasets) == set(DATASETS):
            plot_summary(df_all, os.path.join(QUANT_DIR, 'quantization_summary.png'))
        print(f"\nSaved combined summary to {os.path.join(QUANT_DIR, 'quantization_summary.csv')}")


if __name__ == '__main__':
    main()
