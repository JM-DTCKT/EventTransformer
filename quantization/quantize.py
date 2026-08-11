"""fp32 checkpoint -> deployable integer checkpoint, and what it costs.

One entry point for the whole pipeline. Per dataset:

  1. sanity-check the reference attention forward against PyTorch's own kernel
  2. calibrate activation scales on train batches           (`quant_lib.calibration`)
  3. calibrate the non-linear requantization sites          (`hw_quant.calibrate_sites`)
  4. pack every parameter into integer constants            (`hw_quant.build_payload`)
  5. save the packed checkpoint, reload the model *from it*, and evaluate
  6. evaluate the untouched fp32 checkpoint for the baseline

The configuration is frozen (`hw_quant.FinalConfig`): INT8 weights/activations,
INT32 accumulators and biases, 16-bit Qm.n scales, GELU Q4.11, softmax input
Q6.9, bfloat16 LayerNorm, INT8 positional encoding with one shared input scale.

Usage:
    conda activate evt_new
    python quantize.py                      # all datasets
    python quantize.py --datasets DVS128_10
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QUANT_DIR, os.path.dirname(QUANT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from quant_lib import calibration, data_utils, hw_quant  # noqa: E402

DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']
RESULTS_CSV = os.path.join(QUANT_DIR, 'results.csv')


def _fp32_bytes(model):
    return sum(v.numel() * v.element_size() for v in model.state_dict().values())


def run_dataset(dataset_name, args, device):
    print(f"\n{'=' * 74}\n Dataset: {dataset_name}\n{'=' * 74}")
    # Calibration draws must not depend on which datasets ran before this one:
    # seed per dataset, and seed the train loader's workers too (see
    # `data_utils.seed_dataloader` -- the augmentation uses numpy).
    torch.manual_seed(args.seed)
    ds_dir = os.path.join(QUANT_DIR, dataset_name)
    all_params_path = os.path.join(ds_dir, 'all_params.json')
    weights_path = data_utils.find_best_checkpoint(
        os.path.join(ds_dir, 'weights'), metric='val_acc', mode='max')
    out_dir = os.path.join(ds_dir, 'quantized')
    os.makedirs(out_dir, exist_ok=True)
    print(f" - checkpoint: {os.path.basename(weights_path)}")

    all_params = json.load(open(all_params_path))
    val_loader = data_utils.build_datamodule(all_params, workers=args.workers).val_dataloader()
    train_loader = data_utils.seed_dataloader(
        data_utils.build_datamodule(all_params, workers=args.workers).train_dataloader(),
        args.seed)
    n_calib = min(args.calib_batches, len(train_loader))

    # ---- 0) fp32 baseline (the number everything else is compared against) ----
    model, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    fp32_bytes = _fp32_bytes(model)
    t0 = time.time()
    fp32_acc, n_eval = data_utils.evaluate_accuracy(
        model, val_loader, device, max_samples=args.max_samples, desc='fp32')
    print(f" - fp32 baseline: {fp32_acc * 100:.4f}%  (n={n_eval}, {time.time() - t0:.0f}s)")

    # ---- 1) is the reference attention forward really nn.MultiheadAttention? ----
    diff = calibration.sanity_check(model, val_loader, device, num_batches=3)
    print(f" - sanity check: reference attention vs nn.MultiheadAttention "
          f"max logit diff = {diff:.2e} (PASSED)")
    del model

    # ---- 2) activation scales ----
    model, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    t0 = time.time()
    scales, seen = calibration.calibrate_activation_scales(model, train_loader, device, n_calib)
    print(f" - activation scales: {len(scales['linear_act_scales'])} Linear inputs + "
          f"{len(scales['q_in_scales'])} attention blocks, {seen} train batches "
          f"({time.time() - t0:.0f}s)")
    del model

    # ---- 3) non-linear requantization sites, on the integer datapath ----
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    calib_payload = hw_quant.build_payload(model, scales, hw_quant.FinalConfig(), site_book=None)
    del model
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    model, rt = hw_quant.instantiate(calib_payload, model, device)
    t0 = time.time()
    seen = hw_quant.calibrate_sites(model, rt, train_loader, device, n_calib)
    print(f" - requant sites: {len(rt.sites)} calibrated on {seen} train batches "
          f"({time.time() - t0:.0f}s)")
    site_book = rt.sites
    del model, rt

    # ---- 4) pack + save ----
    cfg = hw_quant.FinalConfig()
    src, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    payload = hw_quant.build_payload(src, scales, cfg, site_book=site_book)
    del src
    out_path = os.path.join(out_dir, 'model_int8.pt')
    torch.save(payload, out_path)
    sizes = hw_quant.payload_size_bytes(payload)
    print(f" - packed: {out_path}  ({sizes['total']:,} B of integer constants, "
          f"file {os.path.getsize(out_path):,} B)")

    # ---- 5) reload from the packed file and evaluate ----
    payload = torch.load(out_path, map_location='cpu', weights_only=False)
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    model, _ = hw_quant.instantiate(payload, model, device)
    t0 = time.time()
    int_acc, n_eval = data_utils.evaluate_accuracy(
        model, val_loader, device, max_samples=args.max_samples, desc=cfg.name)
    eval_s = time.time() - t0
    del model
    print(f" - {cfg.name}: {int_acc * 100:.4f}%   "
          f"(drop {(int_acc - fp32_acc) * 100:+.4f} pp, n={n_eval}, {eval_s:.0f}s)")

    row = dict(dataset=dataset_name, config=cfg.name,
               fp32_accuracy=fp32_acc, int8_accuracy=int_acc,
               accuracy_drop_pp=(int_acc - fp32_acc) * 100.0,
               n_eval_samples=n_eval, n_calib_batches=n_calib,
               sanity_max_logit_diff=diff, eval_time_s=eval_s,
               fp32_bytes=fp32_bytes, int8_bytes=sizes['total'],
               file_bytes=os.path.getsize(out_path),
               compression=fp32_bytes / max(sizes['total'], 1))
    row.update({f'bytes_{k}': v for k, v in sizes.items() if k != 'total'})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    ap.add_argument('--calib_batches', type=int, default=100)
    ap.add_argument('--max_samples', type=int, default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0,
                    help='per-dataset seed; calibration draws depend on it')
    args = ap.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(hw_quant.FinalConfig())

    rows = [run_dataset(ds, args, device) for ds in args.datasets]
    df = pd.DataFrame(rows)

    if os.path.exists(RESULTS_CSV):     # keep datasets that were not re-run
        old = pd.read_csv(RESULTS_CSV)
        df = pd.concat([old[~old.dataset.isin(df.dataset)], df], ignore_index=True)
    df['__o'] = df.dataset.map({d: i for i, d in enumerate(DATASETS)})
    df = df.sort_values('__o').drop(columns='__o')
    df.to_csv(RESULTS_CSV, index=False)

    print(f"\n{'=' * 74}\n Summary\n{'=' * 74}")
    print(df[['dataset', 'fp32_accuracy', 'int8_accuracy', 'accuracy_drop_pp',
              'int8_bytes', 'compression']].to_string(index=False))
    print(f'\nSaved to {RESULTS_CSV}')


if __name__ == '__main__':
    main()
