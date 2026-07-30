"""
Export REAL (bit-packed, actually-smaller-on-disk) quantized checkpoints for
every applicable case, and re-verify test accuracy on the reconstructed
(dequantized-on-load) model -- exactly the way a HuggingFace-style quantized
model repo (bitsandbytes / GPTQ / GGUF ...) is meant to be used: ship a
compact checkpoint, dequantize right before the matmul at inference time.

This is a *verification* step on top of `run_quantization.py`'s fake-quant
sweep, not a new PTQ run: the quantization math (per-channel/per-tensor
scale, symmetric round+clamp) is identical, so accuracy should reproduce the
`quantization_results.csv` numbers almost exactly. What's new here is that
the weight tensors are actually stored as `torch.int8` / bit-packed int4
buffers (+ small fp32 scales) instead of dequantized-back-to-fp32 tensors,
so the file size on disk genuinely shrinks.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python export_real_quantized.py                     # all datasets, all applicable cases
    python export_real_quantized.py --datasets DVS128_10
"""

import argparse
import copy
import json
import os
import sys

import pandas as pd
import torch

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import data_utils, quant_wrapper, real_quant  # noqa: E402
from run_quantization import DATASETS, QUANT_CASES  # noqa: E402

# Only cases with an actual weight bit-width need real packing; fp32_baseline
# has nothing to compress and fp16 already stores genuine fp16 (via
# `model.half()`), so both are skipped here.
APPLICABLE_CASES = [c for c in QUANT_CASES if c['weight_bits'] is not None]


def export_and_verify_case(case, dataset_dir, weights_path, all_params_path,
                            val_dataloader, device, max_samples, old_act_scales):
    real_dir = os.path.join(dataset_dir, 'real_quantized_models')
    os.makedirs(real_dir, exist_ok=True)
    out_path = os.path.join(real_dir, f"{case['name']}.pt")

    # 1) Export: fresh fp32 model -> real int8/int4-packed weights + fp32 scale.
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    extra = dict(case=case, act_scales=old_act_scales, act_bits=case['act_bits'])
    payload = real_quant.export_real_quantized(
        model, weight_bits=case['weight_bits'], per_channel=case['per_channel'],
        out_path=out_path, extra=extra,
    )
    del model

    # 2) Load back into a *fresh* model instance (dequantize real int -> fp32
    #    weights) and, for W*A* cases, re-attach activation quant hooks using
    #    the scales calibrated in the original fake-quant run.
    model2, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    real_quant.load_real_quantized(out_path, model2)
    model2.to(device)
    model2.eval()

    if case['act_bits'] is not None:
        quantizer = quant_wrapper.ModelQuantizer(
            model2, weight_bits=case['weight_bits'],
            weight_per_channel=case['per_channel'], act_bits=case['act_bits'],
        )
        quantizer._act_scale = dict(old_act_scales)
        quantizer.enable_quantized_activations()

    acc, n_eval = data_utils.evaluate_accuracy(
        model2, val_dataloader, device, dtype=torch.float32,
        max_samples=max_samples, desc=f"real:{case['name']}",
    )

    real_bytes = os.path.getsize(out_path)
    payload_bytes = real_quant.real_quantized_size_bytes(payload)
    return dict(case=case['name'], accuracy_real=acc, n_eval_samples=n_eval,
                real_pt_bytes=real_bytes, payload_raw_bytes=payload_bytes)


def run_dataset(dataset_name, args, device):
    print(f"\n{'=' * 70}\n Dataset: {dataset_name}\n{'=' * 70}")
    dataset_dir = os.path.join(QUANT_DIR, dataset_name)
    weights_dir = os.path.join(dataset_dir, 'weights')
    all_params_path = os.path.join(dataset_dir, 'all_params.json')
    quant_models_dir = os.path.join(dataset_dir, 'quantized_models')

    weights_path = data_utils.find_best_checkpoint(weights_dir, metric='val_acc', mode='max')
    all_params = json.load(open(all_params_path))
    val_dm = data_utils.build_datamodule(all_params, workers=args.workers)
    val_dataloader = val_dm.val_dataloader()

    results_csv = os.path.join(dataset_dir, 'quantization_results.csv')
    fake_df = pd.read_csv(results_csv).set_index('case')

    cases_to_run = [c for c in APPLICABLE_CASES if args.cases is None or c['name'] in args.cases]

    rows = []
    for case in cases_to_run:
        name = case['name']
        print(f"\n--- [{dataset_name}] real-export: {name} ---")

        fake_pt_path = os.path.join(quant_models_dir, f"{name}.pt")
        old_ckpt = torch.load(fake_pt_path, map_location='cpu')
        old_act_scales = old_ckpt.get('act_scales', {}) or {}
        fake_pt_bytes = os.path.getsize(fake_pt_path)
        del old_ckpt

        result = export_and_verify_case(
            case, dataset_dir, weights_path, all_params_path,
            val_dataloader, device, args.max_samples, old_act_scales,
        )
        result['fake_pt_mb'] = fake_pt_bytes / 1024 ** 2
        result['real_pt_mb'] = result['real_pt_bytes'] / 1024 ** 2
        result['payload_raw_mb'] = result['payload_raw_bytes'] / 1024 ** 2
        result['estimated_model_size_mb'] = fake_df.loc[name, 'model_size_mb']
        result['accuracy_fake_quant'] = fake_df.loc[name, 'accuracy']
        result['accuracy_diff_pp'] = (result['accuracy_fake_quant'] - result['accuracy_real']) * 100

        rows.append(result)
        print(f"    accuracy: real={result['accuracy_real']*100:.3f}%  "
              f"fake_quant_ref={result['accuracy_fake_quant']*100:.3f}%  "
              f"diff={result['accuracy_diff_pp']:+.4f}pp")
        print(f"    file size: fake(.pt)={result['fake_pt_mb']:.3f} MB  "
              f"real(.pt)={result['real_pt_mb']:.3f} MB  "
              f"(raw tensor bytes={result['payload_raw_mb']:.3f} MB, "
              f"estimated={result['estimated_model_size_mb']:.3f} MB)")

    df = pd.DataFrame(rows)
    df.insert(0, 'dataset', dataset_name)
    out_csv = os.path.join(dataset_dir, 'real_quantized_models', 'real_quant_verification.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved verification table to {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--cases', nargs='+', default=None,
                         choices=[c['name'] for c in APPLICABLE_CASES])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    all_dfs = [run_dataset(ds, args, device) for ds in args.datasets]
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(QUANT_DIR, 'real_quant_verification_summary.csv'), index=False)

    print(f"\n{'=' * 70}\n Summary (all datasets)\n{'=' * 70}")
    print(df_all[['dataset', 'case', 'accuracy_real', 'accuracy_fake_quant',
                   'accuracy_diff_pp', 'fake_pt_mb', 'real_pt_mb', 'estimated_model_size_mb']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
