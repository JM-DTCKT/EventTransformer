"""
Export REAL (bit-packed, actually-smaller-on-disk) quantized checkpoints for
the 3 `attention_mac` versions (`run_attention_mac.py`: `+outproj`, `+qk_mac`,
`+qkv_mac`), and re-verify end-to-end test accuracy after loading back from
the packed checkpoint -- exactly the same "pack for real, dequantize right
before use" workflow `export_real_quantized.py` already does for the
`run_quantization.py` cases, applied here to the 3 new attention-MAC cases.

Weight quantization (INT8 per-channel, every `nn.Linear.weight` +
`nn.MultiheadAttention.in_proj_weight`) is *identical* across all 3
attention_mac versions -- only how the attention matmuls are *computed*
differs (fp32 vs. real INT8 GEMM), not how weights are *stored*. So the
packed weight buffers -- and therefore the on-disk file size -- come out
nearly identical for all 3 (they only differ by a handful of extra fp32
scalars: the calibrated out_proj/Q/K/V activation scales each version needs
to reproduce its exact quantization at inference time, loaded here from the
`attention_mac_scales.json` that `run_attention_mac.py` already saved -- no
recalibration needed). This script exists to make that concrete: actually
pack the weights into real `torch.int8` buffers (not just simulate low-bit
*values* while staying fp32-sized, as `run_attention_mac.py` does) and
report the genuine byte counts.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python export_attention_mac_real_quantized.py                    # all 3 datasets
    python export_attention_mac_real_quantized.py --datasets ASL_DVS
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import attention_mac as am, data_utils, quant_wrapper, real_quant  # noqa: E402
from run_attention_mac import DATASETS, VERSIONS  # noqa: E402


def export_and_verify_version(version, weights_path, all_params_path, val_dataloader,
                               device, max_samples, scales_for_version, out_path):
    # 1) Export: fresh fp32 model -> real int8-packed weights + fp32 scale
    #    (weight quantization is identical for every attention_mac version).
    model, _ = data_utils.load_model(weights_path, all_params_path, device='cpu')
    weight_quantizer = quant_wrapper.ModelQuantizer(model, weight_bits=8, weight_per_channel=True, act_bits=None)
    weight_quantizer.quantize_weights()
    payload = real_quant.export_real_quantized(
        model, weight_bits=8, per_channel=True, out_path=out_path,
        extra=dict(version=version['name'], stage=version['stage'], attention_scales=scales_for_version),
    )
    del model

    # 2) Load back into a *fresh* model instance (dequantize real int8 ->
    #    fp32 weights), patch attention for this version's stage, and inject
    #    the calibrated scales saved by run_attention_mac.py at calibration
    #    time -- no recalibration pass needed.
    model2, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    real_quant.load_real_quantized(out_path, model2)
    model2.to(device)
    model2.eval()

    calibrators = am.patch_attention_blocks(model2, stage=version['stage'], num_bits=8)
    am.inject_calibrated_scales(
        calibrators,
        q_in_scales=scales_for_version['q_in_scales'],
        k_in_scales=scales_for_version['k_in_scales'],
        v_in_scales=scales_for_version['v_in_scales'],
        out_proj_scales=scales_for_version['out_proj_scales'],
        qk_scales=scales_for_version.get('qk_scales'),
        v_scale=scales_for_version.get('v_scale'),
    )

    # Every other standalone nn.Linear's activation (event_projection,
    # proc_event_blocks, linear1-3, ...): reuse the scales calibrated by
    # run_attention_mac.py instead of recalibrating.
    act_quantizer = quant_wrapper.ModelQuantizer(model2, weight_bits=None, weight_per_channel=True, act_bits=8)
    act_quantizer._act_scale = dict(scales_for_version['linear_act_scales'])
    act_quantizer.enable_quantized_activations()

    acc, n_eval = data_utils.evaluate_accuracy(
        model2, val_dataloader, device, dtype=torch.float32,
        max_samples=max_samples, desc=f"real:{version['name']}",
    )

    real_bytes = os.path.getsize(out_path)
    payload_bytes = real_quant.real_quantized_size_bytes(payload)
    return dict(version=version['name'], accuracy_real=acc, n_eval_samples=n_eval,
                real_pt_bytes=real_bytes, payload_raw_bytes=payload_bytes)


def run_dataset(dataset_name, args, device):
    print(f"\n{'=' * 70}\n Dataset: {dataset_name}\n{'=' * 70}")
    dataset_dir = os.path.join(QUANT_DIR, dataset_name)
    weights_dir = os.path.join(dataset_dir, 'weights')
    all_params_path = os.path.join(dataset_dir, 'all_params.json')
    attn_dir = os.path.join(dataset_dir, 'attention_mac')
    real_dir = os.path.join(dataset_dir, 'real_quantized_models')
    os.makedirs(real_dir, exist_ok=True)

    weights_path = data_utils.find_best_checkpoint(weights_dir, metric='val_acc', mode='max')
    print(f" - checkpoint: {os.path.basename(weights_path)}")

    all_params = json.load(open(all_params_path))
    val_dm = data_utils.build_datamodule(all_params, workers=args.workers)
    val_dataloader = val_dm.val_dataloader()

    scales_path = os.path.join(attn_dir, 'attention_mac_scales.json')
    results_csv = os.path.join(attn_dir, 'attention_mac_results.csv')
    assert os.path.exists(scales_path) and os.path.exists(results_csv), (
        f"Run `python run_attention_mac.py --datasets {dataset_name}` first "
        f"(missing {scales_path} or {results_csv}).")
    all_scales = json.load(open(scales_path))
    fake_df = pd.read_csv(results_csv).set_index('case')

    rows = []
    for version in VERSIONS:
        name = version['name']
        print(f"\n--- [{dataset_name}] real-export: {name} ---")
        out_path = os.path.join(real_dir, f"attention_mac__{name.replace('+', '_')}.pt")
        result = export_and_verify_version(
            version, weights_path, all_params_path, val_dataloader, device,
            args.max_samples, all_scales[name], out_path,
        )
        result['real_pt_mb'] = result['real_pt_bytes'] / 1024 ** 2
        result['payload_raw_mb'] = result['payload_raw_bytes'] / 1024 ** 2
        result['accuracy_fake_quant'] = fake_df.loc[name, 'accuracy']
        result['accuracy_diff_pp'] = (result['accuracy_fake_quant'] - result['accuracy_real']) * 100
        rows.append(result)
        print(f"    accuracy: real={result['accuracy_real']*100:.3f}%  "
              f"fake_quant_ref={result['accuracy_fake_quant']*100:.3f}%  "
              f"diff={result['accuracy_diff_pp']:+.4f}pp")
        print(f"    file size: real(.pt)={result['real_pt_mb']:.4f} MB "
              f"({result['real_pt_bytes']:,} bytes; raw tensor bytes={result['payload_raw_bytes']:,})")

    df = pd.DataFrame(rows)
    df.insert(0, 'dataset', dataset_name)
    out_csv = os.path.join(real_dir, 'attention_mac_real_quant_verification.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved verification table to {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    all_dfs = [run_dataset(ds, args, device) for ds in args.datasets]
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(QUANT_DIR, 'attention_mac_real_quant_verification_summary.csv'), index=False)

    print(f"\n{'=' * 70}\n Summary (all datasets)\n{'=' * 70}")
    print(df_all[['dataset', 'version', 'accuracy_real', 'accuracy_fake_quant',
                   'accuracy_diff_pp', 'real_pt_mb', 'real_pt_bytes']].to_string(index=False))


if __name__ == '__main__':
    main()
