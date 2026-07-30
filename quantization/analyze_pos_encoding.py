"""How much does EvT actually use its positional encoding, and how much of it
survives the INT8 activation quantizer?

`hw_flow.md` claims the pos-enc table can be stored INT8 for free because the
tensor it is concatenated into is re-quantized with a *coarser* step anyway.
That raises the obvious question: if the encoding is being crushed, why does
accuracy not move? Two measurements answer it:

  1. **Sensitivity** -- evaluate the untouched fp32 model with
     `backbone.pos_encoding` zeroed out. If accuracy barely moves, the trained
     model does not lean on the encoding and no amount of precision there will
     show up in the score.
  2. **Damage** -- hook `preproc_block_events`'s Linear, split its input back
     into the two halves it was concatenated from (event-projection features |
     positional encoding), and measure what the single shared INT8 scale does
     to each half: effective step, number of distinct levels the pos-enc half
     lands on, and SQNR -- against what a *split* scale (one per half) would
     give.

Usage:
    conda activate evt_new
    cd EventTransformer/quantization
    python analyze_pos_encoding.py --datasets DVS128_10
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
import torch.nn as nn

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
if QUANT_DIR not in sys.path:
    sys.path.insert(0, QUANT_DIR)

from quant_lib import data_utils  # noqa: E402

DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']
SOURCE_CASE = 'int8_w8a8_static+qkv_mac'


def sqnr_db(x, xq):
    err = (xq - x).float()
    sig = x.float()
    p_sig = float((sig ** 2).mean())
    p_err = float((err ** 2).mean())
    return 10.0 * torch.log10(torch.tensor(max(p_sig, 1e-30) / max(p_err, 1e-30))).item()


def int8_round(x, scale):
    return torch.clamp(torch.round(x / scale), -128, 127) * scale


@torch.no_grad()
def measure_concat_split(model, dataloader, device, shared_max_abs, split_at, num_batches=8):
    """Statistics of `preproc_block_events`'s Linear input, split back into
    [event-projection features | positional encoding]."""
    lin = model.backbone.preproc_block_events.seq_init[0]
    grab = {}

    def hook(module, args):
        grab['x'] = args[0].detach()

    h = lin.register_forward_pre_hook(hook)
    rows = []
    seen = 0
    for polarity, pixels, _labels in dataloader:
        if polarity is None:
            continue
        model(polarity.to(device), pixels.to(device))
        x = grab['x']
        a, b = x[..., :split_at], x[..., split_at:]

        s_shared = shared_max_abs / 127.0
        s_a = float(a.abs().max()) / 127.0
        s_b = float(b.abs().max()) / 127.0

        b_shared = int8_round(b, s_shared)
        rows.append(dict(
            max_abs_projection=float(a.abs().max()), max_abs_posenc=float(b.abs().max()),
            rms_projection=float(a.float().pow(2).mean().sqrt()),
            rms_posenc=float(b.float().pow(2).mean().sqrt()),
            step_shared=s_shared, step_posenc_own=s_b,
            posenc_levels_used=int(torch.round(b / s_shared).unique().numel()),
            posenc_zeroed_frac=float((torch.round(b / s_shared) == 0).float().mean()),
            sqnr_posenc_shared_db=sqnr_db(b, b_shared),
            sqnr_posenc_split_db=sqnr_db(b, int8_round(b, s_b)),
            sqnr_projection_shared_db=sqnr_db(a, int8_round(a, s_shared)),
            # how much of the layer's output actually comes from the pos-enc half
            out_norm_projection=float(torch.linalg.vector_norm(
                torch.nn.functional.linear(a, lin.weight[:, :split_at]))),
            out_norm_posenc=float(torch.linalg.vector_norm(
                torch.nn.functional.linear(b, lin.weight[:, split_at:]))),
        ))
        seen += 1
        if seen >= num_batches:
            break
    h.remove()
    return pd.DataFrame(rows).mean().to_dict()


@torch.no_grad()
def position_separability(pos_encoding, step, device):
    """The encoding's job is to make positions *distinguishable*, not to carry
    a precise value. So the question is not "how much SQNR survives" but
    "after quantization, is each position's code still closer to its own
    original code than to any other position's".

    `pos_encoding`: (H, W, D). Returns the fraction of the H*W position codes
    that are still uniquely identifiable, plus the margin that decides it:
    the minimum distance between two distinct codes vs. the RMS distance the
    quantizer moves a code."""
    P = pos_encoding.reshape(-1, pos_encoding.shape[-1]).to(device).float()
    Pq = torch.clamp(torch.round(P / step), -128, 127) * step

    d2 = torch.cdist(P, P)
    n = P.shape[0]
    d2.fill_diagonal_(float('inf'))
    d_min = float(d2.min())

    nearest = torch.cdist(Pq, P).argmin(dim=1)
    identifiable = float((nearest == torch.arange(n, device=device)).float().mean())
    noise_rms = float((Pq - P).pow(2).sum(dim=1).sqrt().mean())
    return dict(n_positions=n, code_dim=P.shape[1], min_code_distance=d_min,
                quant_noise_rms=noise_rms,
                separation_margin=d_min / max(noise_rms, 1e-12),
                positions_identifiable=identifiable)


def run_dataset(ds, args, device):
    print(f"\n{'=' * 74}\n {ds}\n{'=' * 74}")
    ds_dir = os.path.join(QUANT_DIR, ds)
    all_params_path = os.path.join(ds_dir, 'all_params.json')
    weights_path = data_utils.find_best_checkpoint(os.path.join(ds_dir, 'weights'),
                                                   metric='val_acc', mode='max')
    all_params = json.load(open(all_params_path))
    val_dataloader = data_utils.build_datamodule(all_params, workers=args.workers).val_dataloader()

    scales = json.load(open(os.path.join(ds_dir, 'attention_mac',
                                          'attention_mac_scales.json')))[SOURCE_CASE]
    shared_max_abs = float(scales['linear_act_scales']['backbone.preproc_block_events.seq_init.0'])

    # ---- 1) sensitivity: fp32 with the positional encoding zeroed ----
    model, _ = data_utils.load_model(weights_path, all_params_path, device=device)
    split_at = model.backbone.preproc_block_events.seq_init[0].weight.shape[1] \
        - model.backbone.pos_encoding.shape[-1]
    stats = measure_concat_split(model, val_dataloader, device, shared_max_abs,
                                 split_at, num_batches=args.stat_batches)

    sep = position_separability(model.backbone.pos_encoding.data,
                                stats['step_shared'], device)
    sep_split = position_separability(model.backbone.pos_encoding.data,
                                      stats['step_posenc_own'], device)

    acc_fp32, n = data_utils.evaluate_accuracy(model, val_dataloader, device, desc=f'{ds}:fp32')
    with torch.no_grad():
        model.backbone.pos_encoding.zero_()
    acc_nopos, _ = data_utils.evaluate_accuracy(model, val_dataloader, device, desc=f'{ds}:no-pos')
    del model

    print(f"  fp32 baseline                : {acc_fp32 * 100:.3f}%  (n={n})")
    print(f"  fp32 with pos_encoding = 0   : {acc_nopos * 100:.3f}%  "
          f"({(acc_nopos - acc_fp32) * 100:+.3f} pp)")
    print(f"  concat input, projection half: max {stats['max_abs_projection']:.3f}, "
          f"rms {stats['rms_projection']:.4f}")
    print(f"  concat input, pos-enc half   : max {stats['max_abs_posenc']:.3f}, "
          f"rms {stats['rms_posenc']:.4f}")
    print(f"  shared INT8 step             : {stats['step_shared']:.5f}  "
          f"(pos-enc's own step would be {stats['step_posenc_own']:.5f})")
    print(f"  pos-enc distinct INT8 levels : {stats['posenc_levels_used']:.0f} of 256   "
          f"({stats['posenc_zeroed_frac'] * 100:.1f}% of its values round to 0)")
    print(f"  pos-enc SQNR shared / split  : {stats['sqnr_posenc_shared_db']:.1f} dB  /  "
          f"{stats['sqnr_posenc_split_db']:.1f} dB")
    print(f"  projection half SQNR (shared): {stats['sqnr_projection_shared_db']:.1f} dB")
    print(f"  |Wa @ projection| / |Wb @ pos-enc| : "
          f"{stats['out_norm_projection']:.1f} / {stats['out_norm_posenc']:.1f}  "
          f"(pos-enc contributes "
          f"{stats['out_norm_posenc'] / (stats['out_norm_projection'] + stats['out_norm_posenc']) * 100:.1f}%)")

    print(f"  {sep['n_positions']} position codes ({sep['code_dim']}-dim), closest pair "
          f"{sep['min_code_distance']:.3f} apart")
    print(f"    shared scale: quantizer moves a code by {sep['quant_noise_rms']:.3f} rms "
          f"-> margin {sep['separation_margin']:.2f}x, "
          f"{sep['positions_identifiable'] * 100:.2f}% of positions still uniquely identifiable")
    print(f"    split scale : quantizer moves a code by {sep_split['quant_noise_rms']:.3f} rms "
          f"-> margin {sep_split['separation_margin']:.2f}x, "
          f"{sep_split['positions_identifiable'] * 100:.2f}%")

    return dict(dataset=ds, accuracy_fp32=acc_fp32, accuracy_pos_zeroed=acc_nopos,
                delta_pp=(acc_nopos - acc_fp32) * 100, n_eval_samples=n, **stats,
                **{f'shared_{k}': v for k, v in sep.items()},
                **{f'split_{k}': v for k, v in sep_split.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--stat_batches', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    rows = [run_dataset(ds, args, device) for ds in args.datasets]
    df = pd.DataFrame(rows)
    out = os.path.join(QUANT_DIR, 'pos_encoding_analysis.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
