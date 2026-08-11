"""Figure: what the integer datapath costs, against the fp32 checkpoint.

Reads `results.csv` (written by `quantize.py`) and writes `accuracy_drop.png`.

Left panel  -- accuracy change in percentage points. Zero is a real origin
               here (it is a difference), so bars are honest; each dataset's
               +-1-test-sample resolution is drawn behind them, because two of
               the three test sets are far too small to resolve a tenth of a
               point.
Right panel -- what the packing bought: fp32 state dict vs the integer
               constants that actually go on the board.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']

# --- palette (validated: dataviz scripts/validate_palette.js, light mode) ----
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BAND = '#e1e0d9'
NEG = '#e34948'        # accuracy lost
POS = '#2a78d6'        # accuracy gained
NEUTRAL = '#c3c2b7'    # no change
FP32 = '#2a78d6'       # categorical slot 1
INT8 = '#eb6834'       # categorical slot 2


def main():
    df = pd.read_csv(os.path.join(QUANT_DIR, 'results.csv'))
    df['__o'] = df.dataset.map({d: i for i, d in enumerate(DATASETS)})
    df = df.sort_values('__o')
    ys = list(range(len(df)))[::-1]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 3.9), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.65, 1]))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.70, bottom=0.165, wspace=0.42)

    # ---------------- accuracy change ----------------
    ax.set_facecolor(SURFACE)
    for y, (_, r) in zip(ys, df.iterrows()):
        res = 100.0 / r.n_eval_samples          # one test sample, in pp
        ax.barh(y, 2 * res, left=-res, height=0.62, color=BAND, zorder=0, lw=0)
        d = r.accuracy_drop_pp
        col = NEG if d < -1e-9 else POS if d > 1e-9 else NEUTRAL
        ax.barh(y, d, height=0.34, color=col, zorder=2)
        ax.annotate(f'{d:+.3f} pp' if abs(d) > 1e-9 else '0.000 pp',
                    (d, y), xytext=(-7 if d < 0 else 7, 0), textcoords='offset points',
                    ha='right' if d < 0 else 'left', va='center',
                    fontsize=9, color=INK if abs(d) > 1e-9 else INK_MUTED)
        ax.annotate(f'{r.fp32_accuracy * 100:.4f}%  →  {r.int8_accuracy * 100:.4f}%'
                    f'      n = {int(r.n_eval_samples):,}',
                    (0, y), xytext=(0, 17), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, color=INK_MUTED)

    ax.axvline(0, color=AXIS, lw=1.2, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(df.dataset, fontsize=10, color=INK)
    span = max((df.accuracy_drop_pp.abs().max()), (100.0 / df.n_eval_samples).max())
    ax.set_xlim(-span * 1.55, span * 1.55)
    ax.set_ylim(-0.62, len(df) - 0.30)
    ax.set_xlabel('accuracy change vs. fp32 checkpoint  (pp)', fontsize=9, color=INK_2)
    ax.set_title('Accuracy cost of the integer datapath', fontsize=11.5, color=INK, pad=30)
    ax.annotate('grey band = $\\pm$1 test sample', (0.015, 0.035), xycoords='axes fraction',
                ha='left', va='bottom', fontsize=8, color=INK_MUTED)
    ax.grid(axis='x', color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0, colors=INK_2, labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)

    # ---------------- model size ----------------
    ax2.set_facecolor(SURFACE)
    h = 0.3
    for y, (_, r) in zip(ys, df.iterrows()):
        ax2.barh(y + h / 1.7, r.fp32_bytes / 1024, height=h, color=FP32, zorder=2)
        ax2.barh(y - h / 1.7, r.int8_bytes / 1024, height=h, color=INT8, zorder=2)
        ax2.annotate(f'{r.compression:.2f}$\\times$', (r.fp32_bytes / 1024, y),
                     xytext=(7, 0), textcoords='offset points',
                     ha='left', va='center', fontsize=9, color=INK)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, df.fp32_bytes.max() / 1024 * 1.30)
    ax2.set_ylim(-0.62, len(df) - 0.30)
    ax2.set_xlabel('KiB', fontsize=9, color=INK_2)
    ax2.set_title('Parameter bytes', fontsize=11.5, color=INK, pad=30)
    ax2.grid(axis='x', color=GRID, lw=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='both', length=0, colors=INK_2, labelsize=8.5)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=FP32),
                        plt.Rectangle((0, 0), 1, 1, color=INT8)],
               labels=['fp32 state dict', 'integer constants'],
               loc='lower left', bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handlelength=1.1, handleheight=1.1,
               columnspacing=1.4, borderpad=0)

    fig.suptitle('EvT on ZCU102 -- INT8 weights/activations, INT32 accumulators, Qm.n scales, '
                 'GELU Q4.11, softmax in Q6.9, LayerNorm bf16 $\\rightarrow$ Q4.11',
                 fontsize=11, color=INK_2, y=0.955)
    out = os.path.join(QUANT_DIR, 'accuracy_drop.png')
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print('wrote', out)
    print(df[['dataset', 'fp32_accuracy', 'int8_accuracy', 'accuracy_drop_pp',
              'n_eval_samples', 'compression']].to_string(index=False))


if __name__ == '__main__':
    main()
