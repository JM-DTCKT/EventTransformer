"""Figures for the ZCU102 integer-only quantization sweep (`run_hw_quantization.py`).

  1. hw_quant_accuracy_drop.png  -- accuracy lost vs. the fp32 baseline, for
     each change on its own and all of them together, per dataset.
  2. hw_quant_model_size.png     -- where the bytes go, per config.
  3. hw_quant_requant_ranges.png -- the evidence behind the Q-format choices:
     how much each requantization site's magnitude swings between calls.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['ASL_DVS', 'DVS128_10', 'DVS128_11']

# --- palette (validated: node scripts/validate_palette.js, light mode) -------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
DIVERGE_NEG = '#d03b3b'      # accuracy lost
DIVERGE_POS = '#2a78d6'      # accuracy gained
NEUTRAL = '#c3c2b7'          # no change
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']

CONFIG_LABEL = {
    'base':                 'reference: INT8 W/A + INT8 attention MAC',
    'int32_bias':           '① INT32 bias',
    'int8_pos_enc':         '② INT8 positional encoding',
    'fx16_scales':          '③ FX16 scales (Qm.n)',
    'fx_nonlinear':         '④ FX16 non-linear I/O — LayerNorm in: static Q16',
    'fx_nonlinear_lndyn':   '④ᵃ LayerNorm in: Q16 + run-time exponent',
    'fx_nonlinear_lnstepq': '④ᵇ LayerNorm in: Q16, one format per time step',
    'fx_nonlinear_ln32':    '④ᶜ LayerNorm in: static Q32',
    'all':                  '①+②+③+④   (LayerNorm in: static Q16)',
    'all_lndyn':            '①+②+③+④ᵃ  (run-time exponent)',
    'all_lnstepq':          '①+②+③+④ᵇ  (per-time-step Q16)',
    'all_ln32':             '①+②+③+④ᶜ  (static Q32)',
    'all_ln32_splitpos':    '①+②+③+④ᶜ + split pos-enc scale',
    'all_ln32_ctr16':       '①+②+③+④ᶜ, variance path narrowed to 16 bit',
    'all_ln32_ctr32':       '①+②+③+④ᶜ, variance path 32 bit',
    'all_ln24':             '①+②+③+④ + split pos-enc, LayerNorm path 24 bit',
    'all_ln24_guard2':      '①+②+③+④ + split pos-enc, LayerNorm 24 bit + 2-bit guard',
    'all_ln18_guard2':      '①+②+③+④, LayerNorm 18 bit + 2-bit guard',
    'all_ln20_guard2':      '①+②+③+④, LayerNorm 20 bit + 2-bit guard',
    'all_ln22_guard2':      '①+②+③+④, LayerNorm 22 bit + 2-bit guard',
    'deploy_nosplit':       '①+②+③+④, LayerNorm 24 bit + 2-bit guard',
    'deploy_gelu_q411':     '①+②+③+④, LN 24 bit + guard, GELU Q4.11   ← deploy',
}
CONFIG_SHORT = {
    'base': 'reference', 'int32_bias': '① bias', 'int8_pos_enc': '② pos-enc',
    'fx16_scales': '③ scales', 'fx_nonlinear': '④ Q16',
    'fx_nonlinear_lndyn': '④ᵃ dyn-exp', 'fx_nonlinear_lnstepq': '④ᵇ per-step',
    'fx_nonlinear_ln32': '④ᶜ Q32',
    'all': 'all (Q16)', 'all_lndyn': 'all + dyn-exp', 'all_lnstepq': 'all + per-step',
    'all_ln32': 'all + Q32', 'all_ln32_splitpos': 'all + Q32 + splitpos',
    'all_ln32_ctr16': 'all + Q32, var 16b', 'all_ln32_ctr32': 'all + Q32, var 32b',
    'all_ln24': 'all + Q24', 'all_ln24_guard2': 'all + Q24 + guard',
    'all_ln18_guard2': 'LN 18b', 'all_ln20_guard2': 'LN 20b',
    'all_ln22_guard2': 'LN 22b',
    'deploy_nosplit': 'LN 24b\n+ guard', 'deploy_gelu_q411': 'deploy\nGELU Q4.11',
}
# figures 1 and 2: the change-by-change story, ending at the deployment config
# (no pos-enc K-split -- that option is measured in the tables, not shipped)
CONFIG_ORDER = ['base', 'int32_bias', 'int8_pos_enc', 'fx16_scales',
                'fx_nonlinear', 'fx_nonlinear_ln32', 'all',
                'deploy_nosplit', 'deploy_gelu_q411']
# figure 4: the four ways to requantize the LayerNorm input, and what each costs in RTL
LN_OPTIONS = [
    ('all',         'Q16\nstatic',      '#2a78d6',
     'static Q16 — constant shift, no extra logic'),
    ('all_lndyn',   'Q16\n+ dyn. exp.', '#eb6834',
     'Q16 + run-time exponent — 128-way max tree + leading-zero count + barrel shifter, '
     'and the whole vector must be held before anything can shift'),
    ('all_lnstepq', 'Q16\nper step',    '#1baf7a',
     'Q16, one format per time step — a 20x4-bit ROM indexed by the step counter + barrel shifter'),
    ('all_ln32',    'Q32\nstatic',      '#eda100',
     'static Q32 — constant shift into a 32-bit register, no extra logic'),
]
SIZE_GROUPS = [
    ('bytes_weights_int8', 'INT8 weights', SERIES[0]),
    ('bytes_pos_encoding', 'positional encoding', SERIES[1]),
    ('bytes_fx_params', 'LayerNorm affine + latent memory', SERIES[2]),
    ('bytes_biases', 'biases', SERIES[3]),
    ('bytes_weight_scales', 'weight scales', SERIES[4]),
    ('bytes_act_scales', 'activation scales', SERIES[5]),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)


def load_results(config_order=None):
    config_order = config_order or CONFIG_ORDER
    df = pd.read_csv(os.path.join(QUANT_DIR, 'hw_quant_summary.csv'))
    base_acc = {}
    for ds in df['dataset'].unique():
        ref = pd.read_csv(os.path.join(QUANT_DIR, ds, 'quantization_results.csv'))
        base_acc[ds] = float(ref.loc[ref['case'] == 'fp32_baseline', 'accuracy'].iloc[0])
    df['fp32_accuracy'] = df['dataset'].map(base_acc)
    df['delta_pp'] = (df['accuracy'] - df['fp32_accuracy']) * 100
    ref = df[df['config'] == 'base'].set_index('dataset')['accuracy']
    df['delta_vs_ref_pp'] = (df['accuracy'] - df['dataset'].map(ref)) * 100
    df['_order'] = df['config'].map({c: i for i, c in enumerate(config_order)})
    return df.dropna(subset=['_order']).sort_values(['dataset', '_order'])


# =============================================================================
def plot_accuracy_drop(df, out_path):
    datasets = [d for d in DATASETS if d in set(df['dataset'])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 5.0),
                             dpi=170, facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds]
        labels = [CONFIG_LABEL[c] for c in sub['config']]
        y = np.arange(len(sub))[::-1]
        vals = sub['delta_pp'].to_numpy()
        colors = [DIVERGE_NEG if v < -1e-9 else (DIVERGE_POS if v > 1e-9 else NEUTRAL)
                  for v in vals]
        ax.barh(y, vals, height=0.62, color=colors, zorder=3)

        base_delta = float(sub.loc[sub['config'] == 'base', 'delta_pp'].iloc[0])
        ax.axvline(base_delta, color=INK_MUTED, ls=(0, (4, 3)), lw=1.0, zorder=2)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=2)

        span = max(abs(vals).max(), 0.2)
        pad = span * 0.55
        ax.set_xlim(min(vals.min(), base_delta) - pad * 2.0, max(0.0, vals.max()) + pad * 1.6)
        # one label per bar: how far it fell, nothing else
        for yi, v in zip(y, vals):
            off = span * 0.09
            ax.text(v - off if v < 0 else v + off, yi, f'{v:+.3f} pp',
                    va='center', ha='right' if v < 0 else 'left',
                    fontsize=8.5, color=INK if abs(v) > 1e-9 else INK_MUTED, zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9, color=INK_2)
        ax.set_xlabel('accuracy vs. fp32 baseline  (percentage points)',
                      fontsize=9, color=INK_2)
        fp32 = float(sub['fp32_accuracy'].iloc[0]) * 100
        n = int(sub['n_eval_samples'].iloc[0])
        ax.set_title(f'{ds}\nfp32 baseline {fp32:.3f}%   ·   {n:,} test samples\n'
                     f'1 test sample = {100 / n:.3f} pp',
                     fontsize=11, color=INK, fontweight='bold', loc='left', pad=12)
        ax.grid(axis='x', color=GRID, lw=1.0, zorder=0)
        ax.set_axisbelow(True)
        _style(ax)

    fig.text(0.5, 0.965,
             'Integer-only (INT8 / INT32 / fixed-point) EvT on ZCU102 — accuracy cost of each change',
             ha='center', fontsize=13, color=INK, fontweight='bold')
    fig.text(0.5, 0.935,
             'red = accuracy lost   ·   gray = unchanged   ·   blue = gained   ·   '
             'dashed line = the INT8 starting point (everything left of it is pre-existing INT8 loss)   ·   '
             'note the per-panel x scale — each panel spans its own dataset’s range',
             ha='center', fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(out_path, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {out_path}')


# =============================================================================
def plot_model_size(df, out_path):
    datasets = [d for d in DATASETS if d in set(df['dataset'])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 5.2),
                             dpi=170, facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds]
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        gap = float(sub['bytes_total'].max()) / 1024 * 0.004   # 2px-equivalent surface gap
        for key, label, color in SIZE_GROUPS:
            vals = sub[key].to_numpy() / 1024.0
            ax.bar(x, vals - gap, bottom=bottom + gap / 2, width=0.62,
                   color=color, label=label, zorder=3)
            bottom += vals
        fp32_kib = float(sub['fp32_ckpt_bytes'].iloc[0]) / 1024.0
        top = bottom.max()
        for xi, tot in zip(x, bottom):
            ax.text(xi, tot + top * 0.02, f'{tot:.0f}\n{tot / fp32_kib * 100:.1f}%',
                    ha='center', va='bottom', fontsize=8.5, color=INK, zorder=4,
                    linespacing=1.35)

        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_SHORT[c] for c in sub['config']],
                           fontsize=8.5, color=INK_2, rotation=30, ha='right')
        ax.set_ylabel('packed model size (KiB, and % of the fp32 checkpoint)',
                      fontsize=9, color=INK_2)
        ax.set_ylim(0, top * 1.22)
        ax.set_title(f'{ds}   ·   fp32 checkpoint {fp32_kib:,.0f} KiB',
                     fontsize=11, color=INK, fontweight='bold', loc='left', pad=10)
        ax.grid(axis='y', color=GRID, lw=1.0, zorder=0)
        ax.set_axisbelow(True)
        _style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.945))
    fig.text(0.5, 0.975, 'Where the bytes go — every tensor is INT8, INT32 or 16-bit fixed point',
             ha='center', fontsize=13, color=INK, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {out_path}')


# =============================================================================
def plot_requant_ranges(out_path, dataset='ASL_DVS'):
    path = os.path.join(QUANT_DIR, dataset, 'hw_quantized_models', 'hw_quant_distributions.csv')
    if not os.path.exists(path):
        print(f'skip requant-range figure: {path} missing')
        return
    d = pd.read_csv(path)
    d = d[d['group'].str.contains('requant')].copy()
    d = d[np.isfinite(d['dynamic_range'])]
    # The residual stream is `latent_vectors`, accumulated across time steps.
    # It feeds layer_norm_1 / layer_norm_att everywhere, layer_norm_x inside the
    # *latent* attentions, and proc_embs_block's LayerNorm. It does NOT feed
    # layer_norm_2 (that comes from a GELU) nor cross_attention's layer_norm_x
    # (that normalizes the event tokens).
    d['is_stream'] = d['name'].str.contains(
        r'layer_norm_(?:1|att)\.in$'
        r'|latent_attentions\.\d+\.layer_norm_x\.in$'
        r'|proc_embs_block\.layer_norm\.in$', regex=True)
    d = d.sort_values('dynamic_range')

    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=170, facecolor=SURFACE)
    for is_stream, yi, label in ((False, 0, f'every other requantization site  (n={int((~d.is_stream).sum())})'),
                                 (True, 1, f'LayerNorm on the residual stream  (n={int(d.is_stream.sum())})')):
        sub = d[d['is_stream'] == is_stream]
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.16, 0.16, len(sub))
        ax.scatter(sub['dynamic_range'], np.full(len(sub), yi) + jitter, s=52,
                   color=DIVERGE_NEG if is_stream else SERIES[0], alpha=0.85,
                   edgecolors=SURFACE, linewidths=2, zorder=4, label=label)

    worst = d.loc[d['dynamic_range'].idxmax()]
    ax.annotate(worst['name'].replace('backbone.proc_memory_blocks.0.', '')
                + f"\n{worst['dynamic_range']:,.0f}x",
                xy=(worst['dynamic_range'], 1), xytext=(worst['dynamic_range'] * 0.55, 1.55),
                fontsize=8, color=INK_2, ha='center',
                arrowprops=dict(arrowstyle='-', color=INK_MUTED, lw=0.9))
    quiet = d[~d['is_stream']]['dynamic_range'].max()
    ax.annotate(f'all of these fit one static Q format\n(worst {quiet:.1f}x)',
                xy=(quiet, 0), xytext=(quiet * 1.6, -0.62),
                fontsize=8, color=INK_2, ha='left',
                arrowprops=dict(arrowstyle='-', color=INK_MUTED, lw=0.9))

    ax.set_xscale('log')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['other sites', 'residual-stream\nLayerNorm'],
                       fontsize=9, color=INK_2)
    ax.set_ylim(-0.95, 2.0)
    ax.set_xlabel('magnitude swing between calls:  max|x| of the loudest call / max|x| of the quietest call   (log scale)',
                  fontsize=9, color=INK_2)
    ax.set_title(f'{dataset}: why LayerNorm inputs need a run-time exponent and nothing else does\n'
                 f'{dataset} runs 20 time steps, so `latent_vectors` accumulates '
                 f'(ASL_DVS runs a single step and shows no spread at all — spread 1.0x)',
                 fontsize=11, color=INK, fontweight='bold', loc='left', pad=10)
    ax.grid(axis='x', color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {out_path}')


def plot_ln_options(out_path):
    """Four ways to requantize the LayerNorm input, and what each costs in RTL.
    All four are measured on top of the same fully-integer configuration, so
    the only thing that differs between the bars is the requantizer."""
    df = load_results([c for c, *_ in LN_OPTIONS])
    datasets = [d for d in DATASETS if d in set(df['dataset'])]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.6 * len(datasets), 4.8),
                             dpi=170, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds].set_index('config')
        x = np.arange(len(LN_OPTIONS))
        vals = [float(sub.loc[c, 'delta_vs_ref_pp']) for c, *_ in LN_OPTIONS]
        colors = [col for _, _, col, _ in LN_OPTIONS]
        span = max(max(abs(v) for v in vals), 0.2)
        # a zero-length bar would be invisible; give it a short stub so every
        # option still reads as a mark
        stub = span * 0.012
        heights = [v if abs(v) > stub else (stub if v >= 0 else -stub) for v in vals]
        ax.bar(x, heights, width=0.6, color=colors, zorder=3)
        ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
        ax.set_ylim(min(min(vals), 0) - span * 0.55, max(max(vals), 0) + span * 0.75)
        for xi, v, c in zip(x, vals, [c for c, *_ in LN_OPTIONS]):
            ax.text(xi, v + span * 0.06 * (1 if v >= 0 else -1), f'{v:+.3f}',
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=8.5, color=INK, zorder=4)
            ax.text(xi, ax.get_ylim()[0] + span * 0.06,
                    f"{float(sub.loc[c, 'accuracy']) * 100:.3f}%", ha='center',
                    fontsize=7.5, color=INK_MUTED, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _, _ in LN_OPTIONS], fontsize=8.5, color=INK_2)
        ax.set_ylabel('accuracy vs. the INT8 reference  (pp)', fontsize=9, color=INK_2)
        n = int(sub['n_eval_samples'].iloc[0])
        steps = '1 time step' if ds == 'ASL_DVS' else '20 time steps'
        ax.set_title(f'{ds}   ·   {steps}\n{n:,} test samples  ·  1 sample = {100 / n:.3f} pp',
                     fontsize=10.5, color=INK, fontweight='bold', loc='left', pad=10)
        ax.grid(axis='y', color=GRID, lw=1.0, zorder=0)
        ax.set_axisbelow(True)
        _style(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col, _ in LN_OPTIONS]
    labels = [cost for _, _, _, cost in LN_OPTIONS]
    fig.legend(handles, labels, loc='upper left', ncol=1, frameon=False,
               fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.02, 0.005),
               title='RTL cost of the requantizer', title_fontsize=9,
               alignment='left')
    fig.text(0.5, 1.0, 'How to requantize the LayerNorm input — four options, same model',
             ha='center', fontsize=13, color=INK, fontweight='bold')
    fig.text(0.5, 0.955,
             'ASL_DVS runs a single time step, so its residual stream never accumulates '
             'and all four options are equivalent there',
             ha='center', fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    df = load_results()
    plot_accuracy_drop(df, os.path.join(QUANT_DIR, 'hw_quant_accuracy_drop.png'))
    plot_model_size(df, os.path.join(QUANT_DIR, 'hw_quant_model_size.png'))
    # DVS128 is the multi-time-step case, which is where the residual stream
    # actually accumulates -- ASL_DVS runs a single time step and shows nothing.
    ds = sys.argv[1] if len(sys.argv) > 1 else (
        'DVS128_10' if 'DVS128_10' in set(df['dataset']) else df['dataset'].iloc[0])
    plot_requant_ranges(os.path.join(QUANT_DIR, 'hw_quant_requant_ranges.png'), ds)
    plot_ln_options(os.path.join(QUANT_DIR, 'hw_quant_layernorm_options.png'))


if __name__ == '__main__':
    main()
