"""
Generates PPT-ready flowchart PNGs describing the quantized EvT model
(ASL_DVS example, `int8_w8a8_static` case: INT8 weights per-channel +
INT8 activations per-tensor/static).

Outputs (into ASL_DVS/figs/):
  1. flow_overview.png     - full model, per-stage shapes/dtypes/#scales
  2. flow_linear_detail.png- zoom-in: how one quantized nn.Linear computes
  3. flow_attention_detail.png - zoom-in: how AttentionBlock (MHA) computes
  4. outlier_analysis.png  - per-tensor outlier check for attention Q/K/V
                             vs. other activations (real ASL_DVS data)

All numbers (shapes, scales) are read from the real ASL_DVS checkpoint /
calibration stats -- nothing here is illustrative/fake.
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT_DIR = os.path.join(os.path.dirname(__file__), "ASL_DVS", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_FP32 = "#E8E8E8"      # untouched fp32 tensor / op
C_FP32_EDGE = "#888888"
C_W_INT8 = "#BBDEFB"    # int8 weight (stored, per-channel)
C_W_INT8_EDGE = "#1565C0"
C_A_INT8 = "#FFE0B2"    # int8 activation (fake-quantized, per-tensor)
C_A_INT8_EDGE = "#E65100"
C_ATTN = "#C8E6C9"      # attention block
C_ATTN_EDGE = "#2E7D32"
C_OUT = "#F8BBD0"
C_OUT_EDGE = "#AD1457"

FONT = "DejaVu Sans"
plt.rcParams["font.family"] = FONT


def box(ax, xy, w, h, text, fc, ec, fontsize=9.5, weight="normal", lw=1.6, ls="-", zorder=3, align="center"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=weight, zorder=zorder + 1, linespacing=1.35)
    return p


def varrow(ax, x, y0, y1, color="#444444", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle=style, color=color,
                                  mutation_scale=14, linewidth=lw, zorder=2))


def harrow(ax, x0, x1, y, color="#444444", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle=style, color=color,
                                  mutation_scale=14, linewidth=lw, zorder=2))


# ===========================================================================
# FIGURE 1: full-model overview (ASL_DVS, int8_w8a8_static)
# ===========================================================================
def fig_overview():
    fig, ax = plt.subplots(figsize=(12.5, 17.5), dpi=170)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(6.25, 99, "EvT Quantized Inference Pipeline  \u2014  ASL_DVS (24 classes)",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(6.25, 97.3, "Case: int8_w8a8_static  \u2014  W: INT8 per-channel  |  A: INT8 per-tensor (static, calibrated)  |  B = batch, N = #events, L = 96 latents, D = 128, T = #timesteps",
            ha="center", fontsize=9, color="#333333")

    # legend
    ly = 95.4
    legend_items = [
        (C_FP32, C_FP32_EDGE, "fp32 tensor / op (not quantized)"),
        (C_W_INT8, C_W_INT8_EDGE, "INT8 weight, stored per-channel (Cout scales)"),
        (C_A_INT8, C_A_INT8_EDGE, "fake-quantized activation, INT8 per-tensor (1 scale)"),
        (C_ATTN, C_ATTN_EDGE, "attention block (Q\u00b7K\u1d40, softmax, \u00b7V kept fp32)"),
    ]
    lx = 0.4
    for fc, ec, label in legend_items:
        ax.add_patch(Rectangle((lx, ly), 0.35, 0.55, facecolor=fc, edgecolor=ec, linewidth=1.3))
        ax.text(lx + 0.5, ly + 0.27, label, va="center", fontsize=8.2)
        lx += 0.5 + len(label) * 0.072 + 0.55

    y = 93.5
    cx = 6.25
    W = 9.6

    def stage(y, h, text, fc, ec, fontsize=9.3, weight="normal"):
        box(ax, (cx - W / 2, y - h), W, h, text, fc, ec, fontsize=fontsize, weight=weight)
        return y - h

    # ---- Input ----
    y2 = stage(y, 3.0,
               "INPUT  kv: (T, B, N, 144) fp32   pixels: (T, B, N, 2) int  \u2014 raw event-patch tokens (token_dim=144)",
               C_FP32, C_FP32_EDGE, weight="bold")
    varrow(ax, cx, y2, y2 - 1.0); y = y2 - 1.0

    # ---- event_projection ----
    y2 = stage(y, 4.4,
               "event_projection: Linear(144\u219296) + GELU\n"
               "W int8 (96,144) \u2014 96 per-channel scales   |   act\u2192int8 (1 scale = 8.76)\n"
               "out: (T, B, N, 96) fp32",
               C_W_INT8, C_W_INT8_EDGE)
    varrow(ax, cx, y2, y2 - 1.0); y = y2 - 1.0

    # ---- pos enc concat ----
    y2 = stage(y, 2.6,
               "+ Fourier positional encoding (64-d, fp32, not quantized) \u2192 concat \u2192 (T,B,N,160) \u2192 permute \u2192 (T,N,B,160)",
               C_FP32, C_FP32_EDGE)
    varrow(ax, cx, y2, y2 - 1.0); y = y2 - 1.0

    # ---- preproc_block_events ----
    y2 = stage(y, 4.4,
               "preproc_block_events: Linear(160\u2192128) + GELU\n"
               "W int8 (128,160) \u2014 128 per-channel scales   |   act\u2192int8 (1 scale = 15.43)\n"
               "out: (T, N, B, 128) fp32",
               C_W_INT8, C_W_INT8_EDGE)
    varrow(ax, cx, y2, y2 - 1.0); y = y2 - 1.0

    # ---- loop box start ----
    loop_top = y
    loop_bottom_h = 26.6
    loop_box = FancyBboxPatch((cx - W / 2 - 0.35, loop_top - loop_bottom_h), W + 0.7, loop_bottom_h,
                               boxstyle="round,pad=0.02,rounding_size=0.12",
                               linewidth=2.0, edgecolor="#555555", facecolor="none", linestyle=(0, (6, 3)), zorder=1)
    ax.add_patch(loop_box)
    ax.text(cx - W / 2 - 0.15, loop_top - 0.35, "for t = 1..T  (recurrent memory update; loops back to top each step)",
            fontsize=8.8, style="italic", color="#444444")

    y = loop_top - 1.2

    # proc_event_blocks
    y2 = stage(y, 4.6,
               "proc_event_blocks (MLPBlock, +residual): inp_kv = kv[t]: (N,B,128)\n"
               "Linear1(128\u2192128)+ReLU: W int8/128 sc, act sc=11.98  \u2192  Linear2(128\u2192128)+ReLU: W int8/128 sc, act sc=17.27\n"
               "out += inp_kv  \u2192  (N, B, 128) fp32",
               C_W_INT8, C_W_INT8_EDGE, fontsize=8.7)
    varrow(ax, cx, y2, y2 - 0.9); y = y2 - 0.9

    # cross-attention
    y2 = stage(y, 7.6,
               "cross_attention (AttentionBlock, heads=4)\n"
               "LN(x) \u2192 K=V: (N,B,128)  |  LN(z) \u2192 Q: (L=96,B,128)\n"
               "in_proj W int8 (384,128) \u2014 384 sc  \u2192 Wq,Wk,Wv (128,128 each)\n"
               "act\u2192int8: q sc=2.00, k sc=4.59, v sc=4.59 (per-tensor)\n"
               "Q\u00b7K\u1d40/\u221a32 \u2192 softmax \u2192 \u00b7V  [fp32, NOT quantized]  \u2192 (L,B,128)\n"
               "out_proj Linear(128\u2192128): W int8/128 sc (act NOT hooked, see note)\n"
               "+residual \u2192 Linear1\u2192GELU\u2192Linear2\u2192GELU\u2192Linear3 (128\u2192128 each, W int8/128 sc)\n"
               "\u2192 z_cross: (L=96, B, 128) fp32",
               C_ATTN, C_ATTN_EDGE, fontsize=8.1)
    varrow(ax, cx, y2, y2 - 0.9); y = y2 - 0.9

    # self-attention x2
    y2 = stage(y, 5.6,
               "latent_attentions[0], [1]  (self-attention AttentionBlock \u00d7 2, heads=4)\n"
               "Q=K=V = z: (L=96,B,128)   |   in_proj W int8 (384,128)/384 sc each block\n"
               "act sc (block0): q=3.72, k=1.78, v=1.78   |   +FFN (Linear1-3, W int8/128 sc)\n"
               "\u2192 inp_q: (96, B, 128) fp32",
               C_ATTN, C_ATTN_EDGE, fontsize=8.3)
    varrow(ax, cx, y2, y2 - 0.9); y = y2 - 0.9

    y2 = stage(y, 3.0,
               "mask pad timesteps \u2192 latent_vectors = inp_q + latent_vectors  (carried to next t)\n"
               "\u2192 (96, B, 128) fp32",
               C_FP32, C_FP32_EDGE, fontsize=8.6)
    loop_bottom_y = y

    # loop-back arrow (visual): routed along the left margin, outside the box
    lb_x = cx - W / 2 - 0.75
    ax.add_patch(FancyArrowPatch((cx - W / 2 - 0.35, loop_bottom_y + 0.3), (lb_x, loop_bottom_y + 0.3),
                                  arrowstyle="-", color="#888888", lw=1.6, zorder=2))
    ax.add_patch(FancyArrowPatch((lb_x, loop_bottom_y + 0.3), (lb_x, loop_top - 0.65),
                                  arrowstyle="-", color="#888888", lw=1.6, zorder=2))
    ax.add_patch(FancyArrowPatch((lb_x, loop_top - 0.65), (cx - W / 2 - 0.35, loop_top - 0.65),
                                  arrowstyle="-|>", color="#888888", lw=1.6, mutation_scale=13, zorder=2))
    ax.text(lb_x - 0.15, (loop_bottom_y + loop_top) / 2, "next t", rotation=90, ha="center", va="center",
            fontsize=8, color="#666666", style="italic")

    y = loop_top - loop_bottom_h - 1.0

    # proc_embs_block
    y2 = stage(y, 3.8,
               "proc_embs_block (LatentEmbsCompressor): LN(128) \u2192 Linear(128\u2192128) [W int8/128 sc, act sc]\n"
               "\u2192 ReLU \u2192 mean over L=96 latents  \u2192  (B, 128) fp32",
               C_W_INT8, C_W_INT8_EDGE, fontsize=8.6)
    varrow(ax, cx, y2, y2 - 0.9); y = y2 - 0.9

    # CLF
    y2 = stage(y, 3.8,
               "CLFBlock: Linear(128\u2192128) [W int8/128 sc] \u2192 ReLU \u2192 Linear(128\u219224) [W int8/24 sc]\n"
               "\u2192 log_softmax \u2192  (B, 24) fp32  logits",
               C_OUT, C_OUT_EDGE, fontsize=8.8)
    varrow(ax, cx, y2, y2 - 0.9); y = y2 - 0.9

    y_final = stage(y, 2.2, "OUTPUT: class log-probabilities  (B, 24)  fp32", C_FP32, C_FP32_EDGE, weight="bold")

    ax.text(6.25, y_final - 1.6,
            "Totals (all Linear + MHA.in_proj weights): 22 quantized tensors, 3448 per-channel weight scales, 25 per-tensor activation scales.\n"
            "Bias / LayerNorm affine / positional-encoding table / latent memory vectors stay fp32 (96,632 params, 17.8% of total 543,608).",
            ha="center", fontsize=8.3, color="#333333")

    ax.set_ylim(y_final - 3.2, 100)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "flow_overview.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# ===========================================================================
# FIGURE 2: zoom-in on ONE quantized nn.Linear (proc_event_blocks.linear1)
#   -> shows real requant/dequant arithmetic, and contrasts the current
#      fp32-simulation code path with the target INT8-GEMM FPGA path.
# ===========================================================================
def fig_linear_detail():
    fig, ax = plt.subplots(figsize=(13, 8.6), dpi=170)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(6.5, 8.3, "Quantized nn.Linear \u2014 detail (example: proc_event_blocks.linear1, 128\u2192128)",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6.5, 7.85,
            "x: (N,B,128) fp32   |   W: (128,128)   |   b: (128,) fp32   |   weight scale s_w: per-channel (128 values)   |   act scale s_x: per-tensor static = 11.98",
            ha="center", fontsize=9.3, color="#333333")

    # --- Row A: current codebase path (fp32 simulation, matches quant_ops.py) ---
    ay = 6.5
    ax.text(0.3, ay + 0.85, "A. Current implementation (accuracy simulation \u2014 quant_lib/quant_ops.py)", fontsize=10.5, fontweight="bold", color="#1565C0")

    box(ax, (0.3, ay - 0.55), 2.3, 1.1, "x  fp32\n(N,B,128)", C_FP32, C_FP32_EDGE, fontsize=9)
    harrow(ax, 2.6, 3.1, ay)
    box(ax, (3.1, ay - 0.55), 2.7, 1.1,
        "fake_quantize(x, s_x, 8)\nq=round(x/s_x).clamp(\u2212128,127)\nreturn q\u00d7s_x", C_A_INT8, C_A_INT8_EDGE, fontsize=7.8)
    harrow(ax, 5.8, 6.3, ay)
    box(ax, (6.3, ay - 0.55), 1.9, 1.1, "x_fq  fp32\n(int8-valued\ngrid, 256 lvls)", C_A_INT8, C_A_INT8_EDGE, fontsize=8)
    harrow(ax, 8.2, 8.7, ay)
    box(ax, (8.7, ay - 0.75), 2.2, 1.5,
        "F.linear(x_fq, W_fq, b)\n(plain fp32 matmul,\nW_fq pre-baked at\nquantize_weights() time)",
        C_W_INT8, C_W_INT8_EDGE, fontsize=7.8)
    harrow(ax, 10.9, 11.9, ay)
    box(ax, (11.9, ay - 0.55), 0.9, 1.1, "y\nfp32", C_FP32, C_FP32_EDGE, fontsize=9)

    ax.text(0.3, ay - 1.35,
            "\u2192 W_fq is produced once, offline: W_fq = fake_quantize(W, s_w[cout], 8)  (per-channel, 128 scales; stored back into the fp32 Parameter)",
            fontsize=8.4, color="#444444")
    ax.text(0.3, ay - 1.75,
            "This is why accuracy \u2248 real-int8 accuracy, but wall-clock compute is still fp32 \u2014 only real_quant.py's exported .pt file is actually bit-packed on disk.",
            fontsize=8.4, color="#444444")

    ax.plot([0.15, 12.85], [ay - 2.25, ay - 2.25], color="#cccccc", lw=1.2)

    # --- Row B: target FPGA path (real int8 GEMM + requantization) ---
    by = ay - 3.3
    ax.text(0.3, by + 0.85, "B. Target FPGA datapath (what the fake-quant above emulates)", fontsize=10.5, fontweight="bold", color="#AD1457")

    box(ax, (0.3, by - 0.55), 1.7, 1.1, "x_int8\n(N,B,128)\nINT8", C_A_INT8, C_A_INT8_EDGE, fontsize=8.3)
    box(ax, (0.3, by - 1.35), 1.7, 0.55, "s_x  (1 fp32 value)", C_FP32, C_FP32_EDGE, fontsize=7.5)
    harrow(ax, 2.0, 2.7, by)
    box(ax, (2.7, by - 0.75), 2.4, 1.5,
        "INT8 \u00d7 INT8 MAC array\n\u03a3 x_int8[n]\u00b7W_int8[cout,n]\n\u2192 INT32 accumulator\n(exact, no rounding yet)", C_W_INT8, C_W_INT8_EDGE, fontsize=7.8)
    box(ax, (2.7, by - 2.35), 2.4, 0.55, "s_w[cout]  (128 fp32 values)", C_FP32, C_FP32_EDGE, fontsize=7.3)
    harrow(ax, 5.1, 5.8, by)
    box(ax, (5.8, by - 0.75), 2.7, 1.5,
        "requantize:\ny_i32 \u00d7 (s_x\u00b7s_w[cout]) + b\n\u2192 fp32, then\nround(\u00b7/s_y).clamp(int8)", C_A_INT8, C_A_INT8_EDGE, fontsize=7.6)
    harrow(ax, 8.5, 9.2, by)
    box(ax, (9.2, by - 0.55), 1.7, 1.1, "y_int8\n(N,B,128)\nINT8 \u2192 next layer", C_A_INT8, C_A_INT8_EDGE, fontsize=8)
    box(ax, (9.2, by - 1.35), 1.7, 0.55, "s_y (calibrated, 1 value)", C_FP32, C_FP32_EDGE, fontsize=7.3)

    ax.text(0.3, by - 2.9,
            "Only INT8/INT32 fixed-point math + a per-channel scalar multiply is needed on-chip \u2014 no fp32 units required for the Linear/attention-projection datapath.",
            fontsize=8.4, color="#444444")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "flow_linear_detail.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# ===========================================================================
# FIGURE 3: zoom-in on the AttentionBlock (cross_attention example)
# ===========================================================================
def fig_attention_detail():
    fig, ax = plt.subplots(figsize=(13.2, 11.6), dpi=170)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    ax.text(6.6, 11.3, "Quantized Attention \u2014 detail (example: proc_memory_blocks[0].cross_attention, heads=4, d_head=32)",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(6.6, 10.88,
            "Q source = latent z: (L=96,B,128)   |   K,V source = events x: (N,B,128)   |   in_proj_weight: (384,128) \u2014 384 per-channel scales (split Wq,Wk,Wv)",
            ha="center", fontsize=9, color="#333333")

    y0 = 9.9
    # Q/K/V projection row
    box(ax, (0.3, y0 - 0.6), 2.1, 1.2, "z (Q in)\nfp32 (L,B,128)", C_FP32, C_FP32_EDGE, fontsize=8.5)
    box(ax, (0.3, y0 - 2.1), 2.1, 1.2, "x (K,V in)\nfp32 (N,B,128)", C_FP32, C_FP32_EDGE, fontsize=8.5)
    harrow(ax, 2.4, 3.0, y0)
    harrow(ax, 2.4, 3.0, y0 - 1.5)
    box(ax, (3.0, y0 - 1.9), 2.5, 2.4,
        "act\u2192int8 fake-quant\n(per-tensor, static)\nq sc=2.00  k sc=4.59\nv sc=4.59", C_A_INT8, C_A_INT8_EDGE, fontsize=8)
    harrow(ax, 5.5, 6.1, y0 - 0.7)
    box(ax, (6.1, y0 - 1.9), 2.6, 2.4,
        "\u00d7 Wq,Wk,Wv (int8,\n128,128 each, from\nin_proj W int8 384\u00d7128)\n\u2192 Q:(L,B,4,32)\nK,V:(N,B,4,32)", C_W_INT8, C_W_INT8_EDGE, fontsize=7.8)
    harrow(ax, 8.7, 9.3, y0 - 0.7)
    box(ax, (9.3, y0 - 1.4), 3.4, 1.9,
        "Q\u00b7K\u1d40 / \u221a32  \u2192  softmax  \u2192  \u00b7V\nkept in fp32 \u2014 NOT quantized\n(scores/softmax dominate accuracy,\nsee outlier note below)", C_ATTN, C_ATTN_EDGE, fontsize=8.2)

    col_x = 9.3 + 1.7  # center x of the right-hand column, for vertical connector arrows
    box_w, box_x = 3.4, 9.3

    # attn_out box
    b2_top, b2_bot = 6.7, 5.5
    box(ax, (box_x, b2_bot), box_w, b2_top - b2_bot, "attn_out: fp32\n(L=96, B, 128)", C_FP32, C_FP32_EDGE, fontsize=8.5)
    ax.add_patch(FancyArrowPatch((col_x, y0 - 1.4), (col_x, b2_top), arrowstyle="-|>", color="#444444", mutation_scale=13, lw=1.6))

    # out_proj box
    b3_top, b3_bot = 4.9, 3.4
    box(ax, (box_x, b3_bot), box_w, b3_top - b3_bot,
        "out_proj Linear(128\u2192128)\nW int8/128 sc  (bias fp32)\n\u26a0 input act NOT hooked \u2014\nruns in fp32 (see note)", C_W_INT8, C_W_INT8_EDGE, fontsize=7.8)
    ax.add_patch(FancyArrowPatch((col_x, b2_bot), (col_x, b3_top), arrowstyle="-|>", color="#444444", mutation_scale=13, lw=1.6))

    # z_att box
    b4_top, b4_bot = 2.8, 1.6
    box(ax, (box_x, b4_bot), box_w, b4_top - b4_bot,
        "z_att = out+z_input\nfp32 (L=96,B,128)\n\u2192 LN \u2192 FFN (Linear1-3)", C_FP32, C_FP32_EDGE, fontsize=8)
    ax.add_patch(FancyArrowPatch((col_x, b3_bot), (col_x, b4_top), arrowstyle="-|>", color="#444444", mutation_scale=13, lw=1.6))

    # Outlier note panel (kept clear of the right-hand column, which ends at y=1.6)
    panel_bot, panel_top = 0.25, 1.35
    ax.add_patch(FancyBboxPatch((0.3, panel_bot), 12.6, panel_top - panel_bot, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 linewidth=1.6, edgecolor="#E65100", facecolor="#FFF3E0"))
    ax.text(0.55, panel_top - 0.22, "Outlier check for per-tensor Q/K/V activation quantization (real ASL_DVS train data, 60 batches, |x| stats):",
            fontsize=9.0, fontweight="bold", color="#E65100")
    ax.text(0.55, panel_top - 0.52,
            "max/p99 ratio: cross-attn Q/K/V = 2.2\u20132.6\u00d7, self-attn Q/K/V = 2.3\u20132.8\u00d7 \u2014 NOT worse than other per-tensor activations "
            "(procEvt-Linear-in = 2.8\u20135.5\u00d7, event_proj-in = 5.4\u00d7). No LLM-style (>20\u00d7) extreme outliers in Q/K/V overall.",
            fontsize=8.0)
    ax.text(0.55, panel_top - 0.85,
            "Exception: cross-attn Query has a per-channel max/median ratio of ~43\u00d7 on a couple of latent-query channels (see outlier_analysis.png) \u2014 "
            "still safe under per-tensor (uses the global max) but the most outlier-prone tensor found; would benefit most from per-channel scaling if pushed <8-bit.",
            fontsize=8.0, color="#8a4a00")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "flow_attention_detail.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# ===========================================================================
# FIGURE 4: outlier analysis bar charts (real ASL_DVS activations)
# ===========================================================================
def fig_outlier_analysis():
    # stats gathered from real ASL_DVS train-set forward passes (60 batches);
    # see chat / terminal logs for the measurement script.
    tensor_stats = [
        # name, mean, p99, p999, max
        ("event_projection.in", 0.1189, 1.6094, 1.9459, 8.7603),
        ("procEvt.linear1.in", 0.3519, 2.3560, 2.9564, 6.6549),
        ("procEvt.linear2.in", 0.1324, 2.6906, 3.3466, 14.6522),
        ("crossAttn.linear1.in", 0.2268, 0.7916, 1.0682, 1.9861),
        ("selfAttn0.linear1.in", 0.2377, 0.8505, 1.1736, 2.0632),
        ("crossAttn.Q", 0.0946, 0.7607, 1.2112, 2.0025),
        ("crossAttn.K", 0.2269, 1.8746, 2.2203, 4.2734),
        ("crossAttn.V", 0.2270, 1.8746, 2.2203, 4.1909),
        ("selfAttn0.Q", 0.3009, 1.1642, 1.6111, 3.1912),
        ("selfAttn0.K", 0.1629, 0.6513, 0.9307, 1.6837),
        ("selfAttn0.V", 0.1629, 0.6515, 0.9304, 1.7960),
    ]
    is_attn = [("Attn" in n and (".Q" in n or ".K" in n or ".V" in n)) for n, *_ in tensor_stats]
    names = [n for n, *_ in tensor_stats]
    ratio_p99 = [mx / p99 for _, _, p99, _, mx in tensor_stats]
    ratio_p999 = [mx / p999 for _, _, _, p999, mx in tensor_stats]

    chan_stats = [
        ("procEvt.linear2.in", 4.33),
        ("crossAttn.Q", 42.67),
        ("crossAttn.K", 2.70),
        ("crossAttn.V", 2.70),
        ("selfAttn0.Q", 2.29),
        ("selfAttn0.K", 2.28),
        ("selfAttn0.V", 2.28),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=170, gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    x = np.arange(len(names))
    colors = ["#EF6C00" if a else "#1565C0" for a in is_attn]
    w = 0.38
    ax.bar(x - w / 2, ratio_p99, width=w, color=colors, alpha=0.9, label="max / p99")
    ax.bar(x + w / 2, ratio_p999, width=w, color=colors, alpha=0.45, label="max / p99.9")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("outlier ratio (higher = more extreme tail)")
    ax.set_title("Per-tensor outlier ratio: attention Q/K/V (orange) vs. other\nper-tensor-quantized activations (blue)  \u2014  ASL_DVS, 60 train batches", fontsize=10.5)
    ax.axhline(1.0, color="#999999", lw=0.8)
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    handles = [Patch(color="#EF6C00", label="attention Q/K/V"), Patch(color="#1565C0", label="other activations")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper right")
    ax.text(0.99, 0.90, "darker bar = max/p99\nlighter bar = max/p99.9", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#555555")

    ax2 = axes[1]
    names2 = [n for n, _ in chan_stats]
    vals2 = [v for _, v in chan_stats]
    colors2 = ["#1565C0" if "procEvt" in n else "#EF6C00" for n in names2]
    ax2.barh(names2[::-1], vals2[::-1], color=colors2[::-1])
    ax2.set_xlabel("channel-max / channel-median ratio")
    ax2.set_title("Per-channel structure check\n(is the outlier a specific channel, or spread out?)", fontsize=10.2)
    ax2.grid(axis="x", alpha=0.25)
    for i, v in enumerate(vals2[::-1]):
        ax2.text(v + 0.6, i, f"{v:.1f}\u00d7", va="center", fontsize=8.5)

    fig.suptitle("Attention per-tensor quantization \u2014 outlier check (ASL_DVS)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "outlier_analysis.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_overview()
    fig_linear_detail()
    fig_attention_detail()
    fig_outlier_analysis()
