# EvT Quantization Flow (with real example values)

This document walks through **exactly what happens, tensor by tensor**, when an
EvT model is fake-quantized by `quant_lib/`. Every number below was produced by
actually running the trained `DVS128_10` checkpoint on one real validation
sample (see the reproduction snippet at the end of each section) — nothing is
hand-made up. It's meant to answer: *which layers get touched, and what does
quantization actually do to the numbers right before/after GELU and inside
attention?*

For the "why these choices" rationale, see the docstring at the top of
`quant_lib/quant_wrapper.py`. This file focuses on the "what happens
numerically" side.

## 1. Where the hooks sit in the forward pass

```
polarity/pixels
     │
     ▼
┌─────────────────────┐
│ event_projection     │  MLPBlock: Linear(144→96) + GELU        <- W quantized (in-place)
│ (per-event token)    │                                          <- act-quant hook on Linear's INPUT
└─────────────────────┘
     │  (96-dim token, POST-GELU)
     ▼
┌─────────────────────┐
│ proc_event_blocks[i] │  MLPBlock: [Linear+ReLU, Linear+ReLU]    <- W quantized
│ (token-level MLP)    │                                          <- act-quant hook before EACH Linear
└─────────────────────┘
     │  x  (memory tokens, per time step)
     ▼
┌─────────────────────────────┐
│ proc_memory_blocks[i]        │
│  cross_attention:             │
│    attention(z, x, x) ───────┼─►  nn.MultiheadAttention          <- in_proj_weight & out_proj.weight quantized
│      Q=z (latents) K,V=x      │                                  <- act-quant hook on (query,key,value) INPUTS
│      softmax(QKᵀ/√d)·V         │                                  <- softmax / QKᵀ / attn·V stay FP32
│    self_attention (same shape)│
│    MLPBlock (linear1..3)       │                                 <- W + act quantized like above
└─────────────────────────────┘
     │  (repeated per time step, latents z carried across steps)
     ▼
┌─────────────────────┐
│ LatentEmbsCompressor  │  LayerNorm → Linear → ReLU → Linear      <- Linear weights quantized
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ CLFBlock              │  Linear → log_softmax                    <- Linear weight quantized
└─────────────────────┘
     │
     ▼
   logits
```

Kept at **FP32 everywhere** (never quantized): biases, `LayerNorm` affine
params, the learned positional encoding, the latent memory init vectors, and
all non-linear ops themselves (`GELU`, `ReLU`, `softmax`, `log_softmax`). Only
`nn.Linear.weight` and `nn.MultiheadAttention.in_proj_weight` are quantized
(`quant_wrapper.py:52-67`) — these two module types hold essentially all of
the model's parameters and FLOPs.

## 2. Weight quantization — `event_projection` example

`event_projection` is the very first block: a single `Linear(144 → 96)` whose
output feeds a `GELU`. `ModelQuantizer.quantize_weights()` walks every
`nn.Linear`/`nn.MultiheadAttention` once (before any data is seen) and
replaces `weight.data` in place with its fake-quantized version — a
quantize→dequantize round trip, so the tensor is still FP32-typed but only
takes on 2^bits distinct values per (channel or tensor):

```python
scale = max(|w_channel|) / 127                 # INT8 symmetric, per-output-channel
w_int8 = round(w_channel / scale).clamp(-128, 127)
w_fake = w_int8 * scale                         # what the rest of the fp32 model actually sees
```

Real numbers, output-channel #3 of `event_projection`'s weight (`INT8`,
per-channel — the default weight scheme):

| | value |
|---|---|
| scale = max\|w\| / 127 | **0.002017** |
| original weight[:6] | `[0.04746, 0.04445, -0.04761, -0.20481, -0.09047, 0.06737]` |
| → int8 codes | `[24, 22, -24, -102, -45, 33]` |
| dequantized ("fake-quantized") | `[0.0484, 0.04437, -0.0484, -0.20569, -0.09075, 0.06655]` |
| max abs error (this channel) | **0.0010** |

An INT8 weight only ever needs ⌈log2(2·127+1)⌉ = 8 bits to store versus 32
for FP32 → this is where the **model-size reduction** comes from
(`complexity.estimate_model_size_bytes`); the FP32 `scale` is stored once per
channel/tensor, so its overhead is negligible for anything but the smallest
layers.

## 3. Activation quantization around GELU

This is the part the user specifically asked about: **what happens right
before/after the GELU inside `event_projection`?**

The activation-quantization hook (`quant_wrapper.py:111-115`,
`_make_linear_hook`) fires on a `Linear`'s **input**, right before that
`Linear` runs — it never touches what's *between* the `Linear` and its
`GELU`, because `GELU` isn't a hooked module. So the actual flow for a
2-layer block (`event_projection` has only one Linear+GELU; the deeper
`proc_event_blocks` have two) is:

```
 x_input ──►[quantize: INT8]──► Linear #1 (weight: INT8) ──► GELU (fp32) ──► [quantize: INT8]──► Linear #2 (weight: INT8) ──► ReLU (fp32) ──► ...
     ▲ hook here                                                    ▲ hook here (next Linear's input)
```

I.e. quantization is applied to the **input of every Linear**, which for any
Linear that isn't first in a block is exactly *"the activation right after
the previous non-linearity"* — the numerically correct place to insert a
quantizer for fixed-point hardware (you always quantize right before a
matmul, using whatever precision the non-linearity produced).

Concretely, take one real event patch token (a 144-dim log-count vector, only
33/144 dims non-zero) passed through `event_projection` in the `int8_w8a8_static`
case:

```
raw token, nonzero dims [0,4,8,9,12,13] = [0.693, 1.609, 0.693, 1.099, 1.099, 2.485]

Linear(144→96), fp32 weights, output[:6]   = [-2.579,  1.099,  0.542, -0.466, -0.292,  0.627]
+ GELU (fp32, always), output[:6]          = [-0.013,  0.950,  0.383, -0.149, -0.112,  0.461]
```

That GELU output is exactly the tensor the **next** Linear's `forward_pre_hook`
intercepts. Using the scale calibrated from one full pass over the training
set (`calibration.py`), per-tensor INT8:

```
calibrated scale = max|activation over calib set| / 127 = 0.011118

int8 codes (this sample) [:6]  = [-1, 85, 34, -13, -10, 41]
dequantized ("what the next
 Linear actually multiplies")  = [-0.011, 0.945, 0.378, -0.145, -0.111, 0.456]
max abs error on this sample   = 0.0056   (≈0.6% of the GELU output's dynamic range)
```

So GELU's own math is untouched (still fp32, exact) — only its *output*, right
before it's consumed by the next matmul, gets snapped onto a 256-level INT8
grid. This is what an FPGA fixed-point pipeline would actually implement:
GELU stays a small LUT/piecewise-poly in higher precision, its output is
requantized to INT8 before feeding the next systolic-array matmul.

## 4. Attention — Q/K/V and `in_proj_weight`

`AttentionBlock.forward` (`models/EvT.py:54`) calls
`self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)`, i.e.
`query=z` (the latent vectors), `key=value=x` (the memory/token embeddings).
`_make_mha_hook` (`quant_wrapper.py:117-125`) intercepts exactly these three
tensors — **before** they're multiplied by `in_proj_weight` — and
fake-quantizes each one (per-tensor INT8) using its own calibrated scale.

```
                    ┌── quantize(query) ──┐
 query (z) ─────────┤                     │
                    │   Q = query·Wqᵀ + bq   ┐
 key (x)   ── quantize(key) ──► K = key·Wkᵀ + bk ├─  scores = QKᵀ/√d  ──► softmax ──► ·V ──► out_proj ──► output
 value (x) ── quantize(value)──► V = value·Wvᵀ + bv ┘         (fp32)      (fp32)  (fp32)   (weight: INT8)
                                       ▲
                          Wq,Wk,Wv are slices of in_proj_weight (INT8, per-channel)
```

`in_proj_weight` is a single `(3·embed_dim, embed_dim)` Parameter
(`Wq;Wk;Wv` stacked) rather than 3 separate `Linear`s, so it's quantized
directly as a Parameter (`quant_wrapper.py:59-60`), same per-channel INT8
scheme as any other weight:

```
Wq row 0, scale = max|row| / 127        = 0.001444
original   [:6] = [ 0.05134, -0.00720, -0.01800,  0.12724,  0.01732,  0.12543]
dequantized[:6] = [ 0.05199, -0.00722, -0.01733,  0.12709,  0.01733,  0.12564]
```

**Real effect on an actual attention distribution** — real cross-attention
layer (`proc_memory_blocks[0]`), latent query #61, head #2 (chosen because it
has one of the more peaked, non-uniform attention patterns; many
heads/queries in this trained model are close to uniform, which itself
doesn't showcase quantization effects well), comparing FP32 vs the
`int8_w8a8_static` case (INT8 weights **and** INT8 query/key activations) on
the top-6 attended memory tokens:

| attended token idx | 75 | 84 | 70 | 72 | 82 | 67 |
|---|---|---|---|---|---|---|
| FP32 softmax weight | 0.289 | 0.183 | 0.097 | 0.041 | 0.032 | 0.025 |
| INT8 (W8A8) softmax weight | 0.278 | 0.189 | 0.098 | 0.041 | 0.033 | 0.026 |

The **arg-max token (75) is identical**, and the largest change in any single
softmax weight is ≈0.011 (1.1 percentage points) — the attention *pattern* is
preserved, only slightly "blurred", which is consistent with the small (<1pp)
accuracy drop this case shows on the actual test set (see
`quantization_results.csv`).

Note the `QKᵀ/√d`, `softmax`, and `·V` operations themselves are **not**
quantized in this implementation — only their *inputs* (Q/K/V activations,
in_proj weight) are. This mirrors the FPGA data path: you'd quantize the
tensors flowing into/out of the systolic-array matmuls, but accumulate and
run softmax in wider fixed-point/float to avoid compounding numerical error
in a normalization op.

**`out_proj` caveat** (documented in `quant_wrapper.py`'s module docstring,
verified empirically): `out_proj.weight` **is** quantized (it's caught by the
same `isinstance(m, nn.Linear)` walk used everywhere else), but its *input
activation* is not separately hooked — PyTorch's fused
`F.multi_head_attention_forward` reads `out_proj.weight`/`bias` as raw
tensors internally instead of calling `out_proj.forward(...)`, so a
`forward_pre_hook` registered on that submodule never fires. A quick check
confirms this: hooking `out_proj` directly gives **0** calls across a full
forward pass, while the `nn.MultiheadAttention` module's own pre-hook (used
above for Q/K/V) fires once per attention call as expected (60 times for 20
time steps × 3 attention sub-blocks in this model). This applies to
`int8_w8a8_static` (this case, above) specifically -- see
`quant_lib/attention_mac.py` + `run_attention_mac.py` for a from-scratch
reimplementation of attention (bypassing the fused kernel entirely) that
closes this gap and additionally runs `Q.Kᵀ` and `softmax(·)·V` themselves as
real INT8×INT8→INT32 GEMMs instead of fp32.

## 5. Net effect (why the two axes — size vs. BOPs — move differently)

| quantization case | weight bits | act bits | drives... |
|---|---|---|---|
| `int8_weightonly_*`, `int4_weightonly_*` | 4-8 | 32 (fp32) | **model size ↓** (packed weights), BOPs mostly unchanged (activations still fp32-wide in the BOPs model) |
| `int8_w8a8_static`, `int4w_int8a` | 4-8 | 8 | **both** model size ↓ **and** BOPs ↓ (`compute_bops = MACs × weight_bits × act_bits`, `complexity.py`) |

This is exactly the trade-off surfaced in `quantization_comparison.png`'s
bottom row: weight-only cases barely move on the "compute (BOPs) reduction"
axis, while the two activation-quantized cases (`int8_w8a8_static`,
`int4w_int8a`) get the largest BOPs reduction (16x, 32x) at the cost of a
larger — but still modest — accuracy drop, because on real FPGA fixed-point
arithmetic, a matmul's cost scales with *both* operands' bit-widths, not just
the weights'.

## Reproducing the numbers in this document

```bash
conda activate evt_new
cd EventTransformer/quantization
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from quant_lib import data_utils, quant_ops
model, all_params = data_utils.load_model(
    data_utils.find_best_checkpoint('DVS128_10/weights'), 'DVS128_10/all_params.json')
# ... see quant_lib/quant_ops.py: compute_qparams / fake_quantize for the
# exact functions used to produce the weight/activation examples above.
EOF
```
