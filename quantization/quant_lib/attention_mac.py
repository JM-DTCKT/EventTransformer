"""
Real INT8xINT8->INT32 MAC computation for `AttentionBlock`'s attention
(from scratch reimplementation of `nn.MultiheadAttention`, this codebase's
exact usage pattern only -- see `models/EvT.py:AttentionBlock.forward`).

Why this module exists
-----------------------
The existing `int8_w8a8_static` case (`quant_wrapper.ModelQuantizer`) only
fake-quantizes the tensors going *into* `nn.MultiheadAttention` (`query`,
`key`, `value` -- i.e. the raw latent/token tensors *before* the Wq/Wk/Wv
projection). Everything that happens *inside* that module -- the Wq/Wk/Wv
projection, `Q.K^T`, `softmax`, `softmax(...)@V`, and the final `out_proj`
projection -- is computed by PyTorch's fused `F.multi_head_attention_forward`
kernel entirely in fp32, because a `forward_pre_hook` can only see a module's
*external* inputs, never its internal intermediate tensors. So on real FPGA
fixed-point hardware, where `Q.K^T` and `softmax(...)@V` would also be int8
MACs, `int8_w8a8_static`'s reported accuracy is optimistic: it never actually
exercises int8 arithmetic for either attention matmul, nor for `out_proj`.

This module fixes that by monkey-patching each `AttentionBlock` instance to
run `attention_forward()` below -- a faithful from-scratch reimplementation
of `nn.MultiheadAttention(query, key, value, key_padding_mask=...)` for this
codebase's fixed usage (batch_first=False, add_bias_kv=True, attn_mask always
None, eval mode) -- instead of calling the fused module. This gives us direct
access to every intermediate tensor, so we can insert *real* INT8xINT8->INT32
GEMMs (`real_int8_matmul`, not fake-quantize-then-fp32-matmul) exactly where
an FPGA systolic array would sit: `Q.K^T`, `softmax(scores)@V`, and
`out_proj`'s input.

Three cumulative stages (`STAGES`), matching what was asked for:
  * 'baseline'  : reproduces `int8_w8a8_static` (pre-projection Q/K/V
                  activations + all weights quantized; `Q.K^T` and
                  `softmax(...)@V` computed in fp32 on the *unquantized*
                  post-projection Q/K/V) -- **plus** `out_proj`'s input
                  activation is now also quantized and its matmul is a real
                  int8 GEMM (the one gap `int8_w8a8_static` always had, see
                  `quant_wrapper.py`'s docstring). This is the fairest
                  reference point: identical to the original case except for
                  the out_proj fix everyone agreed was missing.
  * 'qk_mac'    : 'baseline' + Q and K are *also* quantized right after their
                  projection (a second, independent quantizer from the
                  pre-projection one), and `Q.K^T` is computed as a real
                  int8 GEMM instead of fp32.
  * 'qkv_mac'   : 'qk_mac' + V is also quantized after its projection and
                  `softmax(scores)` is quantized (its range is analytically
                  [0, 1], so no calibration is needed for it -- see
                  `SoftmaxCalibrator`), and `softmax(...)@V` is computed as a
                  real int8 GEMM too.

Why "fake-quantize + fp32 matmul" and "real int8 GEMM" give identical numbers
-------------------------------------------------------------------------------
For every matmul in this file the two are numerically IDENTICAL, not merely
"close": once a tensor has been snapped onto a 256-level int8 grid, each
element equals `q_int * scale` for some integer `q_int` in [-128, 127]. Their
product is `scale_a * scale_b * (q_int_a * q_int_b)`, and float32 represents
every integer up to 2**24 exactly, so summing `q_int_a * q_int_b` products
over a reduction axis of length `d` in float32 is bit-exact as long as
`127 * 127 * d < 2**24`. That holds comfortably for every matmul here (see
`REAL_INT8_ACCUM_BOUND` and the runtime assertion in `real_int8_matmul`):
  * in-proj (Wq/Wk/Wv):  d = embed_dim   (128 here)      -> 127*127*128 ~ 2.06M
  * Q.K^T:               d = head_dim    (32 here)       -> 127*127*32  ~ 0.52M
  * softmax(.)@V:        d = Lk (#keys, <=~400 datasets)  -> 127*127*400 ~ 6.45M
  * out_proj:            d = embed_dim   (128 here)       -> 127*127*128 ~ 2.06M
all well under 2**24 = 16,777,216. This module still implements Q.K^T,
softmax(.)@V and out_proj as an *explicit* round -> clamp -> integer-valued
matmul -> rescale (rather than relying on the "it's mathematically the same"
argument, which the in-proj projection above still uses, matching the rest of
the codebase's convention for plain `nn.Linear`s), so a skeptical reader can
see the literal integer arithmetic happening, not just trust an equivalence
argument. `sanity_check_against_reference()` at the bottom of this file
verifies end-to-end that (with quantization disabled) `attention_forward`
reproduces PyTorch's own `nn.MultiheadAttention` bit-for-bit up to fp32
rounding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import quant_ops

STAGES = ('baseline', 'qk_mac', 'qkv_mac')
REAL_INT8_ACCUM_BOUND = 2 ** 24


# =============================================================================
# Calibrators
# =============================================================================
class TensorCalibrator:
    """Per-tensor symmetric INT8 calibrator (max(|x|) / qmax), with an
    optional mask to exclude positions (e.g. padded key/value tokens) from
    the statistic -- those can carry non-zero bias-only activations that have
    nothing to do with real event content."""

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmax = 2 ** (num_bits - 1) - 1
        self.calibrating = False
        self._max_abs = None
        self.scale = None

    def start(self):
        self.calibrating = True
        self._max_abs = None
        self.scale = None

    @torch.no_grad()
    def observe(self, x, exclude_mask=None):
        xabs = x.detach().abs()
        if exclude_mask is not None:
            xabs = torch.where(exclude_mask, torch.zeros((), dtype=xabs.dtype, device=xabs.device), xabs)
        m = xabs.max()
        self._max_abs = m if self._max_abs is None else torch.maximum(self._max_abs, m)

    def finalize(self):
        self.calibrating = False
        assert self._max_abs is not None, "finalize() called before any observe()"
        self.scale = (self._max_abs.clamp(min=1e-8) / self.qmax).detach()

    def to_int(self, x):
        return torch.clamp(torch.round(x / self.scale), -self.qmax - 1, self.qmax)

    def fake_quantize(self, x):
        return self.to_int(x) * self.scale


class SoftmaxCalibrator:
    """Quantizer for a `softmax` output: every row sums to 1 and every
    element is in [0, 1] by construction, so the theoretical max (1.0,
    attention collapsed onto a single key) is known analytically -- no
    calibration pass is needed, `scale = 1 / qmax` is fixed up front. Codes
    always land in [0, qmax] (never negative)."""

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmax = 2 ** (num_bits - 1) - 1
        self.scale = 1.0 / self.qmax
        self.calibrating = False  # never calibrates; kept for a uniform interface

    def to_int(self, x):
        return torch.clamp(torch.round(x / self.scale), 0, self.qmax)


def _maybe_observe_or_pass(calib, x, exclude_mask=None):
    """During calibration: record statistics, return `x` unchanged (so the
    surrounding fp32 forward pass is exact, matching `calibration.py`'s
    convention elsewhere in this codebase). Otherwise: fake-quantize `x`."""
    if calib.calibrating:
        calib.observe(x, exclude_mask=exclude_mask)
        return x
    return calib.fake_quantize(x)


# =============================================================================
# Real INT8xINT8 -> INT32 GEMM (see module docstring for the exactness proof)
# =============================================================================
def real_int8_matmul(a, a_scale, b, b_scale, transpose_b=False, num_bits=8):
    """`a`, `b`: fp32 tensors (arbitrary batch dims). Rounds both onto an
    N-bit signed integer grid, contracts over the last dimension with an
    integer-valued matmul (stored as float32 -- exact, see module docstring),
    then rescales. Both `a_scale` and `b_scale` must be *scalars* (a single
    shared scale per tensor) -- this is used only for the two attention
    matmuls (`Q.K^T`, `softmax(.)@V`), where that always holds. For a matmul
    against a per-output-channel-scaled weight matrix, use
    `real_int8_linear` instead (its scale must vary along the *output*
    dimension, which is not one of this function's contracted dimensions,
    so it needs different broadcasting)."""
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax - 1
    reduce_dim = a.shape[-1]
    bound = qmax * qmax * reduce_dim
    assert bound < REAL_INT8_ACCUM_BOUND, (
        f"INT8 accumulator bound ({bound}) for a reduction of size {reduce_dim} exceeds "
        f"float32's exact-integer range ({REAL_INT8_ACCUM_BOUND}) -- the float32-stored-int "
        f"trick used here would no longer be bit-exact; use real int32 accumulation instead.")

    a_int = torch.clamp(torch.round(a / a_scale), qmin, qmax)
    b_int = torch.clamp(torch.round(b / b_scale), qmin, qmax)
    out_int = torch.matmul(a_int, b_int.transpose(-2, -1) if transpose_b else b_int)
    return out_int * (a_scale * b_scale)


def real_int8_linear(x, x_scale, weight, weight_scale, bias=None, num_bits=8):
    """Real INT8xINT8->INT32 GEMM for `x @ weight.T + bias` (i.e. `F.linear`),
    where `weight` carries a *per-output-channel* scale (as produced by
    `quant_ops.compute_qparams(..., per_channel=True, channel_dim=0)`).

    x: (..., E_in) fp32 activation. x_scale: scalar.
    weight: (E_out, E_in) fp32 (already snapped onto the int8 grid in place
        by `ModelQuantizer.quantize_weights()`).
    weight_scale: (E_out, 1) -- broadcasts correctly against `weight`'s own
        shape for the round step (each output row divided by its own scale);
        deliberately kept 2-D here rather than a flat `(E_out,)` vector,
        since a flat vector would (mis-)align against `weight`'s *last* dim
        (E_in) instead of its first (E_out) under numpy/torch broadcasting.
    """
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax - 1
    reduce_dim = x.shape[-1]
    bound = qmax * qmax * reduce_dim
    assert bound < REAL_INT8_ACCUM_BOUND, (
        f"INT8 accumulator bound ({bound}) for a reduction of size {reduce_dim} exceeds "
        f"float32's exact-integer range ({REAL_INT8_ACCUM_BOUND}).")
    assert weight_scale.dim() == 2 and weight_scale.shape == (weight.shape[0], 1), (
        f"weight_scale must be shape (E_out, 1) = {(weight.shape[0], 1)}, got {tuple(weight_scale.shape)}")

    x_int = torch.clamp(torch.round(x / x_scale), qmin, qmax)
    w_int = torch.clamp(torch.round(weight / weight_scale), qmin, qmax)   # (E_out, E_in)
    out_int = torch.matmul(x_int, w_int.transpose(-2, -1))                # (..., E_out)
    out = out_int * (x_scale * weight_scale.view(-1))                     # (E_out,) broadcasts over output's last dim
    if bias is not None:
        out = out + bias
    return out


# =============================================================================
# Head reshape helpers (must match torch's internal MultiheadAttention split
# exactly: embed_dim cut into `num_heads` contiguous chunks of `head_dim`)
# =============================================================================
def split_heads(t, num_heads):
    """(L, B, E) -> (B, H, L, Dh)."""
    L, B, E = t.shape
    head_dim = E // num_heads
    return t.reshape(L, B, num_heads, head_dim).permute(1, 2, 0, 3)


def merge_heads(t):
    """(B, H, L, Dh) -> (L, B, E), inverse of `split_heads`."""
    B, H, L, Dh = t.shape
    return t.permute(2, 0, 1, 3).reshape(L, B, H * Dh)


# =============================================================================
# Per-AttentionBlock calibrator bundle
# =============================================================================
class AttentionCalibrators:
    """Every quantizer a single `AttentionBlock` needs, for a given `stage`.
    `q_in`/`k_in`/`v_in` (pre-projection) and `out_proj` (post attn@V, pre
    out_proj weight) exist for every stage; `q_proj`/`k_proj` only from
    'qk_mac' on; `v_proj`/`attn` only for 'qkv_mac'."""

    def __init__(self, stage, num_bits=8):
        assert stage in STAGES
        self.stage = stage
        self.q_in = TensorCalibrator(num_bits)
        self.k_in = TensorCalibrator(num_bits)
        self.v_in = TensorCalibrator(num_bits)
        self.out_proj = TensorCalibrator(num_bits)
        self.q_proj = TensorCalibrator(num_bits) if stage in ('qk_mac', 'qkv_mac') else None
        self.k_proj = TensorCalibrator(num_bits) if stage in ('qk_mac', 'qkv_mac') else None
        self.v_proj = TensorCalibrator(num_bits) if stage == 'qkv_mac' else None
        self.attn = SoftmaxCalibrator(num_bits) if stage == 'qkv_mac' else None

    def _all(self):
        return [c for c in (self.q_in, self.k_in, self.v_in, self.out_proj,
                             self.q_proj, self.k_proj, self.v_proj) if c is not None]

    def start_calibration(self):
        for c in self._all():
            c.start()

    def finalize_calibration(self):
        for c in self._all():
            c.finalize()


# =============================================================================
# The from-scratch attention forward
# =============================================================================
def attention_forward(q_in, k_in, v_in, mha, key_padding_mask, stage, calib):
    """Faithful reimplementation of
    `nn.MultiheadAttention(q_in, k_in, v_in, key_padding_mask=key_padding_mask)`
    (batch_first=False, add_bias_kv=True, attn_mask=None, eval mode -- this
    codebase's exact and only usage pattern, see `AttentionBlock.forward`).

    `calib`: an `AttentionCalibrators` (or `None` for a pure-fp32 sanity
    check with zero quantization anywhere, see `sanity_check_against_reference`).
    """
    E = mha.embed_dim
    H = mha.num_heads
    head_dim = E // H
    Lq, B, _ = q_in.shape
    Lk = k_in.shape[0]

    # ---- 1) pre-projection activation quantization (matches int8_w8a8_static:
    #         fake-quantize the raw latent/token tensors *before* Wq/Wk/Wv). ----
    if calib is not None:
        q_in_q = _maybe_observe_or_pass(calib.q_in, q_in)
        k_in_q = _maybe_observe_or_pass(calib.k_in, k_in, exclude_mask=_expand_kv_mask(key_padding_mask, k_in))
        v_in_q = _maybe_observe_or_pass(calib.v_in, v_in, exclude_mask=_expand_kv_mask(key_padding_mask, v_in))
    else:
        q_in_q, k_in_q, v_in_q = q_in, k_in, v_in

    Wq, Wk, Wv = mha.in_proj_weight.split(E, dim=0)
    if mha.in_proj_bias is not None:
        bq, bk, bv = mha.in_proj_bias.split(E, dim=0)
    else:
        bq = bk = bv = None

    # in-proj matmul: input already on the int8 grid (fake-quantized above),
    # weight already int8-per-channel-quantized in place by
    # `ModelQuantizer.quantize_weights()` -- this plain `F.linear` call is
    # therefore already numerically identical to a real INT8xINT8->INT32 GEMM
    # (see module docstring for the exactness argument).
    q = F.linear(q_in_q, Wq, bq)   # (Lq, B, E)
    k = F.linear(k_in_q, Wk, bk)   # (Lk, B, E)
    v = F.linear(v_in_q, Wv, bv)   # (Lk, B, E)

    if mha.bias_k is not None:
        k = torch.cat([k, mha.bias_k.expand(1, B, E)], dim=0)
        v = torch.cat([v, mha.bias_v.expand(1, B, E)], dim=0)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, key_padding_mask.new_zeros(B, 1)], dim=1)  # bias slot is never padded
        Lk += 1

    qh = split_heads(q, H)   # (B, H, Lq, Dh)
    kh = split_heads(k, H)   # (B, H, Lk, Dh)
    vh = split_heads(v, H)   # (B, H, Lk, Dh)

    kv_head_mask = None
    if key_padding_mask is not None:
        kv_head_mask = key_padding_mask.view(B, 1, Lk, 1)  # True = pad, broadcasts over (B,H,Lk,Dh)

    # ---- 2) Q.K^T ----
    use_qk_mac = calib is not None and stage in ('qk_mac', 'qkv_mac')
    if use_qk_mac:
        if calib.q_proj.calibrating:
            calib.q_proj.observe(qh)
            calib.k_proj.observe(kh, exclude_mask=kv_head_mask.expand_as(kh) if kv_head_mask is not None else None)
            raw_scores = torch.matmul(qh, kh.transpose(-2, -1))
        else:
            raw_scores = real_int8_matmul(qh, calib.q_proj.scale, kh, calib.k_proj.scale, transpose_b=True)
    else:
        raw_scores = torch.matmul(qh, kh.transpose(-2, -1))

    scores = raw_scores / math.sqrt(head_dim)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, Lk), float('-inf'))
    attn = torch.softmax(scores, dim=-1)

    # ---- 3) softmax(scores) @ V ----
    use_av_mac = calib is not None and stage == 'qkv_mac'
    if use_av_mac:
        if calib.v_proj.calibrating:
            calib.v_proj.observe(vh, exclude_mask=kv_head_mask.expand_as(vh) if kv_head_mask is not None else None)
            out = torch.matmul(attn, vh)
        else:
            attn_int = calib.attn.to_int(attn)
            v_int = calib.v_proj.to_int(vh)
            reduce_dim = attn_int.shape[-1]
            bound = calib.attn.qmax * calib.v_proj.qmax * reduce_dim
            assert bound < REAL_INT8_ACCUM_BOUND, (
                f"attn@V accumulator bound ({bound}) for Lk={reduce_dim} exceeds float32's "
                f"exact-integer range ({REAL_INT8_ACCUM_BOUND}).")
            out_int = torch.matmul(attn_int, v_int)
            out = out_int * (calib.attn.scale * calib.v_proj.scale)
    else:
        out = torch.matmul(attn, vh)   # (B, H, Lq, Dh)

    out = merge_heads(out)   # (Lq, B, E)

    # ---- 4) out_proj -- the gap `int8_w8a8_static` always had (see module
    #         docstring); quantized + a real int8 GEMM in every stage,
    #         including 'baseline'. ----
    if calib is not None:
        if calib.out_proj.calibrating:
            calib.out_proj.observe(out)
            result = F.linear(out, mha.out_proj.weight, mha.out_proj.bias)
        else:
            w = mha.out_proj.weight
            w_scale = quant_ops.compute_qparams(w.data, calib.out_proj.num_bits, per_channel=True, channel_dim=0)
            result = real_int8_linear(out, calib.out_proj.scale, w, w_scale, bias=mha.out_proj.bias)
    else:
        result = F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

    return result


def _expand_kv_mask(key_padding_mask, ref_tensor):
    """key_padding_mask: (B, Lk) bool, True=pad. ref_tensor: (Lk, B, E).
    Returns a mask broadcastable against ref_tensor, or None."""
    if key_padding_mask is None:
        return None
    B, Lk = key_padding_mask.shape
    return key_padding_mask.transpose(0, 1).unsqueeze(-1)  # (Lk, B, 1) broadcasts over (Lk, B, E)


# =============================================================================
# Monkey-patching every AttentionBlock instance
# =============================================================================
def patch_attention_blocks(model, stage, num_bits=8):
    """Replaces every `AttentionBlock.forward` in `model` (instance-level, not
    class-level) with a version that calls `attention_forward` instead of
    `self.attention(...)`. Returns `{block_name: AttentionCalibrators}`.

    `model.backbone.proc_memory_blocks[*].cross_attention` and
    `...latent_attentions[i]` are the only `AttentionBlock` instances in this
    architecture (see `models/EvT.py:TransformerBlock`)."""
    import types

    from models.EvT import AttentionBlock  # local import: relies on data_utils having put
                                            # EventTransformer/ on sys.path already.

    calibrators = {}
    for name, block in model.named_modules():
        if not isinstance(block, AttentionBlock):
            continue
        calib = AttentionCalibrators(stage, num_bits=num_bits)
        calibrators[name] = calib

        def make_forward(calib):
            def forward(self, x, z_input, mask=None, q_mask=None, **args):
                assert q_mask is None, "attn_mask is never used by this codebase; not implemented here."
                x_ln = self.layer_norm_x(x)
                z = self.layer_norm_1(z_input)

                z_att = attention_forward(z, x_ln, x_ln, self.attention,
                                           key_padding_mask=mask, stage=stage, calib=calib)

                z_att = z_att + z_input
                z = self.layer_norm_att(z_att)

                z = self.dropout(z)
                z = self.linear1(z)
                z = nn.GELU()(z)

                z = self.layer_norm_2(z)
                z = self.linear2(z)
                z = nn.GELU()(z)
                z = self.dropout(z)
                z = self.linear3(z)

                return z + z_att
            return forward

        block.forward = types.MethodType(make_forward(calib), block)

    return calibrators


def start_calibration(calibrators):
    for c in calibrators.values():
        c.start_calibration()


def finalize_calibration(calibrators):
    for c in calibrators.values():
        c.finalize_calibration()


def inject_calibrated_scales(calibrators, q_in_scales, k_in_scales, v_in_scales,
                              out_proj_scales, qk_scales=None, v_scale=None):
    """Populate every calibrator's `.scale` directly from previously-computed
    scalar values (e.g. loaded from a saved `attention_mac_scales.json`),
    instead of running a fresh calibration pass over the training set --
    used by `export_attention_mac_real_quantized.py` so re-verifying a
    packed checkpoint's accuracy doesn't require recalibrating from scratch.
    `q_in_scales`/`k_in_scales`/`v_in_scales`/`out_proj_scales`/`qk_scales`/
    `v_scale`: `{block_name: ...}` dicts with the exact shapes
    `run_attention_mac.py` saves (see its `run_one_version`).
    """
    for name, calib in calibrators.items():
        calib.q_in.calibrating = False
        calib.q_in.scale = torch.tensor(float(q_in_scales[name]))
        calib.k_in.calibrating = False
        calib.k_in.scale = torch.tensor(float(k_in_scales[name]))
        calib.v_in.calibrating = False
        calib.v_in.scale = torch.tensor(float(v_in_scales[name]))
        calib.out_proj.calibrating = False
        calib.out_proj.scale = torch.tensor(float(out_proj_scales[name]))
        if qk_scales is not None and calib.q_proj is not None:
            calib.q_proj.calibrating = False
            calib.q_proj.scale = torch.tensor(float(qk_scales[name]['q_scale']))
            calib.k_proj.calibrating = False
            calib.k_proj.scale = torch.tensor(float(qk_scales[name]['k_scale']))
        if v_scale is not None and calib.v_proj is not None:
            calib.v_proj.calibrating = False
            calib.v_proj.scale = torch.tensor(float(v_scale[name]))


# =============================================================================
# Trust check: with quantization fully disabled, does `attention_forward`
# reproduce PyTorch's own `nn.MultiheadAttention` bit-for-bit (up to fp32
# rounding)? Run once before trusting any of the quantized numbers above.
# =============================================================================
@torch.no_grad()
def sanity_check_against_reference(model, dataloader, device, num_batches=3, atol=1e-4):
    """Runs `num_batches` real batches through the model twice: once with the
    original, untouched `AttentionBlock`s (PyTorch's fused
    `nn.MultiheadAttention` kernel), once with every `AttentionBlock` patched
    to call `attention_forward(..., stage=None, calib=None)` (pure fp32, zero
    quantization). Returns the maximum absolute difference between the two
    models' final logits across all batches -- should be ~1e-6..1e-5 (fp32
    associativity/summation-order noise only), never anything close to the
    quantization-induced differences reported elsewhere.
    """
    import copy
    import types

    from models.EvT import AttentionBlock

    ref_model = copy.deepcopy(model).to(device).eval()
    test_model = copy.deepcopy(model).to(device).eval()

    for _, block in test_model.named_modules():
        if not isinstance(block, AttentionBlock):
            continue

        def make_forward():
            def forward(self, x, z_input, mask=None, q_mask=None, **args):
                assert q_mask is None
                x_ln = self.layer_norm_x(x)
                z = self.layer_norm_1(z_input)
                z_att = attention_forward(z, x_ln, x_ln, self.attention,
                                           key_padding_mask=mask, stage=None, calib=None)
                z_att = z_att + z_input
                z = self.layer_norm_att(z_att)
                z = self.dropout(z)
                z = self.linear1(z)
                z = nn.GELU()(z)
                z = self.layer_norm_2(z)
                z = self.linear2(z)
                z = nn.GELU()(z)
                z = self.dropout(z)
                z = self.linear3(z)
                return z + z_att
            return forward

        block.forward = types.MethodType(make_forward(), block)

    max_diff = 0.0
    seen = 0
    for polarity, pixels, _labels in dataloader:
        if polarity is None:
            continue
        polarity = polarity.to(device)
        pixels = pixels.to(device)
        _e1, logits_ref = ref_model(polarity, pixels)
        _e2, logits_test = test_model(polarity, pixels)
        max_diff = max(max_diff, (logits_ref - logits_test).abs().max().item())
        seen += 1
        if seen >= num_batches:
            break

    assert max_diff < atol, (
        f"attention_forward (stage=None) diverged from nn.MultiheadAttention by {max_diff} "
        f"(> atol={atol}) -- the reimplementation has a bug, do not trust the quantized results.")
    return max_diff
