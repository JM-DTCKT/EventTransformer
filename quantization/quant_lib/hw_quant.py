"""ZCU102 integer datapath for EvT -- see `../hw_flow.md` for the unit-by-unit
description and `../README.md` for how to run it.

The only numeric types left anywhere in the datapath are **INT8** (multiplier
operands), **INT32** (accumulators and folded biases) and **16-bit fixed point
Qm.n** (scales, non-linear I/O, LayerNorm affine params, the latent memory
table) -- plus **bfloat16** inside LayerNorm. There is no fp32 fallback: the
fake-quantization paths this module grew out of have been removed.

What that means concretely
--------------------------
*Bias* is folded into the accumulator: with `x = x_int * s_x` and
`W = W_int * s_w` (per output channel), `b_int = round(b / (s_x * s_w))` is an
INT32 addend and the layer output is `(x_int @ W_int^T + b_int) * (s_x * s_w)`
-- one integer add instead of an fp32 add after an fp32 multiply. Same rule for
`in_proj_bias`, `out_proj.bias` and MHA's `bias_k`/`bias_v` (which live in the
post-projection K/V domain and are appended as one extra K/V token).

*Every scale* is a 16-bit Qm.n code with a per-tensor shared exponent (per
layer for the per-output-channel weight scales: one int16 multiplier vector +
one shift, exactly the shape of a hardware rescale unit). See `fixed_point.py`.

*The positional encoding* is an INT8 table plus one Q-format scale, sharing the
consumer Linear's input scale (no K-split GEMM).

*Every tensor entering or leaving a non-linear unit* is snapped onto a grid
picked by a calibration pass: GELU in/out (pinned Q4.11), softmax in (pinned
Q6.9) / out (analytic Q1.14 -> uint8 step 1/127). The classifier head ends at
`argmax(acc*M)` -- no log_softmax, no output grid. `ReLU` is the exception --
only the sign matters, so its input is
requantized to INT8. LayerNorm runs in bfloat16, so its Qm.n sites are
observers only; its affine params and `memory_vertical` are still Q-format
int16 so no fp32 constant sneaks back in.

Everything runs through `build_payload()` -> `torch.save` -> `instantiate()`,
so the model that gets evaluated is always the one reconstructed from the saved
file, and `payload_size_bytes()` reports what that file actually costs.
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fixed_point as fx
# float32 represents every integer below 2**24 exactly, so an INT8xINT8->INT32
# accumulator modelled in fp32 is bit-exact as long as it stays under this.
REAL_INT8_ACCUM_BOUND = 2 ** 24


# --- head reshape (must match torch's own MultiheadAttention split exactly:
# embed_dim cut into `num_heads` contiguous chunks of `head_dim`) -------------
def split_heads(t, num_heads):
    """(L, B, E) -> (B, H, L, Dh)."""
    L, B, E = t.shape
    return t.reshape(L, B, num_heads, E // num_heads).permute(1, 2, 0, 3)


def merge_heads(t):
    """(B, H, L, Dh) -> (L, B, E), inverse of `split_heads`."""
    B, H, L, Dh = t.shape
    return t.permute(2, 0, 1, 3).reshape(L, B, H * Dh)

INT8_QMAX = 127
INT8_QMIN = -128
# the Linear whose input is `cat([event_projection output, positional encoding])`
POSENC_LINEAR = 'backbone.preproc_block_events.seq_init.0'

# Set to a callable(tag, a_int, b_int, acc, bias_int) to observe every integer
# MAC in the model; see `audit_inference_datapath.py`. None = no overhead.
MAC_PROBE = None


# Set to a dict to capture the attention block's internal GEMM I/O, which no
# module forward hook can see (`attention_forward_hw` calls `int8_linear`
# directly). `export_fpga.py --verify` uses this to check `in_proj`/`out_proj`.
ATTN_TAP = None


def _attn_tap(name, **kw):
    if ATTN_TAP is not None:
        ATTN_TAP.setdefault(name, {}).update(
            {k: v.detach().clone() for k, v in kw.items()})


def _probe(tag, a_int, b_int, acc, bias_int=None):
    if MAC_PROBE is not None:
        MAC_PROBE(tag, a_int, b_int, acc, bias_int)
INT32_MAX = 2 ** 31 - 1
# float32 stores integers exactly only below 2**24; every integer accumulator
# in this file is kept in float32 (fast matmul) so this bound is asserted.
EXACT_INT_BOUND = 2 ** 24


# =============================================================================
# Config -- frozen
# =============================================================================
class FinalConfig:
    """The one configuration this repo ships. There are no alternatives left:
    every choice below is the outcome recorded in `hw_flow.md`, and the fp32
    fallbacks that used to sit beside them are gone.

        weights           INT8, per-output-channel scale (one shared exponent)
        activations       INT8, per-tensor scale
        accumulators      INT32; bias folded to an INT32 accumulator addend
        scales            16-bit Qm.n codes (no fp32 scale anywhere)
        GELU              Q4.11 in and out -- one LUT for the whole network
        softmax           input Q6.9 -- one exp LUT; output Q1.14 -> uint8, step 1/127
        LayerNorm         bfloat16 datapath (input, mean, centered, var, rsqrt, affine)
        positional enc.   INT8 table, sharing the consumer's input scale (no K-split)
    """

    name = 'fpga_int8'
    gelu_frac_bits = 11        # Q4.11
    softmax_frac_bits = 9      # Q6.9
    # LayerNorm is bf16 only up to `rsqrt`; after normalisation the range is
    # bounded (|xhat| < 9) so a 16-bit Qm.n grid is ~16x finer than bf16 there.
    ln_xhat_frac_bits = 11     # Q4.11   (observed |xhat| <= 9.02)
    ln_out_frac_bits = 11      # Q4.11   (observed |out|  <= 9.45)
    ln_gamma_frac_bits = 14    # Q1.14   (|gamma| <= 1.26; multiplies xhat, keep it fine)
    ln_beta_frac_bits = 11     # Q4.11   (added to the output, so finer than out's LSB is wasted)
    FIELDS = ('gelu_frac_bits', 'softmax_frac_bits', 'ln_xhat_frac_bits',
              'ln_out_frac_bits', 'ln_gamma_frac_bits', 'ln_beta_frac_bits')

    def as_dict(self):
        return dict(name=self.name, **{f: getattr(self, f) for f in self.FIELDS})

    @classmethod
    def from_dict(cls, d):
        cfg = cls()
        assert d.get('name') == cls.name, (
            f"checkpoint was packed by config {d.get('name')!r}, not {cls.name!r} -- "
            f"re-run quantize.py")
        for f in cls.FIELDS:
            assert int(d[f]) == getattr(cls, f), f'{f} mismatch: {d[f]} != {getattr(cls, f)}'
        return cfg

    def __repr__(self):
        return (f'FinalConfig({self.name}: INT8 w/a, INT32 acc+bias, Qm.n scales, '
                f'GELU Q4.{self.gelu_frac_bits}, softmax in Q6.{self.softmax_frac_bits}, '
                f'LayerNorm bf16 -> Q4.{self.ln_out_frac_bits})')


# =============================================================================
# Requantization sites (in front of / behind every non-linear unit)
# =============================================================================
class RequantSite:
    """One requantization point. `kind='fx16'` -> 16-bit Qm.n grid whose
    fractional-bit count is chosen from the calibrated range; `kind='int8'` ->
    symmetric INT8 (used for ReLU inputs, where only the sign matters).

    `dynamic` (fx16 only) switches from a *static* per-site Q format to
    block floating point: the shift amount is derived at run time from the
    tensor's own magnitude (exponent-detect tree + barrel shifter in
    hardware, still a 16-bit mantissa). This matters for the sites on EvT's
    residual stream: `latent_vectors` is accumulated across time steps, so
    the same LayerNorm input spans several orders of magnitude between the
    first and the last step and one static format cannot cover both --
    `call_max_min`/`call_max_max` record that spread during calibration.
    """

    def __init__(self, name, kind='fx16', dynamic=False, total_bits=fx.TOTAL_BITS,
                 role=None):
        assert kind in ('fx16', 'int8')
        self.name = name
        self.kind = kind
        # which non-linear unit this site belongs to ('gelu', 'relu', 'softmax',
        # 'layernorm', ...) -- lets a config pin one unit's Q format
        self.role = role
        self.dynamic = dynamic
        self.total_bits = total_bits     # widen a site instead of making it dynamic
        self.force_off = False           # site exists but this config does not use it
        self.per_call = False            # ...or one static format per time step
        self.per_call_max = []           # max|x| seen at each call index of a forward
        self.per_call_frac_bits = None
        self.call_idx = 0
        self.max_abs = 0.0
        self.call_max_min = float('inf')     # smallest per-call max|x| seen
        self.call_max_max = 0.0              # largest per-call max|x| seen
        self.calibrating = False
        self.enabled = False
        self.frac_bits = None
        self.scale = None

    @torch.no_grad()
    def observe(self, x):
        t = x.detach()
        t = t[torch.isfinite(t)]
        if not t.numel():
            return
        m = float(t.abs().max())
        self.max_abs = max(self.max_abs, m)
        if m > 0:
            self.call_max_min = min(self.call_max_min, m)
            self.call_max_max = max(self.call_max_max, m)
        while len(self.per_call_max) <= self.call_idx:
            self.per_call_max.append(0.0)
        self.per_call_max[self.call_idx] = max(self.per_call_max[self.call_idx], m)

    def finalize(self):
        self.calibrating = False
        if self.kind == 'fx16':
            self.frac_bits = fx.choose_frac_bits(self.max_abs, self.total_bits)
        else:
            self.scale = max(self.max_abs, 1e-8) / INT8_QMAX

    def set_width(self, total_bits, headroom_bits=0):
        """Re-pick the Q format(s) for a different register width.

        `headroom_bits` drops that many fractional bits, i.e. widens the
        representable range by 2**headroom_bits. Calibration only sees the
        training set; the test set can exceed it (see
        `audit_inference_datapath.py`), and a fixed-point format saturates
        rather than degrading gracefully, so a guard band is cheap insurance."""
        self.total_bits = total_bits
        if self.kind == 'fx16':
            self.frac_bits = fx.choose_frac_bits(self.max_abs, total_bits) - headroom_bits
            self.per_call_frac_bits = [fx.choose_frac_bits(m, total_bits) - headroom_bits
                                       for m in self.per_call_max] or None

    @property
    def call_spread(self):
        """max|x| of the loudest call / max|x| of the quietest call."""
        if self.call_max_min in (0.0, float('inf')):
            return float('nan')
        return self.call_max_max / self.call_max_min

    def __call__(self, x):
        if self.calibrating:
            self.observe(x)
            self.call_idx += 1
            return x
        if not self.enabled or self.force_off:
            return x
        if self.kind == 'int8':
            q = torch.clamp(torch.round(x / self.scale), INT8_QMIN, INT8_QMAX)
            return q * self.scale
        if self.per_call and self.per_call_frac_bits:
            # one static Q format per time step: a small ROM of shift amounts
            # indexed by the time-step counter -- no data-dependent logic
            n = self.per_call_frac_bits[min(self.call_idx, len(self.per_call_frac_bits) - 1)]
            self.call_idx += 1
            return fx.fx_round(x, n, self.total_bits)
        if not self.dynamic:
            return fx.fx_round(x, self.frac_bits, self.total_bits)
        # block floating point: shift = floor(log2(qmax / max|x|)), per call
        qmax = 2 ** (self.total_bits - 1) - 1
        m = x.detach().abs().amax().clamp(min=1e-12)
        shift = torch.pow(2.0, torch.floor(torch.log2(qmax / m)))
        return torch.clamp(torch.round(x * shift), -qmax - 1, qmax) / shift

    def to_dict(self):
        return dict(kind=self.kind, max_abs=self.max_abs, dynamic=self.dynamic,
                    role=self.role,
                    total_bits=self.total_bits, per_call_max=list(self.per_call_max),
                    call_max_min=self.call_max_min, call_max_max=self.call_max_max,
                    frac_bits=self.frac_bits, scale=self.scale)

    @classmethod
    def from_dict(cls, name, d):
        s = cls(name, d['kind'], d.get('dynamic', False), d.get('total_bits', fx.TOTAL_BITS),
                role=d.get('role'))
        s.max_abs = d['max_abs']
        s.per_call_max = list(d.get('per_call_max', []))
        s.call_max_min = d.get('call_max_min', float('inf'))
        s.call_max_max = d.get('call_max_max', 0.0)
        s.frac_bits = d['frac_bits']
        s.scale = d['scale']
        return s


class SiteBook(dict):
    def site(self, name, kind='fx16', role=None):
        if name not in self:
            self[name] = RequantSite(name, kind, role=role)
        elif role is not None:
            self[name].role = role
        return self[name]

    def start_calibration(self):
        for s in self.values():
            s.calibrating, s.enabled, s.max_abs = True, False, 0.0
            s.per_call_max = []

    def finalize_calibration(self):
        for s in self.values():
            s.finalize()

    def enable(self, on=True):
        for s in self.values():
            s.calibrating = False
            s.enabled = on

    def reset_call_counters(self):
        """Called once per model forward, so a site's call index equals the
        time step it is being invoked for."""
        for s in self.values():
            s.call_idx = 0

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.items()}

    @classmethod
    def from_dict(cls, d):
        book = cls()
        for k, v in d.items():
            book[k] = RequantSite.from_dict(k, v)
        return book


# =============================================================================
# Integer GEMM primitives
# =============================================================================
def compute_weight_scale(w, num_bits=8):
    """Symmetric per-output-channel weight scale: `max|W[c,:]| / qmax`.

    One scale per row of a (E_out, E_in) weight, i.e. per output neuron. The
    exponent is shared across the row vector when it is packed (`_pack_scale`),
    so the datapath carries one int16 code per channel plus one shift.
    """
    qmax = 2 ** (num_bits - 1) - 1
    max_abs = w.detach().abs().amax(dim=1, keepdim=True)
    return max_abs.clamp(min=1e-8) / qmax


def _to_int8(x, scale):
    return torch.clamp(torch.round(x / scale), INT8_QMIN, INT8_QMAX)


def int8_linear(x, x_scale, w_int, w_scale, b_int=None):
    """`(x_int @ w_int^T + b_int) * (x_scale * w_scale)`.

    x: (..., E_in) fp32 tensor of a value that is representable on the INT8
       grid defined by `x_scale` (scalar).
    w_int: (E_out, E_in) fp32 tensor holding exact INT8 integers.
    w_scale: (E_out,) fp32 per-output-channel scale.
    b_int: (E_out,) fp32 tensor holding exact INT32 integers, already divided
       by `x_scale * w_scale` at build time.
    """
    reduce_dim = x.shape[-1]
    bound = INT8_QMAX * INT8_QMAX * reduce_dim
    assert bound < EXACT_INT_BOUND, (
        f"INT8 accumulator bound {bound} for reduction {reduce_dim} exceeds float32's "
        f"exact-integer range {EXACT_INT_BOUND}")
    x_int = _to_int8(x, x_scale)
    acc = torch.matmul(x_int, w_int.transpose(-2, -1))
    if b_int is not None:
        acc = acc + b_int
    _probe(f'linear_K{reduce_dim}', x_int, w_int, acc, b_int)
    return acc * (x_scale * w_scale)


def fold_bias_to_int32(bias, x_scale, w_scale):
    """b_int = round(b / (x_scale * w_scale)), saturated to INT32."""
    step = x_scale * w_scale
    b_int = torch.round(bias.detach() / step)
    return torch.clamp(b_int, -INT32_MAX - 1, INT32_MAX)


# =============================================================================
# Payload construction (fp32 checkpoint + calibrated scales -> packed dict)
# =============================================================================
def _block_name_of_attention(module_name):
    """'...cross_attention.attention.out_proj' -> '...cross_attention'."""
    for suffix in ('.attention.out_proj', '.attention'):
        if module_name.endswith(suffix):
            return module_name[: -len(suffix)]
    return None


def _pack_scale(t):
    """Per-tensor shared-exponent packing of a scale tensor: int16 codes + one
    shift. No fp32 scale survives anywhere in the datapath."""
    codes, frac_bits = fx.pack_fx16(t.detach().float().reshape(-1))
    return dict(codes=codes, frac_bits=frac_bits)


def _unpack_scale(entry):
    return fx.unpack_fx16(entry['codes'], entry['frac_bits'])


def _pack_param_fx16(t, frac_bits=None):
    codes, frac_bits = fx.pack_fx16(t, frac_bits)
    return dict(codes=codes, frac_bits=frac_bits, shape=tuple(t.shape))


def _unpack_param_fx16(entry):
    return fx.unpack_fx16(entry['codes'], entry['frac_bits']).reshape(entry['shape'])


def build_payload(model, scales, cfg, site_book=None, weight_bits=8):
    """Pack a *fresh fp32* `model` into a fully-integer payload.

    `scales`: what `calibration.calibrate_activation_scales` returned
        (pre-projection Q/K/V scales, post-projection Q/K/V scales, out_proj
        scales, per-Linear activation max-abs).
    `site_book`: calibrated non-linear requantization sites (see
        `calibrate_sites`); may be None for the calibration run itself.
    """
    qmax = 2 ** (weight_bits - 1) - 1
    act_max = scales['linear_act_scales']
    layers = {}          # sd_key(weight) -> packed weight/bias/scale entry
    fx_params = {}       # sd_key -> fx16-packed parameter (LN affine, memory table)
    fp32_other = {}      # everything that stays fp32 in this config
    diag = dict(max_abs_bias_int=0.0, act_scale_missing=[])

    covered = set()
    sd = model.state_dict()

    # The [event-projection features | positional encoding] concatenation shares
    # ONE input scale across both halves (no K-split GEMM), so it needs no
    # special case here -- see hw_flow.md 9.1 [2].
    def _act_scale_for(name, module):
        if name in act_max:
            return float(act_max[name]) / INT8_QMAX
        blk = _block_name_of_attention(name)
        if blk is not None and blk in scales['out_proj_scales']:
            return float(scales['out_proj_scales'][blk])
        diag['act_scale_missing'].append(name)
        return None

    with torch.no_grad():
        # ---- Linear layers (incl. every MHA out_proj) ----
        for name, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            w = m.weight.data
            w_scale = compute_weight_scale(w, weight_bits)  # (E_out,1)
            w_int = torch.clamp(torch.round(w / w_scale), -qmax - 1, qmax)
            x_scale = _act_scale_for(name, m)
            entry = dict(kind='linear', shape=tuple(w.shape),
                         q=w_int.to(torch.int8),
                         scale=_pack_scale(w_scale.reshape(-1)),
                         x_scale=_pack_scale(torch.tensor([x_scale if x_scale else 1.0])),
                         x_scale_valid=x_scale is not None)
            covered.add(f'{name}.weight')
            if m.bias is not None:
                entry['bias'] = _pack_bias(m.bias.data, x_scale,
                                           _unpack_scale(entry['scale']), diag)
                covered.add(f'{name}.bias')
            layers[f'{name}.weight'] = entry

        # ---- MultiheadAttention in_proj (packed Wq|Wk|Wv) ----
        for name, m in model.named_modules():
            if not isinstance(m, nn.MultiheadAttention):
                continue
            blk = _block_name_of_attention(name)
            E = m.embed_dim
            w = m.in_proj_weight.data
            w_scale = compute_weight_scale(w, weight_bits)
            w_int = torch.clamp(torch.round(w / w_scale), -qmax - 1, qmax)
            # per-row activation scale: rows [0:E)=Q, [E:2E)=K, [2E:3E)=V
            x_scales = torch.tensor(
                [float(scales['q_in_scales'][blk])] * E
                + [float(scales['k_in_scales'][blk])] * E
                + [float(scales['v_in_scales'][blk])] * E)
            entry = dict(kind='in_proj', shape=tuple(w.shape), embed_dim=E,
                         q=w_int.to(torch.int8),
                         scale=_pack_scale(w_scale.reshape(-1)))
            covered.add(f'{name}.in_proj_weight')
            if m.in_proj_bias is not None:
                entry['bias'] = _pack_bias_vec(m.in_proj_bias.data, x_scales,
                                               _unpack_scale(entry['scale']), diag)
                covered.add(f'{name}.in_proj_bias')
            # bias_k / bias_v live in the *post-projection* K/V domain: they are
            # appended as an extra K/V token, so fold them with the same
            # accumulator step as the K/V rows of in_proj.
            if m.bias_k is not None:
                wk_scale = _unpack_scale(entry['scale'])[E:2 * E]
                wv_scale = _unpack_scale(entry['scale'])[2 * E:3 * E]
                entry['bias_k'] = _pack_bias_vec(m.bias_k.data.reshape(-1),
                                                 x_scales[E:2 * E], wk_scale, diag)
                entry['bias_v'] = _pack_bias_vec(m.bias_v.data.reshape(-1),
                                                 x_scales[2 * E:3 * E], wv_scale, diag)
                covered.add(f'{name}.bias_k')
                covered.add(f'{name}.bias_v')
            layers[f'{name}.in_proj_weight'] = entry

        # ---- positional encoding: INT8 table + one scale ----
        pos_key = 'backbone.pos_encoding'
        pos_enc = None
        if pos_key in sd:
            p = sd[pos_key]
            p_scale = fx.fx_round_scalar(float(p.abs().max()) / INT8_QMAX)[0]
            pos_enc = dict(q=fx.pack_int8(p, p_scale),
                           scale=_pack_scale(torch.tensor([p_scale])),
                           shape=tuple(p.shape))
            covered.add(pos_key)

        # ---- LayerNorm affine + latent memory table: fixed point when the
        #      non-linear datapath is fixed point ----
        for name, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for attr, n in (('weight', cfg.ln_gamma_frac_bits),
                                ('bias', cfg.ln_beta_frac_bits)):
                    t = getattr(m, attr)
                    if t is None:
                        continue
                    # gamma multiplies xhat, beta is added to the output -- so
                    # they get different (pinned) grids, see FinalConfig
                    fx_params[f'{name}.{attr}'] = _pack_param_fx16(t.data, n)
                    covered.add(f'{name}.{attr}')
        mem_key = 'backbone.memory_vertical'
        if mem_key in sd:
            fx_params[mem_key] = _pack_param_fx16(sd[mem_key])
            covered.add(mem_key)

        # ---- anything left (training-only buffers) ----
        for k, v in sd.items():
            if k not in covered:
                fp32_other[k] = v.clone()

    payload = dict(
        format='evt_hw_quant_v1',
        config=cfg.as_dict(),
        weight_bits=weight_bits,
        layers=layers,
        fx_params=fx_params,
        pos_encoding=pos_enc,
        fp32_other=fp32_other,
        attention_scales=_pack_attention_scales(scales),
        sites=_pack_sites(site_book, cfg),
        diagnostics=diag,
    )
    return payload


def _pack_sites(site_book, cfg):
    """Serialise the requantization sites. An fx16 site stores only a shift
    amount; an INT8 site (a ReLU input) stores a step, which is a datapath
    multiplier like any other scale and is therefore snapped onto a Q format
    when the design keeps its multipliers in fixed point."""
    if site_book is None:
        return None
    out = site_book.to_dict()
    for entry in out.values():
        if entry['kind'] == 'int8' and entry['scale'] is not None:
            entry['scale'] = fx.fx_round_scalar(entry['scale'])[0]
    return out


def _pack_bias(bias, x_scale, w_scale, diag):
    """Per-output-channel bias of a Linear, folded onto the INT32 accumulator."""
    assert x_scale is not None, 'bias folding needs the input activation scale'
    return _pack_bias_vec(bias, x_scale, w_scale, diag)


def _pack_bias_vec(bias, x_scales, w_scale, diag):
    """Same, with a per-row activation scale vector (MHA in_proj)."""
    b_int = fold_bias_to_int32(bias, x_scales, w_scale)
    diag['max_abs_bias_int'] = max(diag['max_abs_bias_int'], float(b_int.abs().max()))
    return dict(q=b_int.to(torch.int32))


def _pack_attention_scales(scales):
    """The handful of per-attention-block activation scales (pre-projection
    Q/K/V, post-projection Q/K/V, out_proj input) -- fx16 or fp32."""
    out = {}
    for key in ('q_in_scales', 'k_in_scales', 'v_in_scales', 'out_proj_scales', 'v_scale'):
        src = scales.get(key)
        if src is None:
            continue
        out[key] = {b: _pack_scale(torch.tensor([float(v)])) for b, v in src.items()}
    if scales.get('qk_scales'):
        out['qk_scales'] = {
            b: dict(q_scale=_pack_scale(torch.tensor([float(v['q_scale'])])),
                    k_scale=_pack_scale(torch.tensor([float(v['k_scale'])])))
            for b, v in scales['qk_scales'].items()}
    # softmax output scale is analytic (1/127), no calibration needed
    out['softmax_scale'] = _pack_scale(torch.tensor([1.0 / INT8_QMAX]))
    return out


# =============================================================================
# Size accounting
# =============================================================================
def payload_size_bytes(payload, include_non_inference=False):
    """Bytes the payload's tensors genuinely occupy, broken down by group.
    Scalar metadata (a Q format's shift amount) is charged 1 byte each."""
    groups = dict(weights_int8=0, weight_scales=0, biases=0, pos_encoding=0,
                  fx_params=0, act_scales=0, other=0)

    def _t(x):
        return x.numel() * x.element_size()

    def _scale_bytes(entry):
        return _t(entry['codes']) + 1        # int16 codes + one shared shift

    for entry in payload['layers'].values():
        groups['weights_int8'] += _t(entry['q'])
        groups['weight_scales'] += _scale_bytes(entry['scale'])
        for sk in ('x_scale',):
            if sk in entry:
                groups['act_scales'] += _scale_bytes(entry[sk])
        for bkey in ('bias', 'bias_k', 'bias_v'):
            b = entry.get(bkey)
            if b is None:
                continue
            groups['biases'] += _t(b['q'])

    pe = payload.get('pos_encoding')
    if pe is not None:
        groups['pos_encoding'] += _t(pe['q']) + _scale_bytes(pe['scale'])

    for entry in payload['fx_params'].values():
        groups['fx_params'] += _t(entry['codes']) + 1

    def _walk_scales(d):
        total = 0
        for v in d.values():
            if isinstance(v, dict) and 'mode' in v:
                total += _scale_bytes(v)
            elif isinstance(v, dict):
                total += _walk_scales(v)
        return total

    groups['act_scales'] += _walk_scales(payload['attention_scales'])

    if payload.get('sites'):
        # one shift amount (fx16) or one int16 scale (int8 sites) per site
        for s in payload['sites'].values():
            groups['other'] += 1 if s['kind'] == 'fx16' else 2

    if include_non_inference:
        for v in payload['fp32_other'].values():
            groups['other'] += _t(v)

    groups['total'] = sum(v for k, v in groups.items() if k != 'total')
    return groups


# =============================================================================
# Instantiate a runnable model from a payload
# =============================================================================
class HWRuntime:
    """Holds the unpacked integer tensors + requantization sites for a model
    whose forwards have been patched by `instantiate`."""

    def __init__(self, payload, device):
        self.cfg = FinalConfig.from_dict(payload['config'])
        self.device = device
        self.sites = SiteBook.from_dict(payload['sites']) if payload.get('sites') else SiteBook()
        self.attention_scales = payload['attention_scales']
        self.softmax_scale = float(_unpack_scale(payload['attention_scales']['softmax_scale'])[0])

    def attn_scale(self, key, block, sub=None):
        d = self.attention_scales[key][block]
        if sub is not None:
            d = d[sub]
        return float(_unpack_scale(d)[0])

    def const(self, value):
        """A hard-wired datapath constant (e.g. 1/sqrt(head_dim)), snapped onto
        a Q format when the design stores its multipliers as fixed point."""
        return fx.fx_round_scalar(value)[0]


@torch.no_grad()
def instantiate(payload, model, device):
    """Reconstruct a runnable model *entirely from `payload`* (the fp32
    checkpoint is never consulted again -- `model` only supplies the
    architecture) and patch every forward onto the integer datapath.

    Returns (model, runtime)."""
    rt = HWRuntime(payload, device)
    cfg = rt.cfg
    sd = model.state_dict()

    # ---------- 1) materialise parameters from the payload ----------
    layer_info = {}
    for wkey, entry in payload['layers'].items():
        w_scale = _unpack_scale(entry['scale'])                       # (E_out,)
        w_int = entry['q'].to(torch.float32).reshape(entry['shape'])
        sd[wkey] = w_int * w_scale.reshape(-1, 1)
        info = dict(w_int=w_int, w_scale=w_scale, kind=entry['kind'])
        base = wkey.rsplit('.', 1)[0]

        if entry['kind'] == 'linear':
            info['x_scale'] = float(_unpack_scale(entry['x_scale'])[0]) if entry['x_scale_valid'] else None
            # accumulator step of every output channel: s_x * s_w
            steps = dict(bias=(info['x_scale'] or 1.0) * w_scale)
            sd_names = dict(bias=f'{base}.bias')
        else:
            E = entry['embed_dim']
            info['embed_dim'] = E
            blk = base[: -len('.attention')]
            x_rows = torch.cat([
                torch.full((E,), rt.attn_scale('q_in_scales', blk)),
                torch.full((E,), rt.attn_scale('k_in_scales', blk)),
                torch.full((E,), rt.attn_scale('v_in_scales', blk))])
            steps = dict(bias=x_rows * w_scale,
                         bias_k=x_rows[E:2 * E] * w_scale[E:2 * E],
                         bias_v=x_rows[2 * E:3 * E] * w_scale[2 * E:3 * E])
            sd_names = dict(bias=f'{base}.in_proj_bias',
                            bias_k=f'{base}.bias_k', bias_v=f'{base}.bias_v')

        for bkey in ('bias', 'bias_k', 'bias_v'):
            b = entry.get(bkey)
            if b is None:
                continue
            info[f'{bkey}_int'] = b['q'].to(torch.float32)
            effective = info[f'{bkey}_int'] * steps[bkey]
            # write the *effective* fp32 value back into the state dict so the
            # saved checkpoint stays a self-contained, loadable model
            sd[sd_names[bkey]] = effective.reshape(sd[sd_names[bkey]].shape)
        layer_info[wkey] = info

    for k, entry in payload['fx_params'].items():
        sd[k] = _unpack_param_fx16(entry).reshape(sd[k].shape)

    pe = payload.get('pos_encoding')
    if pe is not None:
        pscale = float(_unpack_scale(pe['scale'])[0])
        sd['backbone.pos_encoding'] = (pe['q'].to(torch.float32) * pscale).reshape(pe['shape'])

    for k, v in payload['fp32_other'].items():
        sd[k] = v

    model.load_state_dict(sd)
    model.to(device).eval()

    # move the integer tensors to the device
    for info in layer_info.values():
        info['w_int'] = info['w_int'].to(device)
        info['w_scale'] = info['w_scale'].to(device)
        for key in ('bias_int', 'bias_k_int', 'bias_v_int'):
            if info.get(key) is not None:
                info[key] = info[key].to(device)

    # ---------- 2) patch every forward ----------
    _patch_linears(model, layer_info, rt)
    _patch_layernorms(model, rt)
    _patch_activations(model, rt)
    _patch_attention_blocks(model, layer_info, rt)
    _patch_compressor_and_clf(model, rt)
    # a site's call index must mean "which time step", so reset once per forward
    if getattr(model, '_hw_reset_handle', None) is not None:
        model._hw_reset_handle.remove()
    model._hw_reset_handle = model.register_forward_pre_hook(
        lambda _m, _args: rt.sites.reset_call_counters())

    # ---------- 3) arm the non-linear requantizers ----------
    # the INT8 requant step in front of each ReLU is a datapath constant too --
    # it must be a Q-format multiplier, not an fp32 one
    for site in rt.sites.values():
        if site.kind == 'int8' and site.scale is not None:
            site.scale = fx.fx_round_scalar(site.scale)[0]
    for key, site in rt.sites.items():
        site.set_width(fx.TOTAL_BITS)
        if site.role == 'gelu':
            # one LUT for the whole network: same Q format at every GELU
            site.frac_bits = cfg.gelu_frac_bits
        elif site.role == 'softmax' and key.endswith('.in'):
            # one exp LUT for the whole network: same input Q format at every
            # softmax (the output is analytic and already shared)
            site.frac_bits = cfg.softmax_frac_bits
        elif '.layer_norm' in key and key.endswith('.hat'):
            site.frac_bits = cfg.ln_xhat_frac_bits
        elif '.layer_norm' in key and key.endswith('.out'):
            site.frac_bits = cfg.ln_out_frac_bits
    # LayerNorm runs in bf16, so its Qm.n sites are observers only
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m._ln_bf16 = True
    # A payload built with `site_book=None` is the calibration vehicle: the
    # sites exist but have no format yet, so they stay in observe-and-pass-
    # through mode. Any other payload must have every site calibrated.
    armed = payload.get('sites') is not None
    if armed:
        missing = [k for k, s in rt.sites.items()
                   if (s.frac_bits is None if s.kind == 'fx16' else s.scale is None)]
        assert not missing, (
            f"{len(missing)} requantization site(s) have no calibrated Q format "
            f"(e.g. {missing[:3]}) -- run `calibrate_sites` first.")
    rt.sites.enable(armed)
    return model, rt


# =============================================================================
# Patching
# =============================================================================
def _patch_linears(model, layer_info, rt):
    """Every standalone nn.Linear -> explicit INT8 GEMM + (INT32|fp32) bias.
    `out_proj` inside an MHA is skipped: the patched attention forward calls
    it directly (its module forward is never invoked by this codebase)."""
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        wkey = f'{name}.weight'
        info = layer_info.get(wkey)
        if info is None or _block_name_of_attention(name) is not None:
            continue
        if info['x_scale'] is None:      # never calibrated -> keep fp32 (should not happen)
            continue
        m._hw = info

        def forward(self, x):
            i = self._hw
            return int8_linear(x, i['x_scale'], i['w_int'], i['w_scale'],
                               b_int=i.get('bias_int'))

        m.forward = types.MethodType(forward, m)


def _bf16(t):
    """Round to bfloat16 precision, keep an fp32 container (so the rest of the
    graph is unchanged). bf16 = 1 sign + 8 exponent + 7 stored mantissa bits."""
    return t.to(torch.bfloat16).to(t.dtype)


def _patch_layernorms(model, rt):
    """LayerNorm as a fixed-point unit: input, the normalized intermediate and
    the output are each snapped onto their own Q format.

    The four Q grids are replaced by bfloat16 rounding at the same four points
    once the sites are armed. During calibration the Qm.n path runs instead, so
    the site statistics (and the audit report) still describe the ranges."""
    for name, m in model.named_modules():
        if not isinstance(m, nn.LayerNorm):
            continue
        m._sites = (rt.sites.site(f'{name}.in'), rt.sites.site(f'{name}.hat'),
                    rt.sites.site(f'{name}.out'), rt.sites.site(f'{name}.centered'))
        m._ln_bf16 = False

        def forward(self, x):
            s_in, s_hat, s_out, s_ctr = self._sites
            if self._ln_bf16 and s_in.enabled:
                # --- bf16 front end: the input's dynamic range spans ~5 decades
                # (max|x| up to 2.4e5), which is what bf16's 8-bit exponent is
                # for. Mean/variance reduce in fp32 and round to bf16 on output.
                x = _bf16(x)
                mu = _bf16(x.mean(dim=-1, keepdim=True))
                centered = _bf16(x - mu)
                var = _bf16(centered.pow(2).mean(dim=-1, keepdim=True))
                # --- Qm.n back end: after normalisation |xhat| < 9, so bf16's
                # exponent is wasted and its 8-bit mantissa is the binding
                # limit. Q4.11 is ~16x finer for the same 16 bits, and it hands
                # the next GEMM a Qm.n code -- the same integer requantizer the
                # rest of the datapath uses.
                xhat = s_hat(_bf16(centered * _bf16(torch.rsqrt(var + self.eps))))
                if self.weight is None:
                    return xhat
                return s_out(xhat * self.weight + self.bias)
            x = s_in(x)
            mu = x.mean(dim=-1, keepdim=True)
            # The variance accumulator is the widest thing in a fixed-point
            # LayerNorm: squaring a 32-bit centered value needs a 64-bit
            # product. `ln_centered_bits` narrows the centered value first, so
            # the squarer stays 16x16->32 and the accumulator ~40 bits.
            centered = s_ctr(x - mu)
            var = centered.pow(2).mean(dim=-1, keepdim=True)
            xhat = s_hat(centered * torch.rsqrt(var + self.eps))
            y = xhat * self.weight + self.bias if self.weight is not None else xhat
            return s_out(y)

        m.forward = types.MethodType(forward, m)


def _patch_activations(model, rt):
    """nn.GELU / nn.ReLU module instances inside the MLP blocks."""
    for name, m in model.named_modules():
        if isinstance(m, nn.GELU):
            m._sites = (rt.sites.site(f'{name}.in', role='gelu'),
                        rt.sites.site(f'{name}.out', role='gelu'))

            def gelu_forward(self, x):
                s_in, s_out = self._sites
                return s_out(F.gelu(s_in(x)))

            m.forward = types.MethodType(gelu_forward, m)
        elif isinstance(m, nn.ReLU):
            m._sites = (rt.sites.site(f'{name}.in', kind='int8'),)

            def relu_forward(self, x):
                return F.relu(self._sites[0](x))

            m.forward = types.MethodType(relu_forward, m)


def _patch_compressor_and_clf(model, rt):
    """`LatentEmbsCompressor` and `CLFBlock` call `F.relu` / `F.log_softmax`
    inline, so their forwards are replaced rather than the module patched."""
    from models.EvT import CLFBlock, LatentEmbsCompressor

    for name, m in model.named_modules():
        if isinstance(m, LatentEmbsCompressor):
            m._sites = (rt.sites.site(f'{name}.relu.in', kind='int8'),)

            def comp_forward(self, z):
                if self.embs_norm:
                    z = self.layer_norm(z)
                z = self.linear1(z)
                z = F.relu(self._sites[0](z))
                assert self.clf_mode == 'gap'
                return z.mean(dim=0)

            m.forward = types.MethodType(comp_forward, m)
        elif isinstance(m, CLFBlock):
            m._sites = (rt.sites.site(f'{name}.relu.in', kind='int8'),)

            def clf_forward(self, z):
                z = F.relu(self._sites[0](self.linear_1(z)))
                # No log_softmax and no output requantization: inference only
                # takes the argmax, and log_softmax is monotonic, so
                # argmax(log_softmax(z)) == argmax(z). The hardware compares
                # `acc[c] * M[c]` directly -- no shift, no rounding, no
                # saturation. `M[c]` is still needed because the per-class
                # weight scale s_w[c] would otherwise reorder the classes.
                return self.linear_2(z)

            m.forward = types.MethodType(clf_forward, m)


# =============================================================================
# Attention (qkv_mac stage) on the integer datapath
# =============================================================================
def _patch_attention_blocks(model, layer_info, rt):
    from models.EvT import AttentionBlock

    for name, block in model.named_modules():
        if not isinstance(block, AttentionBlock):
            continue
        mha = block.attention
        ip = layer_info[f'{name}.attention.in_proj_weight']
        op = layer_info[f'{name}.attention.out_proj.weight']
        ctx = dict(
            name=name, in_proj=ip, out_proj=op,
            q_in=rt.attn_scale('q_in_scales', name),
            k_in=rt.attn_scale('k_in_scales', name),
            v_in=rt.attn_scale('v_in_scales', name),
            out_proj_in=rt.attn_scale('out_proj_scales', name),
            q_proj=rt.attn_scale('qk_scales', name, 'q_scale'),
            k_proj=rt.attn_scale('qk_scales', name, 'k_scale'),
            v_proj=rt.attn_scale('v_scale', name),
            softmax=rt.softmax_scale,
            # 1/sqrt(head_dim) is a datapath constant like any other scale
            inv_sqrt_d=rt.const(1.0 / math.sqrt(mha.embed_dim // mha.num_heads)),
            site_scores=rt.sites.site(f'{name}.softmax.in', role='softmax'),
            site_attn=rt.sites.site(f'{name}.softmax.out', role='softmax'),
            gelu1=(rt.sites.site(f'{name}.gelu1.in', role='gelu'),
                   rt.sites.site(f'{name}.gelu1.out', role='gelu')),
            gelu2=(rt.sites.site(f'{name}.gelu2.in', role='gelu'),
                   rt.sites.site(f'{name}.gelu2.out', role='gelu')),
        )
        block._hw_ctx = ctx

        def forward(self, x, z_input, mask=None, q_mask=None, **args):
            assert q_mask is None, "attn_mask is never used by this codebase"
            ctx = self._hw_ctx
            x_ln = self.layer_norm_x(x)
            z = self.layer_norm_1(z_input)

            z_att = attention_forward_hw(z, x_ln, x_ln, self.attention, mask, ctx)

            z_att = z_att + z_input
            z = self.layer_norm_att(z_att)

            z = self.dropout(z)
            z = self.linear1(z)
            s_in, s_out = ctx['gelu1']
            z = s_out(F.gelu(s_in(z)))

            z = self.layer_norm_2(z)
            z = self.linear2(z)
            s_in, s_out = ctx['gelu2']
            z = s_out(F.gelu(s_in(z)))
            z = self.dropout(z)
            z = self.linear3(z)
            return z + z_att

        block.forward = types.MethodType(forward, block)


def attention_forward_hw(q_in, k_in, v_in, mha, key_padding_mask, ctx):
    """`nn.MultiheadAttention(q_in, k_in, v_in, key_padding_mask=...)` with
    every matmul an INT8xINT8->INT32 GEMM, every bias an INT32 accumulator
    addend and softmax fed from / feeding a fixed-point grid."""
    E = mha.embed_dim
    H = mha.num_heads
    head_dim = E // H
    Lq, B, _ = q_in.shape

    ip = ctx['in_proj']
    w_int, w_scale = ip['w_int'], ip['w_scale']
    b_int = ip.get('bias_int')

    def _proj(x, x_scale, lo, hi):
        return int8_linear(x, x_scale, w_int[lo:hi], w_scale[lo:hi],
                           b_int=None if b_int is None else b_int[lo:hi])

    q = _proj(q_in, ctx['q_in'], 0, E)
    k = _proj(k_in, ctx['k_in'], E, 2 * E)
    v = _proj(v_in, ctx['v_in'], 2 * E, 3 * E)

    _attn_tap(ctx['name'], q_in=q_in, k_in=k_in, v_in=v_in, q=q, k=k, v=v)

    if mha.bias_k is not None:
        # bias_k/bias_v are stored in the same post-projection domain (INT32
        # addends on the K/V accumulator); mha.bias_k already holds their
        # effective value, reconstructed by `instantiate`.
        k = torch.cat([k, mha.bias_k.expand(1, B, E)], dim=0)
        v = torch.cat([v, mha.bias_v.expand(1, B, E)], dim=0)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, key_padding_mask.new_zeros(B, 1)], dim=1)

    qh, kh, vh = split_heads(q, H), split_heads(k, H), split_heads(v, H)
    Lk = kh.shape[2]

    # ---- Q.K^T : real INT8 GEMM ----
    q_int = _to_int8(qh, ctx['q_proj'])
    k_int = _to_int8(kh, ctx['k_proj'])
    bound = INT8_QMAX * INT8_QMAX * head_dim
    assert bound < REAL_INT8_ACCUM_BOUND
    qk_acc = torch.matmul(q_int, k_int.transpose(-2, -1))
    _probe(f'attn_QK^T_K{head_dim}', q_int, k_int, qk_acc)
    raw_scores = qk_acc * (ctx['q_proj'] * ctx['k_proj'])

    scores = ctx['site_scores'](raw_scores * ctx['inv_sqrt_d'])
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, Lk), float('-inf'))
    attn = ctx['site_attn'](torch.softmax(scores, dim=-1))

    # ---- softmax(scores) @ V : real INT8 GEMM ----
    attn_int = torch.clamp(torch.round(attn / ctx['softmax']), 0, INT8_QMAX)
    v_int = _to_int8(vh, ctx['v_proj'])
    bound = INT8_QMAX * INT8_QMAX * Lk
    assert bound < REAL_INT8_ACCUM_BOUND, (
        f"attn@V accumulator bound {bound} for Lk={Lk} exceeds {REAL_INT8_ACCUM_BOUND}")
    av_acc = torch.matmul(attn_int, v_int)
    _probe(f'attn_AV_K{Lk}', attn_int, v_int, av_acc)
    out = av_acc * (ctx['softmax'] * ctx['v_proj'])
    out = merge_heads(out)

    # ---- out_proj ----
    op = ctx['out_proj']
    y = int8_linear(out, ctx['out_proj_in'], op['w_int'], op['w_scale'],
                    b_int=op.get('bias_int'))
    _attn_tap(ctx['name'], out_proj_in=out, out_proj_out=y)
    return y


# =============================================================================
# Calibration of the non-linear requantization sites
# =============================================================================
@torch.no_grad()
def calibrate_sites(model, rt, dataloader, device, num_batches):
    """One pass with every site in observe-and-pass-through mode: records the
    dynamic range at each non-linear boundary so a Q format can be chosen."""
    model.eval()
    rt.sites.start_calibration()
    seen = 0
    for polarity, pixels, _labels in dataloader:
        if polarity is None:
            continue
        model(polarity.to(device), pixels.to(device))
        seen += 1
        if seen >= num_batches:
            break
    rt.sites.finalize_calibration()
    rt.sites.enable(False)
    return seen
