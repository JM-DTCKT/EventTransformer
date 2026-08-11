"""Activation-scale calibration for the integer datapath.

One pass over training batches records `max|x|` at every point the integer
datapath needs an INT8 activation step, and turns it into a symmetric
per-tensor scale (`max / 127`). Nothing here fake-quantizes anything: the
model runs in fp32 and the observers only look.

Calibrated points
-----------------
`linear_act_scales`   input of every standalone `nn.Linear` (stored as the raw
                      `max|x|`; `hw_quant.build_payload` divides by 127).
`q_in/k_in/v_in`      the three inputs of every `nn.MultiheadAttention`, i.e.
                      what the packed `Wq|Wk|Wv` projection multiplies.
`qk_scales`, `v_scale`  post-projection Q / K / V -- the operands of the two
                      attention MACs.
`out_proj_scales`     input of each `out_proj`, i.e. the merged `attn @ V`.
These four are stored already divided by 127.

Padded key/value positions are excluded from the K/V maxima: an all-zero
padded event slot is not real data, and letting it into the max would only
ever shrink the step. `bias_k`/`bias_v` occupy one extra key slot that is
never padded (see `hw_flow.md` 10.2), so they are included.

Why a reference attention forward lives here
--------------------------------------------
`out_proj`'s input only exists *inside* attention, and PyTorch's fused
`F.multi_head_attention_forward` never calls `out_proj.forward`, so a hook
cannot see it. `attention_forward_ref` below reproduces
`nn.MultiheadAttention` for this codebase's fixed usage (batch_first=False,
add_bias_kv=True, attn_mask unused, eval mode) in fp32. `sanity_check` asserts
it matches PyTorch's own kernel before any scale is trusted -- run it once per
dataset.
"""

import copy
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hw_quant import INT8_QMAX, _block_name_of_attention, merge_heads, split_heads


class _MaxObserver:
    """Running `max|x|`, optionally ignoring masked-out positions."""

    def __init__(self):
        self.max_abs = 0.0

    @torch.no_grad()
    def observe(self, x, exclude_mask=None):
        t = x.detach()
        if exclude_mask is not None:
            t = torch.where(exclude_mask, torch.zeros((), dtype=t.dtype, device=t.device), t)
        t = t[torch.isfinite(t)]
        if not t.numel():
            return
        self.max_abs = max(self.max_abs, float(t.abs().max()))

    @property
    def scale(self):
        return max(self.max_abs, 1e-8) / INT8_QMAX


def _kv_mask_like(key_padding_mask, x):
    """(B, Lk) bool -> broadcastable to x of shape (Lk, B, E). True = padded."""
    if key_padding_mask is None:
        return None
    return key_padding_mask.t().unsqueeze(-1).expand_as(x)


def attention_forward_ref(q_in, k_in, v_in, mha, key_padding_mask, obs=None):
    """fp32 `nn.MultiheadAttention` for this codebase's fixed usage.

    With `obs=None` this is a plain reference implementation (that is what
    `sanity_check` compares against PyTorch). With `obs` it additionally
    records the maxima the integer datapath needs.
    """
    E, H = mha.embed_dim, mha.num_heads
    B = q_in.shape[1]

    if obs is not None:
        obs['q_in'].observe(q_in)
        obs['k_in'].observe(k_in, _kv_mask_like(key_padding_mask, k_in))
        obs['v_in'].observe(v_in, _kv_mask_like(key_padding_mask, v_in))

    Wq, Wk, Wv = mha.in_proj_weight.chunk(3, dim=0)
    bq, bk, bv = (mha.in_proj_bias.chunk(3, dim=0) if mha.in_proj_bias is not None
                  else (None, None, None))
    q = F.linear(q_in, Wq, bq)
    k = F.linear(k_in, Wk, bk)
    v = F.linear(v_in, Wv, bv)

    if mha.bias_k is not None:
        k = torch.cat([k, mha.bias_k.expand(1, B, E)], dim=0)
        v = torch.cat([v, mha.bias_v.expand(1, B, E)], dim=0)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, key_padding_mask.new_zeros(B, 1)], dim=1)

    qh, kh, vh = split_heads(q, H), split_heads(k, H), split_heads(v, H)
    Lk = kh.shape[2]
    head_mask = (key_padding_mask.view(B, 1, Lk, 1) if key_padding_mask is not None else None)

    if obs is not None:
        obs['q_proj'].observe(qh)
        obs['k_proj'].observe(kh, head_mask.expand_as(kh) if head_mask is not None else None)
        obs['v_proj'].observe(vh, head_mask.expand_as(vh) if head_mask is not None else None)

    scores = torch.matmul(qh, kh.transpose(-2, -1)) / float(E // H) ** 0.5
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, Lk), float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = merge_heads(torch.matmul(attn, vh))

    if obs is not None:
        obs['out_proj'].observe(out)
    return mha.out_proj(out)


# =============================================================================
# Patching
# =============================================================================
def _patch_attention(model, obs_by_block=None):
    """Route every `nn.MultiheadAttention` through `attention_forward_ref`."""
    for name, m in model.named_modules():
        if not isinstance(m, nn.MultiheadAttention):
            continue
        m._obs = obs_by_block.get(_block_name_of_attention(name)) if obs_by_block else None

        def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, **kw):
            assert attn_mask is None, 'attn_mask is unused in this model'
            return attention_forward_ref(query, key, value, self,
                                         key_padding_mask, self._obs), None

        m.forward = types.MethodType(forward, m)


def _standalone_linears(model):
    """Every `nn.Linear` except the `out_proj` inside an attention block -- that
    one is calibrated as `out_proj_scales`, keyed by block."""
    inside_mha = {f'{n}.out_proj' for n, m in model.named_modules()
                  if isinstance(m, nn.MultiheadAttention)}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name not in inside_mha:
            yield name, m


@torch.no_grad()
def calibrate_activation_scales(model, dataloader, device, num_batches):
    """One observe-only pass. Returns (`scales` dict for `build_payload`, n_batches)."""
    model.eval().to(device)

    blocks = [_block_name_of_attention(n) for n, m in model.named_modules()
              if isinstance(m, nn.MultiheadAttention)]
    keys = ('q_in', 'k_in', 'v_in', 'q_proj', 'k_proj', 'v_proj', 'out_proj')
    obs_by_block = {b: {k: _MaxObserver() for k in keys} for b in blocks}
    lin_obs, handles = {}, []

    for name, m in _standalone_linears(model):
        lin_obs[name] = _MaxObserver()
        handles.append(m.register_forward_pre_hook(
            (lambda o: lambda _m, args: o.observe(args[0]))(lin_obs[name])))
    _patch_attention(model, obs_by_block)

    seen = 0
    for batch in dataloader:
        if seen >= num_batches:
            break
        if batch[0] is None:
            continue
        model(batch[0].to(device), batch[1].to(device))
        seen += 1
    for h in handles:
        h.remove()

    return dict(
        linear_act_scales={n: o.max_abs for n, o in lin_obs.items()},   # raw max|x|
        q_in_scales={b: o['q_in'].scale for b, o in obs_by_block.items()},
        k_in_scales={b: o['k_in'].scale for b, o in obs_by_block.items()},
        v_in_scales={b: o['v_in'].scale for b, o in obs_by_block.items()},
        qk_scales={b: dict(q_scale=o['q_proj'].scale, k_scale=o['k_proj'].scale)
                   for b, o in obs_by_block.items()},
        v_scale={b: o['v_proj'].scale for b, o in obs_by_block.items()},
        out_proj_scales={b: o['out_proj'].scale for b, o in obs_by_block.items()},
    ), seen


@torch.no_grad()
def sanity_check(model, dataloader, device, num_batches=3, atol=1e-4):
    """`attention_forward_ref` (no quantization) vs PyTorch's own kernel.

    If the reimplementation were wrong every calibrated scale would be wrong
    too, so this runs before the scales are used. Returns the max logit diff.
    """
    model.eval().to(device)
    ref = copy.deepcopy(model)
    _patch_attention(ref)

    worst, seen = 0.0, 0
    for batch in dataloader:
        if seen >= num_batches:
            break
        if batch[0] is None:
            continue
        a, b = batch[0].to(device), batch[1].to(device)
        # the model returns (embeddings, clf_logits) -- compare the logits
        worst = max(worst, float((model(a, b)[1] - ref(a, b)[1]).abs().max()))
        seen += 1
    del ref
    assert worst <= atol, (
        f'attention_forward_ref disagrees with nn.MultiheadAttention by {worst:.3e} '
        f'(> {atol:.0e}) -- calibrated scales would be meaningless')
    return worst
