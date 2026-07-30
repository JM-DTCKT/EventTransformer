"""
Hook-based fake-quantization wrapper for the EvT model.

Scope of what gets quantized (chosen to match what actually dominates the
model's parameter count and compute, without needing to touch/modify
`models/EvT.py`):

  * Weight quantization is applied in-place to:
      - every `nn.Linear.weight` (MLPBlock / LatentEmbsCompressor / CLFBlock /
        AttentionBlock's linear1-3, and `out_proj` inside every
        `nn.MultiheadAttention`)
      - `nn.MultiheadAttention.in_proj_weight` (the packed Q/K/V projection,
        which is a raw Parameter rather than a Linear submodule)
    Biases, LayerNorm affine params, the learned positional encoding and the
    latent memory vectors are kept at fp32: they are a tiny fraction of the
    parameter count and empirically far more sensitive to quantization noise,
    which mirrors common FPGA quantization practice of keeping norms/biases
    at higher precision.

  * Activation quantization (only used for the W8A8 / W4A8 cases) is applied
    via `forward_pre_hook`s on the same `nn.Linear` and `nn.MultiheadAttention`
    modules, fake-quantizing their input tensor(s) (query/key/value for
    attention) using a scale calibrated ahead of time on a few representative
    batches. Note: PyTorch's fused `F.multi_head_attention_forward` calls
    `out_proj`'s weight/bias as plain tensors internally rather than invoking
    `out_proj.forward`, so `out_proj`'s *input* activation is not separately
    hooked (its *weight* still is, via the loop above) -- this is a deliberate
    simplification, documented here rather than worked around.
"""

import torch
import torch.nn as nn

from . import quant_ops


class ModelQuantizer:
    def __init__(self, model, weight_bits=None, weight_per_channel=True, act_bits=None):
        self.model = model
        self.weight_bits = weight_bits
        self.weight_per_channel = weight_per_channel
        self.act_bits = act_bits

        self._act_scale = {}     # module-name -> calibrated scalar scale
        self._hook_handles = []
        self._calibrating = False

    # ------------------------------------------------------------------
    # Weight quantization (applied once, in-place, irreversible for this
    # model instance -- callers should load a fresh model per case).
    # ------------------------------------------------------------------
    def quantize_weights(self):
        if self.weight_bits is None:
            return
        with torch.no_grad():
            for _, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    self._quantize_weight_(m.weight)
                elif isinstance(m, nn.MultiheadAttention):
                    self._quantize_weight_(m.in_proj_weight)

    def _quantize_weight_(self, weight_param):
        scale = quant_ops.compute_qparams(
            weight_param.data, self.weight_bits,
            per_channel=self.weight_per_channel, channel_dim=0,
        )
        weight_param.data.copy_(quant_ops.fake_quantize(weight_param.data, scale, self.weight_bits))

    # ------------------------------------------------------------------
    # Activation quantization: calibrate() first, then enable hooks for eval.
    # ------------------------------------------------------------------
    def _act_targets(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.MultiheadAttention):
                yield name, m

    def start_calibration(self):
        if self.act_bits is None:
            return
        self._act_scale.clear()
        self._calibrating = True
        self._register_hooks()

    def stop_calibration(self):
        self._calibrating = False
        self._remove_hooks()

    def enable_quantized_activations(self):
        if self.act_bits is None:
            return
        self._calibrating = False
        self._register_hooks()

    def remove_activation_hooks(self):
        self._remove_hooks()

    def _register_hooks(self):
        self._remove_hooks()
        for name, m in self._act_targets():
            if isinstance(m, nn.Linear):
                handle = m.register_forward_pre_hook(self._make_linear_hook(name))
            else:
                handle = m.register_forward_pre_hook(self._make_mha_hook(name))
            self._hook_handles.append(handle)

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _make_linear_hook(self, name):
        def hook(module, args):
            x = self._process_tensor(name, args[0])
            return (x,) + tuple(args[1:])
        return hook

    def _make_mha_hook(self, name):
        # nn.MultiheadAttention is always called as attention(query, key, value, ...)
        # in this codebase (see models/EvT.py AttentionBlock.forward).
        def hook(module, args):
            q = self._process_tensor(name + ".q", args[0])
            k = self._process_tensor(name + ".k", args[1])
            v = self._process_tensor(name + ".v", args[2])
            return (q, k, v) + tuple(args[3:])
        return hook

    def _process_tensor(self, key, x):
        if x is None or not torch.is_tensor(x):
            return x
        if self._calibrating:
            cur_max = x.detach().abs().max().item()
            self._act_scale[key] = max(self._act_scale.get(key, 0.0), cur_max)
            return x
        max_abs = self._act_scale.get(key)
        if not max_abs:
            return x
        qmax = 2 ** (self.act_bits - 1) - 1
        scale = max_abs / qmax
        q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
        return q * scale

    def act_scales_snapshot(self):
        return dict(self._act_scale)
