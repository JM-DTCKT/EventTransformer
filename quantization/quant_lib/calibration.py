"""Activation calibration for static (weight + activation) quantization cases."""

import torch


@torch.no_grad()
def calibrate_activations(model, quantizer, dataloader, device, num_batches):
    """Run `num_batches` batches through `model` in observe-mode so `quantizer`
    can record per-module max(|activation|) statistics, then freeze them.

    Callers should pass `num_batches=len(dataloader)` to calibrate over a full
    pass of the training set (the default in `run_quantization.py`) -- using
    only a handful of batches is enough for small datasets where a few dozen
    batches already exceed one epoch (e.g. DVS128), but under-covers large
    ones (e.g. ASL_DVS, where 1 epoch = 1260 batches).
    """
    if quantizer.act_bits is None:
        return

    model.eval()
    quantizer.start_calibration()
    seen = 0
    for polarity, pixels, _labels in dataloader:
        if polarity is None:
            continue
        polarity = polarity.to(device)
        pixels = pixels.to(device)
        model(polarity, pixels)
        seen += 1
        if seen >= num_batches:
            break
    quantizer.stop_calibration()
    return seen
