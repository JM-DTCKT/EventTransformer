"""Checkpoint loading, dataset wiring and accuracy evaluation shared by all
quantization cases.

Reuses the existing `EventTransformer/trainer.py::EvNetModel` and
`EventTransformer/data_generation.py::Event_DataModule` (read-only imports,
nothing under `EventTransformer/` outside of `quantization/` is modified).
"""

import copy
import json
import os
import sys

import torch
from tqdm import tqdm

QUANT_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
QUANTIZATION_DIR = os.path.dirname(QUANT_LIB_DIR)
EVENTTRANSFORMER_DIR = os.path.dirname(QUANTIZATION_DIR)

if EVENTTRANSFORMER_DIR not in sys.path:
    sys.path.insert(0, EVENTTRANSFORMER_DIR)

from trainer import EvNetModel          # noqa: E402
from data_generation import Event_DataModule  # noqa: E402

# `Event_DataModule` hard-codes dataset locations relative to CWD (DVS128) or
# to an absolute path outside the repo (ASL_DVS). DVS128's relative path is
# rewritten below to an absolute path so this package works regardless of the
# caller's working directory; ASL_DVS's absolute `/data/sgh/ASL_DVS/...` path
# already exists on this machine and is used as-is (see plan notes).
DVS128_LOCAL_DATA_FOLDER = os.path.join(
    EVENTTRANSFORMER_DIR, 'datasets', 'DvsGesture', 'clean_dataset_frames_12000'
) + '/'


def find_best_checkpoint(weights_dir, metric='val_acc', mode='max'):
    """Pick the checkpoint file whose filename encodes the best `metric`.

    Checkpoint filenames look like:
      epoch=210-val_loss_total=0.00145-val_loss_clf=0.00145-val_acc=0.99965.ckpt
    (possibly with a `-v1`/`-v2` suffix for duplicate ModelCheckpoint callbacks
    that happened to trigger on the same epoch -- those are identical weights).
    """
    assert mode in ('min', 'max')
    agg = max if mode == 'max' else min
    files = [f for f in os.listdir(weights_dir) if f.endswith('.ckpt')]

    def _score(fname):
        stem = fname[:-len('.ckpt')]
        for token in stem.split('-'):
            if token.startswith(metric + '='):
                try:
                    return float(token.split('=')[1])
                except ValueError:
                    pass
        return float('-inf') if mode == 'max' else float('inf')

    best = agg(files, key=_score)
    return os.path.join(weights_dir, best)


def load_model(weights_path, all_params_path, device='cpu'):
    """Load a fresh `EvNetModel` from checkpoint. Call this once per
    quantization case -- weight quantization mutates the model in-place, so
    reusing an already-quantized instance would compound quantization noise.
    """
    all_params = json.load(open(all_params_path))
    model = EvNetModel.load_from_checkpoint(
        weights_path, map_location=torch.device('cpu'), **copy.deepcopy(all_params)
    )
    model.eval()
    model.to(device)
    return model, all_params


def build_datamodule(all_params, workers=4, batch_size=None):
    data_params = copy.deepcopy(all_params['data_params'])
    data_params['pin_memory'] = False
    data_params['workers'] = workers
    if batch_size is not None:
        data_params['batch_size'] = batch_size
    dm = Event_DataModule(**data_params)
    if dm.dataset_name == 'DVS128':
        dm.data_folder = DVS128_LOCAL_DATA_FOLDER
    return dm


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device, dtype=torch.float32, max_samples=None, desc=''):
    """Full test-set top-1 accuracy. `dtype` should be float16 for the fp16
    case (with `model` already cast via `.half()`), float32 otherwise --
    `pixels`/`labels` are always kept as integer tensors regardless of dtype.
    """
    model.eval()
    correct, total = 0, 0
    for polarity, pixels, labels in tqdm(dataloader, desc=desc, leave=False):
        if polarity is None:
            continue
        polarity = polarity.to(device=device, dtype=dtype)
        pixels = pixels.to(device)
        labels = labels.to(device)
        _embs, clf_logits = model(polarity, pixels)
        preds = clf_logits.float().argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        if max_samples is not None and total >= max_samples:
            break
    acc = correct / total if total > 0 else float('nan')
    return acc, total
