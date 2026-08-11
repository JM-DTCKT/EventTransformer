"""fp32 학습 → `ffn_fp32.pt`

    python train.py                 # 기본 12 epoch
    python train.py --epochs 20
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data as data_mod                     # noqa: E402
from model import FFNNet                    # noqa: E402

CKPT = os.path.join(SCRIPT_DIR, 'ffn_fp32.pt')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = data_mod.loaders(args.batch_size)
    model = FFNNet().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'device: {device}   dims: {model.dims}')
    print(f'parameters: {n_param:,}  ({n_param * 4 / 1024:.1f} KiB as fp32)')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot_loss, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.numel()
            seen += y.numel()
        sched.step()
        acc, _ = data_mod.accuracy(model, test_loader, device)
        print(f'  epoch {ep:2d}  loss {tot_loss / seen:.4f}   test acc {acc * 100:.2f}%'
              f'   ({time.time() - t0:.0f}s)')

    acc, n = data_mod.accuracy(model, test_loader, device)
    torch.save(dict(state_dict=model.state_dict(), dims=model.dims,
                    fp32_accuracy=acc, n_test=n), CKPT)
    print(f'\nfp32 accuracy: {acc * 100:.2f}%  (n={n})')
    print(f'saved -> {CKPT}')


if __name__ == '__main__':
    main()
