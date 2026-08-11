"""fp32 MLP 학습 → `mlp_fp32.pt`.

    python train.py                 # 기본 8 epoch
    python train.py --epochs 15
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
from model import MLP, DIMS                 # noqa: E402

CKPT = os.path.join(SCRIPT_DIR, 'mlp_fp32.pt')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}   dims: {DIMS}')

    train_loader, test_loader = data_mod.loaders(args.batch_size)
    model = MLP().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'parameters: {n_param:,}  ({n_param * 4 / 1024:.1f} KiB as fp32)')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    best = 0.0
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
        acc, n = data_mod.accuracy(model, test_loader, device)
        best = max(best, acc)
        print(f'  epoch {ep:2d}  loss {tot_loss / seen:.4f}   test acc {acc * 100:.2f}%'
              f'   ({time.time() - t0:.0f}s)')

    acc, n = data_mod.accuracy(model, test_loader, device)
    torch.save(dict(state_dict=model.state_dict(), dims=DIMS,
                    fp32_accuracy=acc, n_test=n), CKPT)
    print(f'\nfp32 accuracy: {acc * 100:.2f}%  (n={n})')
    print(f'saved -> {CKPT}')


if __name__ == '__main__':
    main()
