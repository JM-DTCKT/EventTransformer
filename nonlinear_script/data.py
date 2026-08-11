"""MNIST 로더 — `linear_script/data/` 를 그대로 재사용합니다 (재다운로드 없음).

전처리는 **ToTensor 만** (픽셀 [0,1]). `linear_script/data.py` 와 같은 이유입니다:
입력이 [0,1] 이면 첫 Linear 의 activation scale 이 `1/127` 로 딱 떨어져 RTL 입력 코드가
곧 `round(pixel*127)` 이 되고, 골든 벡터를 손으로 검산할 수 있습니다. 정규화는 첫
Linear 의 weight/bias 로 흡수되므로 정확도 손해가 없습니다.

augmentation 이 없어 **결정론적**입니다 — 캘리브레이션이 재현됩니다.
"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# linear_script 가 이미 받아 둔 것을 씁니다
DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'linear_script', 'data'))


def loaders(batch_size=128, workers=2, download=False):
    tf = transforms.ToTensor()                       # [0,1], (1,28,28)
    train = datasets.MNIST(DATA_DIR, train=True, download=download, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=download, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers),
    )


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=-1)
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return correct / total, total
