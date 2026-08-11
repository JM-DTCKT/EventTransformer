"""MNIST 로더. 처음 실행 시 `linear_script/data/` 로 자동 다운로드합니다.

전처리는 **ToTensor 만** 씁니다 (픽셀 [0,1]).
정규화(mean/std)를 일부러 넣지 않은 이유:
  * 입력이 [0,1] 이면 첫 Linear 의 activation scale 이 `1/127` 로 딱 떨어져
    RTL 로 넣을 입력 코드가 곧 `round(pixel*127)` 이 됩니다. 골든 벡터를 손으로
    검산하기 쉽습니다.
  * 정규화는 첫 Linear 의 weight/bias 로 흡수되므로 정확도 손해가 없습니다.
학습/추론 모두 augmentation 이 없어 **결정론적**입니다 (캘리브레이션 재현 가능).
"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


def loaders(batch_size=128, workers=2, download=True):
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
        pred = model(x.to(device)).argmax(dim=-1)    # 추론은 argmax 만
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return correct / total, total
