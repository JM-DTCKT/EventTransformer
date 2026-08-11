"""RTL 브링업용 MLP — Linear 와 ReLU 만으로 이뤄진 최소 망.

`04_basic_rtl` 의 세 블록을 **정확히 그대로** 태우기 위한 망입니다:

    Linear   → Mac_OS/Linear_Top   (INT8 x INT8 -> INT32 GEMM)
    requant  → Requant/Requant_Int (sat((acc*M + 2^(sh-1)) >> sh))
    ReLU     → ReLU/ReLU           (부호 비트 마스킹, 또는 requant 에 융합)

EvT 와 달리 attention / LayerNorm / GELU / softmax 가 없으므로, 정수 데이터패스에
남는 것은 **INT8 · INT32 · M/shift** 뿐입니다. RTL 이 맞는지 확인하는 데 필요한
최소 구성이면서, 마지막 층이 argmax 로 끝나는 것까지 EvT 와 같습니다.

형상 선택 이유 (Mac_OS 파라미터 기준: N=32, MAX_IN/MAX_OUT/M_MAX = 1024):
    784 -> 128 -> 64 -> 10
    * 784, 128, 64 모두 1024 이하           → 타일링이 그대로 동작
    * 784 = 24.5 x 32 → K 가 N 의 배수가 아님 → 부분 타일 경로를 탐
    * 마지막 out=10 < N=32                  → 출력 부분 타일 경로를 탐
"""

import torch.nn as nn

DIMS = (784, 128, 64, 10)


class MLP(nn.Module):
    """Linear -> ReLU -> Linear -> ReLU -> Linear (마지막은 활성화 없음)."""

    def __init__(self, dims=DIMS):
        super().__init__()
        self.dims = tuple(dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # (B, 1, 28, 28) -> (B, 784).  추론은 argmax 만 쓰므로 softmax 없음.
        return self.net(x.reshape(x.shape[0], -1))

    @property
    def linears(self):
        """[(name, module)] — 양자화/내보내기가 도는 순서 그대로."""
        return [(n, m) for n, m in self.net.named_children() if isinstance(m, nn.Linear)]
