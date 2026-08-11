"""Pre-LN Transformer FFN 스택 — 비선형 유닛 검증용 최소 네트워크

    784 ─Linear─→ 128
      ┌──────────── block × 2 ────────────────┐
      │  h = LayerNorm(x)                     │
      │  h = Linear 128→256                   │
      │  h = GELU(h)                          │
      │  h = Linear 256→128                   │
      │  x = x + h                (residual)  │
      └───────────────────────────────────────┘
    → LayerNorm → Linear 128→10 → softmax

## 왜 이 형태인가

`quantization/hw_flow.md` 의 EvT 블록에서 **attention 만 뺀 것**입니다. attention 이
없어도 포맷 체인은 그대로 다 밟힙니다:

    Linear  → bf16      (LayerNorm 소비)          §2.6
    LN      → Q4.11     → int8                    §2.6
    Linear  → Q4.11     (GELU 소비)               §2.3
    GELU    → Q4.11     → int8                    §2.3
    Linear  → bf16      (residual)                §2.7
    Linear  → Q6.9      (softmax 소비)            §2.4
    softmax → Q1.14     → uint8                   §2.4

`linear_script` 가 int8 경로(GEMM·requant·ReLU·argmax)를 덮었으니, 여기는 **비선형
유닛과 16비트 포맷**을 덮습니다.

## 형상을 이렇게 고른 이유

| | 값 | 이유 |
|---|---|---|
| E (모델 폭) | 128 | EvT 와 동일 |
| FFN 확장 | 256 | 2배 — 실제 블록 비율 |
| 블록 수 | 2 | residual 이 **누적**돼야 bf16 스트림이 의미가 생김 |
| 출력 | 10 | softmax 확률 10개 = 이미지당 검증점 10개 (argmax 는 1개) |

마지막이 요점입니다. argmax 는 이미지당 정수 하나만 비교하지만 softmax 확률 벡터는
**uint8 10개**를 비교하므로 훨씬 촘촘합니다.
"""

import torch
import torch.nn as nn

E_IN, E_MODEL, E_FFN, N_CLASS, N_BLOCK = 784, 128, 256, 10, 2


class FFNBlock(nn.Module):
    """Pre-LN FFN 블록:  x + Linear(GELU(Linear(LayerNorm(x))))

    Pre-LN 인 이유: residual 이 LayerNorm 을 **거치지 않고** 흐르므로 bf16 스트림이
    블록을 가로질러 이어집니다 (`hw_flow.md` §2.7 의 `z_att ⊞ out_proj` 와 같은 구조).
    Post-LN 이면 residual 이 매번 정규화돼 bf16 동적 범위가 안 자랍니다.
    """

    def __init__(self, e_model=E_MODEL, e_ffn=E_FFN):
        super().__init__()
        self.norm = nn.LayerNorm(e_model)
        self.fc1 = nn.Linear(e_model, e_ffn)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(e_ffn, e_model)

    def forward(self, x):
        return x + self.fc2(self.gelu(self.fc1(self.norm(x))))


class FFNNet(nn.Module):
    def __init__(self, e_in=E_IN, e_model=E_MODEL, e_ffn=E_FFN,
                 n_class=N_CLASS, n_block=N_BLOCK):
        super().__init__()
        self.stem = nn.Linear(e_in, e_model)
        self.blocks = nn.ModuleList([FFNBlock(e_model, e_ffn) for _ in range(n_block)])
        self.norm_f = nn.LayerNorm(e_model)
        self.head = nn.Linear(e_model, n_class)
        self.dims = dict(e_in=e_in, e_model=e_model, e_ffn=e_ffn,
                         n_class=n_class, n_block=n_block)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm_f(x))          # logits (softmax 는 추론에서)

    @property
    def linears(self):
        """(이름, 모듈) — 양자화가 순서대로 훑는 Linear 목록."""
        out = [('stem', self.stem)]
        for i, b in enumerate(self.blocks):
            out += [(f'b{i}.fc1', b.fc1), (f'b{i}.fc2', b.fc2)]
        out.append(('head', self.head))
        return out

    @property
    def layernorms(self):
        out = [(f'b{i}.norm', b.norm) for i, b in enumerate(self.blocks)]
        out.append(('norm_f', self.norm_f))
        return out
