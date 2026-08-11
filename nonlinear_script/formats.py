"""RTL 이 그대로 따라할 수 있게 **비트 단위로 정의된** 포맷 primitive.

`quantization/quant_lib/`(fp32 컨테이너에 값을 스냅하는 시뮬 스타일)와 달리, 여기는
전부 **정수 코드**로 계산합니다. `linear_script/quantize.py` 가 int8 경로에서 했던 것과
같은 역할이고, 대상이 16비트 포맷과 비선형 유닛으로 넓어진 것뿐입니다.

여기서 정한 것이 곧 RTL 스펙입니다 — 애매한 곳이 없어야 골든과 하드웨어가 비트 단위로
같아집니다.

## 포맷 (hw_flow.md §1)

    int8 / uint8   MAC 피연산자, softmax 출력
    INT32          누산기, folded bias, requant 곱수
    Q4.11 (16b)    GELU 입출력, LayerNorm xhat·출력, beta
    Q6.9  (16b)    softmax 입력
    Q1.14 (16b)    softmax 출력(내부), LayerNorm gamma
    bfloat16       LayerNorm 앞단 + residual 스트림
"""

import math

import numpy as np
import torch

INT8_QMAX = 127
I16_MAX, I16_MIN = 32767, -32768

N_GELU = 11       # Q4.11
N_SMAX = 9        # Q6.9
N_GAMMA = 14      # Q1.14
N_BETA = 11       # Q4.11
N_XHAT = 11       # Q4.11
N_LNOUT = 11      # Q4.11
LN_SHIFT = N_XHAT + N_GAMMA - N_LNOUT      # = 14, 전 사이트 공통 (hw_flow §2.6)


# =============================================================================
# bfloat16 — 1 부호 + 8 지수 + 7 가수, round-to-nearest-even
# =============================================================================
def bf16(t):
    """bf16 정밀도로 반올림하고 fp32 컨테이너로 되돌립니다.

    `Requant/Int32_To_Bf16.v`, `Bf16_Mul.v` 가 fp64 레퍼런스로 검증한 것과 같은 RNE.
    """
    if isinstance(t, torch.Tensor):
        return t.to(torch.bfloat16).to(torch.float32)
    return float(torch.tensor(float(t)).to(torch.bfloat16))


# =============================================================================
# Qm.n — 16비트 2의 보수, code · 2^-n
# =============================================================================
def q_enc(x, n):
    """실수 → Qm.n 코드 (int64 텐서). 포화 포함."""
    c = torch.round(x.double() * (2.0 ** n))
    return torch.clamp(c, I16_MIN, I16_MAX).long()


def q_dec(code, n):
    return code.double() * (2.0 ** -n)


# =============================================================================
# 정수 requantizer — 데이터패스의 유일한 형태 (hw_flow.md 머리말)
#
#     out = sat( (acc·M + (1 << (sh-1))) >> sh )
#
# `Requant/Requant_Int.v` 와 같은 식이고, `linear_script` 에서 이미 보드 검증됨.
# =============================================================================
def make_requant(ratio, m_bits=32):
    """ratio[c] → (M[c], sh).  |M| < 2^(m_bits-1) 를 만족하는 최대 sh."""
    ratio = torch.as_tensor(ratio).double()
    lim = float(2 ** (m_bits - 1) - 1)
    peak = float(ratio.abs().max())
    sh = int(math.floor(math.log2(lim / peak)))
    while True:
        M = torch.round(ratio * (2.0 ** sh))
        if float(M.abs().max()) <= lim:
            break
        sh -= 1
    assert 0 <= sh < 64, sh
    return M.long(), sh


def requant(acc, M, sh, lo, hi):
    prod = acc.long() * M.long()
    r = (1 << (sh - 1)) if sh > 0 else 0
    return torch.clamp((prod + r) >> sh, lo, hi)


# =============================================================================
# GELU — Q4.11 → Q4.11, LUT 하나 (hw_flow.md §2.3)
#
# 입력 Q4.11 은 65,536 코드지만 전부 저장할 필요가 없습니다:
#
#     code >= +8192  (x >= +4)  →  y = x        GELU(4) = 3.99987, 오차 0.27 LSB
#     code <  -8192  (x <  -4)  →  y = 0        GELU(-4) = -1.3e-4, 반올림하면 0
#     그 사이 16,384개만 LUT                     16,384 x 16b = 256 Kbit = 8 BRAM36
#
# 서브샘플+보간으로 더 줄일 수 있지만 오차가 생깁니다. BRAM 이 5% 밖에 안 쓰이므로
# **전수 LUT** 를 택했습니다 — 보간 오차라는 검증 변수를 아예 없앱니다.
# =============================================================================
GELU_LO, GELU_HI = -8192, 8192            # LUT 가 덮는 코드 구간 [LO, HI)


# `04_basic_rtl/GELU/verilog/gelu_lut.vh` 의 base/delta 를 읽어 옵니다.
# **하드웨어가 쓰는 그 파일이 원본**입니다 — 여기서 다시 계산하지 않습니다.
GELU_VH = '/hai/home/sgh/04_basic_rtl/GELU/verilog/gelu_lut.vh'
PWL_K, PWL_NSEG, PWL_FR = 4, 64, 14
PWL_FRACBITS = N_GELU - PWL_K          # 7
PWL_RSH = PWL_FR - N_GELU              # 3


def _load_pwl():
    import re
    t = open(GELU_VH).read()

    def grab(nm):
        d = {}
        for m in re.finditer(rf"{nm}\[\s*(\d+)\s*\]\s*=\s*(-?)16'sd(\d+)", t):
            d[int(m.group(1))] = (-1 if m.group(2) == '-' else 1) * int(m.group(3))
        assert len(d) == PWL_NSEG, (nm, len(d))
        return torch.tensor([d[i] for i in range(PWL_NSEG)], dtype=torch.int64)
    return grab('base14_rom'), grab('delta14_rom')


def build_gelu_lut(n_in=N_GELU, n_out=N_GELU):
    """LUT[i] = gelu_pwl 회로가 코드 `i + GELU_LO` 에 대해 내놓는 값

    ## 전수 LUT 에서 PWL 로 바뀐 이유

    전에는 `LUT[i] = Q(gelu((i+GELU_LO)·2^-11))` 로 **정확한 GELU 를 전수 저장**
    했습니다. 16,384 x 16b = 8 BRAM36 이고, 레인마다 두면 **256 BRAM** 입니다.

    지금 하드웨어(`GELU/verilog/gelu_pwl.v`)는 residual 정식화 + 64세그먼트 선형
    보간이라 **base/delta 64쌍 = 2 Kb** 면 끝납니다. 정확한 GELU 대비 최대 1 LSB
    (Q4.11 전 코드 65,536개 중 96.7%는 정확).

    그러면 **골든이 하드웨어와 달라집니다.** 그래서 여기서 정확한 GELU 를 계산하지
    않고 **PWL 경로를 그대로 재현**합니다. 표를 미리 펴 두는 것뿐이고 회로와 값이
    비트 단위로 같습니다:

        a    = |code|
        seg  = a >> 7,  frac = a & 127
        R14  = (a >= 8192) ? 0 : base[seg] + ((delta[seg]*frac) >> 7)
        R    = (R14 + 4) >> 3
        y    = (code < 0) ? -R : code - R

    포화 동작은 전수 LUT 판과 **완전히 같습니다** — `|code| >= 8192` 면 R=0 이라
    양수는 항등, 음수는 0 입니다. `gelu_q()` 의 구간 분기를 바꿀 필요가 없습니다.
    """
    assert n_in == N_GELU and n_out == N_GELU, 'PWL 회로는 Q4.11 고정입니다'
    base, delta = _load_pwl()
    code = torch.arange(GELU_LO, GELU_HI, dtype=torch.int64)
    a = code.abs()
    seg = torch.clamp(a >> PWL_FRACBITS, 0, PWL_NSEG - 1)
    frac = a & ((1 << PWL_FRACBITS) - 1)
    r14 = base[seg] + ((delta[seg] * frac) >> PWL_FRACBITS)
    r = (r14 + (1 << (PWL_RSH - 1))) >> PWL_RSH
    y = torch.where(code < 0, -r, code - r)
    return torch.clamp(y, I16_MIN, I16_MAX)


def gelu_q(code, lut):
    """Q4.11 코드 → Q4.11 코드."""
    out = torch.zeros_like(code)
    hi = code >= GELU_HI
    mid = (~hi) & (code >= GELU_LO)
    out[hi] = code[hi]                                            # 항등
    out[mid] = lut[(code[mid] - GELU_LO)]                         # LUT
    return out                                                    # 나머지(< LO) 는 0


# =============================================================================
# Softmax — Q6.9 → Q1.14 → uint8 (hw_flow.md §2.4)
#
# ## hw_flow.md 에 없는 것: max 뺄셈
#
# §2.4 는 `exp LUT → Σ → ×1/Σ` 만 적고 max 뺄셈이 없습니다. attention score 는
# 스케일링돼 있어 괜찮았을 수 있지만, **분류 헤드의 logit 은 양수로 큽니다** —
# Q6.9 범위가 ±64 이므로 exp(64) 는 그대로 넘칩니다. 그래서 max 를 빼서 지수를
# [-16, 0] 으로 묶습니다. 결과는 수학적으로 동일하고(softmax 는 shift-invariant),
# exp LUT 이 절반으로 줄어드는 부수 효과가 있습니다.
#
# ## LUT 와 나눗셈
#
#     d      = clamp(code - max_code, -8192, 0)          Q6.9,  -16 이하는 전부 0
#     e[c]   = EXP_LUT[-d]                               Q1.14, exp(d) ∈ (0, 1]
#     S      = Σ e[c]                                    ≥ 2^14 (최댓값 항이 1.0)
#     r      = RECIP_LUT[normalize(S)]                   127·2^22 / S
#     p[c]   = sat_u8( (e[c]·r + 2^(21+e)) >> (22+e) )   uint8 [0,127]
#
# 나눗셈이 남는 유일한 곳이라(§2.4) reciprocal 을 LUT 으로 못박습니다. 127 을 LUT 에
# 접어 넣어 **곱셈 한 번**으로 끝냅니다.
# =============================================================================
SMAX_RANGE = 8192                          # |d| 상한 코드 = 16.0 (Q6.9)
EXP_N = 14                                 # exp 출력 Q1.14
RECIP_N = 22                               # reciprocal LUT 스케일


def build_exp_lut(n_in=N_SMAX, n_out=EXP_N):
    """EXP_LUT[i] = round( exp(-i·2^-n_in) · 2^n_out ),  i ∈ [0, 8192)"""
    i = torch.arange(0, SMAX_RANGE, dtype=torch.float64)
    return torch.round(torch.exp(-i * (2.0 ** -n_in)) * (2.0 ** n_out)).long()


def build_recip_lut():
    """RECIP_LUT[j] = round( 127 · 2^RECIP_N / (2^14 + j) ),  j ∈ [0, 2^14)

    S 를 [2^14, 2^15) 로 정규화한 뒤 인덱스합니다. 127 을 접어 넣었으므로 곱한 뒤
    시프트만 하면 바로 uint8 입니다.
    """
    j = torch.arange(0, 1 << 14, dtype=torch.float64)
    return torch.round(127.0 * (2.0 ** RECIP_N) / ((1 << 14) + j)).long()


def softmax_u8(code, exp_lut, recip_lut):
    """Q6.9 코드 (B, C) → uint8 확률 (B, C), step 1/127."""
    d = code - code.max(dim=-1, keepdim=True).values          # ≤ 0
    idx = torch.clamp(-d, 0, SMAX_RANGE)                      # 8192 는 '범위 밖'
    e = torch.where(idx >= SMAX_RANGE, torch.zeros_like(idx), exp_lut[idx.clamp(max=SMAX_RANGE - 1)])

    S = e.sum(dim=-1, keepdim=True)                           # ≥ 2^14
    sh = torch.clamp(_bitlen(S) - 15, min=0)                  # 정규화 시프트 e
    Sn = S >> sh                                              # ∈ [2^14, 2^15)
    r = recip_lut[Sn - (1 << 14)]

    num = e * r + (1 << (RECIP_N - 1)) * (1 << sh)
    p = num >> (RECIP_N + sh)
    return torch.clamp(p, 0, 127)


def _bitlen(t):
    """정수 텐서의 비트 길이 (t > 0 가정)."""
    out = torch.zeros_like(t)
    v = t.clone()
    while bool((v > 0).any()):
        out += (v > 0).long()
        v = v >> 1
    return out


# =============================================================================
# LayerNorm affine 뒷단 — `Requant/LN_Affine.v` 와 같은 식 (hw_flow.md §2.6)
#
#     y = sat16( (code_x · code_g + (code_b << 14) + 2^13) >> 14 )
#
# shift = n_xhat + n_gamma - n_out = 11 + 14 - 11 = 14, 전 사이트 공통 상수라
# requant 곱수 M 이 필요 없습니다.
# =============================================================================
def ln_affine(xhat_code, gamma_code, beta_code, shift=LN_SHIFT):
    acc = xhat_code * gamma_code + (beta_code << shift)
    return torch.clamp((acc + (1 << (shift - 1))) >> shift, I16_MIN, I16_MAX)


# =============================================================================
# 가중치 / 파라미터 양자화
# =============================================================================
def weight_scale(W, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    return (W.abs().amax(dim=1).double() / qmax).clamp(min=1e-12)


def quant_weight(W, s_w):
    return torch.clamp(torch.round(W.double() / s_w[:, None]), -128, 127).long()


def fold_bias(b, step):
    """b_int = round(b / (s_x·s_w))  — 누산기 LSB 위의 정수 덧셈 하나."""
    return torch.clamp(torch.round(b.double() / step), -(2**31 - 1), 2**31 - 1).long()
