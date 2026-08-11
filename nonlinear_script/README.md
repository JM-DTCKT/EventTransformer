# nonlinear_script — 비선형 유닛 w8a8 양자화 골든 모델

`quantization/hw_flow.md` 의 **Q4.11 · Q6.9 · bf16 requantizer 와 GELU · Softmax ·
LayerNorm** 을 하드웨어로 검증하기 위한 골든 모델입니다.

`linear_script` 가 int8 경로(GEMM · requant · ReLU · argmax)를 덮었으니, 여기는
**비선형 유닛과 16비트 포맷**을 덮습니다.

여기서 나온 값이 곧 RTL 스펙이고, 이 모델이 만든 골든 벡터로
[`04_basic_rtl/fpga_nl/`](../../../04_basic_rtl/fpga_nl/) 가 ZCU102 에서 검증됐습니다.

---

## 1. 결과

| | |
|---|---|
| fp32 | **98.50 %** |
| 정수 (w8a8 + 고정소수) | **98.48 %** |
| 정확도 손실 | **0.02 %p** |
| 보드 (ZCU102, 10,000장) | **98.48 %** — 확률 10개까지 골든과 바이트 단위 일치 |

`04_basic_rtl/fpga_nl` 보드 실행에서 **10,000장 전부, uint8 확률 100,000개가 이
모델과 완전히 같았습니다.** argmax 만이 아니라 softmax 출력 전체가 일치합니다.

파라미터 234,378개 — fp32 915.5 KiB → **int8 가중치 227.2 KiB**.

---

## 2. 네트워크

attention 만 뺀 EvT 블록입니다. attention 이 없어도 포맷 체인은 하나도 빠짐없이 밟습니다.

```
784 ─Linear─→ 128                                    INT32 → bf16
  ┌─────────────── block × 2 ────────────────────┐
  │  LayerNorm        bf16 → Q4.11 → int8        │
  │  Linear 128→256   INT32 → Q4.11              │
  │  GELU             Q4.11 → Q4.11 → int8       │
  │  Linear 256→128   INT32 → bf16               │
  │  residual ⊞       bf16 + bf16                │
  └──────────────────────────────────────────────┘
  LayerNorm → Linear 128→10 → Q6.9 → Softmax → uint8
```

| | 값 | 이유 |
|---|---|---|
| E (모델 폭) | 128 | EvT 와 동일 |
| FFN 확장 | 256 | 2배 — 실제 블록 비율 |
| 블록 수 | **2** | residual 이 **누적**돼야 bf16 스트림이 의미가 생김 |
| 출력 | 10 | **softmax 확률 10개 = 이미지당 검증점 10개** |

마지막이 요점입니다. argmax 는 이미지당 정수 하나만 비교하지만, softmax 확률 벡터는
uint8 10개를 비교하므로 검증 밀도가 10배입니다. `linear_script` 에서 argmax 버그가
32장 시뮬을 통과해 보드까지 갔던 경험이 이 선택의 이유입니다.

---

## 3. 양자화 방식

### 3.1 공통

- **w8a8, 대칭(zero-point 0)**, 가중치는 **출력채널별** 스케일
- bias 는 누산기 LSB 위의 정수로 접습니다: `b_int = round(b / (s_x·s_w))`
- 재양자화는 데이터패스 전체에서 **한 가지 형태**뿐입니다:

```
out = sat( (acc·M[c] + 2^(sh-1)) >> sh ),    M[c] = round(s_x·s_w[c] / lsb_out · 2^sh)
```

`sh` 는 `|M| < 2^31` 를 만족하는 최대값으로 잡습니다 — 곱수 정밀도를 최대한 씁니다.

### 3.2 소비자별 출력 포맷

각 Linear 의 출력 포맷은 **다음 유닛이 무엇을 원하는지**로 정해집니다.

| Linear | 소비자 | 출력 | sh | M 범위 |
|---|---|---|---|---|
| `stem` | LayerNorm | **bf16** | — | `step` 을 bf16 으로 저장 |
| `b*.fc1` | GELU | **Q4.11** | 33 | 5.9e8 ~ 1.4e9 |
| `b*.fc2` | residual | **bf16** | — | `step` 을 bf16 으로 저장 |
| `head` | Softmax | **Q6.9** | 35 | 8.5e8 ~ 1.2e9 |
| `*.norm` (LN 출력) | GEMM | **int8** | 37 | 스칼라 하나 |
| `*.gelu` (GELU 뒤) | GEMM | **int8** | 37 | 스칼라 하나 |

bf16 소비자는 `M/sh` 대신 `step = s_x·s_w[c]` 를 곱합니다. **이 상수도 bf16 으로
저장**하는 게 중요합니다 — 자세한 것은 §5.

### 3.3 포맷 여유

캘리브레이션으로 관측한 동적 범위 대비 표현 범위입니다.

| 사이트 | 포맷 | 관측 \|x\|max | 범위 | 여유 |
|---|---|---|---|---|
| `b0.fc1.out` | Q4.11 | 7.45 | ±16 | 2.15배 |
| `b1.fc1.out` | Q4.11 | 6.62 | ±16 | 2.42배 |
| `b0.norm.out` | Q4.11 | 4.68 | ±16 | 3.42배 |
| `norm_f.out` | Q4.11 | 4.50 | ±16 | 3.56배 |
| `head.out` | Q6.9 | 15.29 | ±64 | 4.19배 |

가장 빡빡한 곳도 2배 이상 남습니다. `hw_flow.md` 가 고른 포맷이 이 크기의 FFN 에는
충분하다는 뜻입니다.

### 3.4 LayerNorm 파라미터

```
gamma  Q1.14   beta  Q4.11   xhat/출력  Q4.11
y = sat16( (xhat·gamma + (beta << 14) + 2^13) >> 14 )
```

`shift = 11 + 14 - 11 = 14` 로 **전 사이트 공통 상수**라 requant 곱수가 필요 없습니다.

`gamma` 를 Q4.11 로 내리지 않는 이유: gamma 는 xhat 에 *곱해지므로* 오차가 증폭됩니다.
Q4.11(lsb 4.9e-4)이면 출력 오차가 `9.02 × 2.4e-4 ≈ 4.5 LSB`, Q1.14(lsb 6.1e-5)면
0.55 LSB 입니다. 저장 비용은 둘 다 int16 으로 같습니다. `beta` 는 *더해지기만* 하므로
출력 LSB 보다 정밀할 이유가 없어 Q4.11 입니다.

학습된 값은 Q1.14 범위(±2) 안에 잘 들어옵니다:

| | gamma \|max\| | beta \|max\| |
|---|---|---|
| `b0.norm` | 1.131 | 0.053 |
| `b1.norm` | 1.162 | 0.109 |
| `norm_f` | 1.270 | 0.037 |

---

## 4. `hw_flow.md` 에서 바꾼 것

### 4.1 rsqrt 에 Newton-Raphson 이 필요 없습니다

§2.4 는 reciprocal 에 "LUT + Newton-Raphson" 을 적었지만, **bf16 rsqrt 는 256엔트리
LUT 으로 전수**입니다:

```
rsqrt(2^e · 1.m) = 2^(-(e-p)/2) · rsqrt(2^p · 1.m),   p = e mod 2
                                   └─ 가수 7b + 지수 홀짝 1b = 256가지가 전부
```

나머지는 지수 뺄셈뿐이고 **모든 입력에서 정확한 round-to-nearest** 가 나옵니다.
`Bf16_Rsqrt.v` 를 양수 bf16 전 영역 32,768개로 확인했습니다.

이게 성립하려면 `rsqrt` **입력을 먼저 bf16 으로 못박아야** 합니다. 그래서 골든도
`bf16(var + eps)` 를 넣습니다.

### 4.2 Softmax 에 max 뺄셈을 추가했습니다

§2.4 는 `exp LUT → Σ → ×1/Σ` 만 적고 max 뺄셈이 없습니다. attention score 는 스케일링돼
있어 괜찮았을 수 있지만, **분류 헤드의 logit 은 양수로 큽니다** — Q6.9 범위가 ±64 라
`exp(64)` 는 그대로 넘칩니다.

max 를 빼면 지수가 `[-16, 0]` 으로 묶입니다. softmax 는 shift-invariant 라 결과는
수학적으로 동일하고, exp LUT 이 절반으로 줄어드는 부수 효과가 있습니다.

**127 을 reciprocal LUT 에 접어 넣어** 나눗셈을 곱셈 한 번으로 만들었습니다:

```
d    = clamp(code - max_code, -8192, 0)          Q6.9
e[c] = EXP_LUT[-d]                               Q1.14, exp(d) ∈ (0, 1]
S    = Σ e[c]                                    ≥ 2^14  (최댓값 항이 정확히 1.0)
r    = RECIP_LUT[normalize(S)]                   round(127·2^22 / S)
p[c] = sat_u8( (e[c]·r + 2^(21+sh)) >> (22+sh) ) uint8 [0,127]
```

최댓값 항이 `exp(0) = 1.0` 이라 `S ≥ 2^14` 가 **보장**되고, 그래서 정규화가 항상
성립합니다.

### 4.3 GELU 는 보간 없이 전수 LUT

Q4.11 입력은 65,536 코드지만 전부 저장할 필요가 없습니다:

```
code >= +8192  (x >= +4)  →  y = x     GELU(4) = 3.99987, 오차 0.27 LSB
code <  -8192  (x <  -4)  →  y = 0     GELU(-4) = -1.3e-4, 반올림하면 0
그 사이 16,384개만 LUT
```

서브샘플+보간으로 더 줄일 수 있지만 **보간 오차라는 검증 변수가 새로 생깁니다.**
BRAM 이 충분해서 전수 LUT 을 택했습니다.

---

## 5. 골든이 틀렸던 것 — 상수도 bf16 입니다

통합 검증에서 bf16 출력의 30%가 1 ULP 씩 어긋났습니다. 조사해 보니 **RTL 이 옳고
골든이 틀렸습니다.**

하드웨어는 `step = s_x·s_w[c]` 를 **bf16 16비트로** PB_Mem 에 저장합니다. 골든은
full-precision `step` 으로 곱하고 있었습니다.

```python
def _to_bf16(e, acc):                     # 상수도 bf16 으로 저장됩니다
    step_bf = F.bf16(e['step'].float()).double()
    return F.bf16(acc.double() * step_bf)
```

**하드웨어가 실제로 저장하는 값을 기준으로** 골든을 고쳤고, 정확도는 98.48 % 로
유지됐습니다. 골든 모델은 수학적으로 옳은 것이 아니라 **하드웨어와 같아야** 합니다.

---

## 6. 누산 순서를 못박았습니다

`torch.mean` 은 pairwise 누산이라 순서가 정의되지 않습니다. RTL 은 fp32 가산기 하나를
`k = 0..E-1` 순서로 돌리므로 골든도 **순차 누산**으로 맞췄습니다.

```python
def _fp32_sum(x):     # k=0..E-1 순차 fp32 누산 (torch.sum 은 pairwise 라 HW 와 어긋남)
    acc = torch.zeros(x.shape[:-1] + (1,), dtype=torch.float32)
    for k in range(x.shape[-1]):
        acc = acc + x[..., k:k+1].float()
    return acc
```

이 변경만으로 정확도가 **98.52 % → 98.46 %** 로 움직였습니다. 부동소수 덧셈은
결합법칙이 성립하지 않으니 **순서가 실제로 결과를 바꿉니다.**

(이후 다른 수정들과 합쳐져 최종 98.48 %.)

### 누산 정밀도

LayerNorm 의 `mean` 과 `mean(centered²)` 는 **fp32 로 누산하고 마지막에 한 번만 bf16
으로 반올림**합니다. 매 단계 bf16 으로 반올림하면 128개 합에서 오차가 누적돼 분산이
망가집니다. RTL 에는 fp32 가산기가 하나 필요하고 두 패스가 공유합니다.

### 곱셈기가 두 종류인 이유

```
var  = bf16( Σ(ctr × ctr) / E )     ← ctr² 는 fp32 에서 정확해야 함 → Bf16_Mul_Fp32
xhat = Q4.11( bf16(ctr × rstd) )    ← 여기는 bf16 반올림이 맞음    → Bf16_Mul
```

bf16 가수가 8비트라 곱이 16비트고, fp32 가수 24비트 안에 들어가 **반올림이 낄 자리가
없습니다.** 제곱을 먼저 8비트로 깎으면 분산이 달라집니다.

---

## 7. LUT

| | 엔트리 | 크기 | BRAM36 |
|---|---|---|---|
| `gelu.hex` | 16,384 × 16b | 32.0 KiB | 7.1 |
| `recip.hex` | 16,384 × 16b | 32.0 KiB | 7.1 |
| `exp.hex` | 8,192 × 16b | 16.0 KiB | 3.6 |
| `rsqrt.hex` | 256 × 16b | 0.5 KiB | ~0 |

`export_rtl.py` 는 rsqrt LUT 을 만든 뒤 **무작위 20만 개로 전수성을 자체 검증**합니다.

---

## 8. 포맷 경계마다 tap

레이어 출력만 보면 유닛이 많아 어디가 틀렸는지 못 짚습니다. 그래서 경계마다 찍습니다.

| tap | 포맷 | 무엇을 검증하나 |
|---|---|---|
| `stem.acc` | INT32 | GEMM |
| `stem.bf16` | bf16 | `Requant_Bf16` |
| `*.xhat` | Q4.11 | LayerNorm 통계 앞단 (mean/var/rsqrt) |
| `*.norm.out` | Q4.11 | `LN_Affine` |
| `*.norm.int8` | int8 | Q4.11 → int8 requant |
| `*.fc1.q411` | Q4.11 | INT32 → Q4.11 requant |
| `*.gelu.q411` | Q4.11 | GELU LUT |
| `*.gelu.int8` | int8 | Q4.11 → int8 requant |
| `*.res.bf16` | bf16 | bf16 가산기 |
| `head.q69` | Q6.9 | INT32 → Q6.9 requant |
| `softmax.u8` | uint8 | softmax 전체 |

총 22지점입니다. RTL 디버그에서 이게 결정적이었습니다 — `LayerNorm` 에서 tap 정렬을
고치자 `xhat` 이 0 불일치로 떨어지면서 "통계 앞단은 완전히 맞다" 가 확정됐고, 남은
오차가 affine 이후로 좁혀졌습니다.

보드에서도 같은 지점에서 끊어 대조합니다 (`fpga_nl` 모드 A).

---

## 9. 실행

```bash
EVT=/hai/home/sgh/.conda/envs/evt_new/bin/python

$EVT train.py               # fp32 학습    → ffn_fp32.pt   98.50 %
$EVT quantize.py            # 정수 변환    → ffn_int8.pt   98.48 %
$EVT export_rtl.py          # LUT + tap + 유닛벡터 → rtl_export/
$EVT export_rtl.py --n 64   # 배치 크기 바꾸기
```

`data.py` 는 `linear_script/data/` 의 MNIST 를 재사용합니다 (재다운로드 없음).
전처리는 **ToTensor 만** — 픽셀이 `[0,1]` 이면 첫 Linear 의 activation scale 이
`1/127` 로 딱 떨어져 RTL 입력 코드가 곧 `round(pixel·127)` 이 되고, 골든 벡터를 손으로
검산할 수 있습니다. 정규화는 첫 Linear 의 weight/bias 로 흡수되므로 정확도 손해가
없습니다. augmentation 이 없어 **결정론적**이라 캘리브레이션이 재현됩니다.

---

## 10. 파일

| | |
|---|---|
| `model.py` | `FFNNet` — Pre-LN FFN 스택 (attention 뺀 EvT 블록) |
| `data.py` | MNIST 로더 (`linear_script/data/` 재사용) |
| `train.py` | fp32 학습 → `ffn_fp32.pt` |
| `formats.py` | **비트 단위 포맷 primitive** — 여기서 정한 것이 곧 RTL 스펙 |
| `quantize.py` | 캘리브레이션 · 페이로드 · 정수 추론 → `ffn_int8.pt` |
| `export_rtl.py` | LUT · 골든 tap 22개 · 유닛 벡터 · `manifest.json` → `rtl_export/` |

`formats.py` 는 `quantization/quant_lib/`(fp32 컨테이너에 값을 스냅하는 시뮬 스타일)와
달리 **전부 정수 코드로** 계산합니다. 애매한 곳이 없어야 골든과 하드웨어가 비트 단위로
같아집니다.

---

## 11. 이어지는 곳

| | |
|---|---|
| [`04_basic_rtl/fpga_nl/`](../../../04_basic_rtl/fpga_nl/) | 이 골든으로 검증한 ZCU102 가속기 |
| `04_basic_rtl/{Gelu,Bf16,LayerNorm,Softmax}/` | 유닛 RTL + 테스트벤치 |
| `quantization/hw_flow.md` | 원 포맷 명세 |
| `linear_script/` | int8 경로 (GEMM · requant · ReLU · argmax) |
