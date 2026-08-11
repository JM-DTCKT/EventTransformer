# linear_script — RTL 브링업용 MLP

`04_basic_rtl` 의 세 블록(`Mac_OS`, `Requant`, `ReLU`)이 **학습된 망을 제대로
돌리는지** 확인하기 위한 최소 네트워크입니다. Linear 와 ReLU 만 있어서 그 세
블록으로 정확히 덮이고, attention·LayerNorm·GELU·softmax 가 없으니 정수
데이터패스에 남는 것은 **INT8 · INT32 · M/shift** 뿐입니다.

```
Linear   →  Mac_OS/Linear_Top          INT8 × INT8 → INT32 GEMM
+ bias   →  Requant_Int (USE_BIAS=1)   DSP48E2 프리애더
requant  →  Requant_Int                sat((acc·M + 2^(sh-1)) >> sh)
ReLU     →  requant 에 융합             UNSIGNED_OUT=1  → [0,127]
마지막    →  argmax(acc·M)              softmax 없음
```

---

## 1. 망과 데이터셋

**MNIST** (`data/` 로 자동 다운로드, 학습 60k / 테스트 10k).
전처리는 `ToTensor` 만 — 픽셀이 `[0,1]` 이라 첫 Linear 의 activation step 이
`1/127` 로 딱 떨어지고, RTL 입력 코드가 곧 `round(pixel·127)` 이 됩니다.
정규화는 첫 Linear 의 weight/bias 로 흡수되므로 정확도 손해가 없고,
augmentation 이 없어 **캘리브레이션이 결정론적**입니다.

**MLP `784 → 128 → 64 → 10`** (파라미터 109,386개 = fp32 427 KiB)

형상을 이렇게 고른 이유 — `Mac_OS` 의 경로를 골고루 밟게:

| | 값 | 밟는 경로 |
|---|---|---|
| 모든 차원 | ≤ 1024 | `MAX_IN`/`MAX_OUT`/`M_MAX` 안 |
| K = 784 | 32의 배수 아님 (24.5 타일) | **입력 부분 타일** |
| 마지막 out = 10 | < N = 32 | **출력 부분 타일** |
| 배치 M = 32 | = N | 온전한 타일 |

---

## 2. 실행

```bash
conda activate evt_new
cd EventTransformer/linear_script

python train.py                  # fp32 학습 → mlp_fp32.pt   (~40초, GPU)
python quantize.py               # 캘리브레이션 + 패킹 + 정수 평가 → mlp_int8.pt
python export_rtl.py             # RTL 상수 + 골든 벡터 → rtl_export/
```

---

## 3. 결과

| | 정확도 |
|---|---|
| fp32 | **97.43 %** |
| 정수 데이터패스 | **97.42 %** |
| 변화 | **−0.010 pp** (= 테스트 샘플 1개, n=10,000) |

레이어별 상수 (실측):

| 레이어 | 형상 | `sh` | max\|acc\| | max\|b_int\| | 소비자 |
|---|---|---|---|---|---|
| `0` | (128, 784) | 41 | 680,307 | 15,549 | L1 `a_mem` (int8), ReLU 융합 |
| `2` | (64, 128) | 39 | 54,081 | 1,221 | L2 `a_mem` (int8), ReLU 융합 |
| `4` | (10, 64) | 41 | 37,807 | 148 | **argmax** |

- `max|acc|` = 680,307 → **20비트**. INT32 누산기에 충분하고, `acc·M` 은
  20b × 31b + 부호 = **52비트** 라 `Requant_Int` 의 폭 가정과 맞습니다.
- **ReLU 융합이 분리 경로와 10,000/10,000 일치** — `04_basic_rtl/ReLU` 의
  `tb_relu_fused` 가 RTL 에서 증명한 등가성이 실제 망에서도 성립합니다.

---

## 4. `rtl_export/` — RTL 이 읽을 것

전부 `$readmemh` 형식(2의 보수 hex, 한 줄에 하나)입니다.

### 상수

| 파일 | dtype | 크기 | 내용 |
|---|---|---|---|
| `L<i>_W.hex` | int8 | `E_out·E_in` | `w_int[c][k]`, 행 우선 → `Mac_OS.b_mem` (전치 주의) |
| `L<i>_B.hex` | int32 | `E_out` | folded bias → `Requant_Int.bias` |
| `L<i>_M.hex` | int32 | `E_out` | requant 곱수 → `Requant_Int.mult` |
| `manifest.json` | — | — | `shift`, step, `unsigned_out`, 소비자, 파일 매핑 |

### 골든 벡터 (테스트 이미지 32장)

| 파일 | dtype | 검증 대상 |
|---|---|---|
| `L<i>_X.hex` | int8 | 그 레이어의 입력 코드 → `Mac_OS.a_mem` |
| `L<i>_ACC.hex` | int32 | **`Mac_OS` + bias** 가 내야 할 누산기 |
| `L<i>_OUT.hex` | int8 | **`Requant_Int`(ReLU 융합)** 이 내야 할 출력 |
| `argmax.hex` | int32 | 최종 클래스 |
| `labels.hex` | int32 | 정답 (참고) |

### 왜 이게 중요한가

지금까지의 단위 TB 는 **레퍼런스를 TB 안에서 다시 계산**했습니다. 폭·파이프라인·
부호·포화는 검증되지만, "식 자체를 잘못 옮겼을" 위험은 남습니다.
이 골든 벡터는 **학습된 모델에서 나온 독립 출처**라 그 구멍을 메웁니다.

RTL 쪽 사용 예:

```verilog
reg signed [7:0]  x   [0:31][0:783];
reg signed [31:0] acc_exp [0:31][0:127];
initial begin
  $readmemh("rtl_export/L0_X.hex",   x_flat);
  $readmemh("rtl_export/L0_ACC.hex", acc_exp_flat);
  // Mac_OS 로 계산 → acc_exp 와 대조
  // Requant_Int(USE_BIAS=1, UNSIGNED_OUT=1) → L0_OUT.hex 와 대조
end
```

> `Mac_OS` 는 `C = A·B` 이고 `B` 를 전치하지 않으므로, `Y = X·Wᵀ` 를 계산하려면
> 호출측이 `b_mem[k][n] = w_int[n][k]` 로 넣어야 합니다 (`Mac_OS/README` §0).

---

## 5. 파일

| 파일 | 역할 |
|---|---|
| `model.py` | MLP 정의 (Linear + ReLU 만) |
| `data.py` | MNIST 로더 (자동 다운로드) + 정확도 |
| `train.py` | fp32 학습 → `mlp_fp32.pt` |
| `quantize.py` | 캘리브레이션 · 패킹 · 정수 추론 — **RTL 과 한 줄씩 대응하는 골든 모델** |
| `export_rtl.py` | 상수 + 골든 벡터 → `rtl_export/` |

`quantize.py` 는 `quantization/quant_lib` 를 쓰지 않고 **자립**입니다. RTL 을
검증하는 게 목적이라 파이썬이 RTL 의 명백한 거울이어야 하고, EvT 전용 로직
(attention·LayerNorm·pos-enc)이 섞이면 대조가 어려워집니다.
