# EvT 정수 데이터패스 (ZCU102 타깃)

`quant_lib/hw_quant.py`가 **실제로 무엇을 계산하는지**를 유닛 단위로 적은 문서입니다.
데이터패스에 남아 있는 자료형은 **INT8**(곱셈기 피연산자), **INT32**(누산기 및 folded
bias), **16/24/32비트 고정소수점 Qm.n**(scale, 비선형 유닛 입출력, LayerNorm affine
파라미터, latent memory 테이블) 뿐입니다.

`flow.md`는 이전의 fake-quantization 흐름(bias·positional encoding·scale·비선형 주변이
전부 fp32였던 상태)을 설명합니다. 이 문서는 그것을 대체한 내용입니다.

표기: `s_x` = activation scale(텐서당 스칼라), `s_w[c]` = 출력 채널 `c`의 weight scale,
`Qm.n` = 16비트 2의 보수 정수를 `code * 2^-n`으로 해석 (부호 1비트 + 정수 `m`비트 +
소수 `n`비트, `m + n = 15`). `m`이 음수면 상위 비트가 항상 0이라 저장하지 않는다는
뜻이고, `n`이 음수면 LSB의 가중치가 1보다 크다는 뜻입니다.

---

## 0. 이 문서의 숫자가 어디까지 "진짜"인가

정확도 수치는 **저장된 `.pt`에서 모델을 다시 불러와** 아래 데이터패스로 forward를
돌린 결과입니다. 시뮬레이터가 주장대로 동작하는지는 `audit_inference_datapath.py`가
실제 배치를 흘리면서 텐서 단위로 계측해 확인합니다. 실측 결과(DVS128_10,
배포 설정 `deploy_gelu_q411`, 1,755회 GEMM 호출):

| 확인 항목 | 결과 |
|---|---|
| 모든 MAC 피연산자 `x_int`, `w_int`가 [-128,127]의 정확한 정수 | ✅ |
| 모든 누산기가 정확한 정수 | ✅ |
| 최대 \|누산기\| | 252,240 (INT32 한계 2.1e9, float32 정확 정수 한계 1.68e7) |
| 최대 \|folded INT32 bias\| | 5,850 (13비트) |
| 계측된 MAC 종류 | `linear_K128/K144/K160`, `attn_QK^T_K32`, `attn_AV_K94/K97/K112` |
| 비선형 유닛에 들어가는 값이 선언된 Qm.n / INT8 격자 위에 정확히 놓임 | ✅ (전 사이트) |

### 시뮬레이터가 하드웨어와 다른 지점 (숨기지 않고 적음)

1. **비선형 함수 자체는 PyTorch의 부동소수점 구현**입니다 (`F.gelu`, `torch.softmax`,
   `torch.rsqrt`, LayerNorm의 mean). 이건 이상적인 LUT/CORDIC을 모델링한 것이고,
   질문에서 제외하신 "비선형 중간 과정"에 해당합니다. **입력과 출력은 선언된
   Qm.n 격자에 정확히 스냅됩니다.**

2. **요소 사이의 중간값은 fp32 컨테이너에 담겨 흐릅니다.** 값 자체는 정수/Qm.n
   격자 위에 있지만, 두 requantization 지점 사이의 residual 덧셈이나 scale 곱은
   fp32로 계산됩니다. 구체적으로:
   - `acc * (s_x * s_w[c])` — 실제 HW는 int32 × int16 → int64 + shift. 시뮬레이터는
     fp32 곱셈(가수 24비트).
   - residual 덧셈 (`z + z_att`, `latent_vectors + inp_q`, MLPBlock의 `add_x_input`) —
     HW는 고정소수점 덧셈.
   - `LatentEmbsCompressor`의 `z.mean(dim=0)` (latent 96개 평균) — HW는 int 합 +
     `1/96` Q상수 곱.

   이 값들은 **전부 다음 MAC 입력에서 INT8로, 또는 다음 비선형 입력에서 Qm.n으로
   다시 스냅**되므로 구조적으로는 정수 데이터패스가 맞습니다. 다만 위 지점들의
   fp32 반올림(상대 2^-24)이 정확한 고정소수점 반올림과 미세하게 다릅니다.

3. **float32는 24비트 가수만 있어서, 32비트 Q포맷을 온전히 담지 못합니다.**
   크기 `M`, 소수부 `n`비트인 Q값은 `log2(M) + n + 1` 비트가 필요합니다. 감사
   결과 13개 사이트가 최대 **32비트**를 요구합니다(주로 누적되는 LayerNorm 입력).
   즉 Q32 설정의 측정치는 실제로는 "**24비트 이상**"을 잰 것이고, 진짜 32비트
   하드웨어는 그보다 **더** 정확하므로 보고된 정확도는 **하한**입니다.
   → 이 때문에 배포 설정은 LayerNorm 경로를 **24비트**로 둡니다. 24비트는 float32가
   **정확히** 표현할 수 있어서 그 행은 하한이 아니라 정확한 값입니다.

4. **모델 입력(`polarity`)은 fp32 이벤트 데이터**이고, 첫 Linear
   (`event_projection`)의 입력 양자화에서 INT8이 됩니다. 즉 fp32는 칩 경계 바깥에만
   존재합니다.

### 캐비엇이 하나도 남지 않는 설정: `deploy_gelu_q411` (배포 설정)

위 3번(float32가 32비트 Q포맷을 못 담음)과 ⑤(캘리브레이션 범위 포화)를 둘 다
없앤 설정입니다. LayerNorm 경로를 **24비트**로 두고(24비트 Q값은 float32가 정확히
표현합니다) 소수부에 **2비트 가드 밴드**를 주며, pos-enc K-split은 쓰지 않고 GELU는
**Q4.11 하나로 고정**합니다. 세 데이터셋 실측 감사 결과:

| | DVS128_10 | DVS128_11 | ASL_DVS |
|---|---|---|---|
| MAC 피연산자가 [-128,127] 정확한 정수 | ✅ | ✅ | ✅ |
| 누산기가 정확한 정수 | ✅ | ✅ | ✅ |
| 최대 \|누산기\| | 252,240 | 205,782 | 131,685 |
| 최대 \|folded INT32 bias\| | 5,850 | 6,443 | 14,682 |
| 모든 사이트가 선언된 격자 위 | ✅ | ✅ | ✅ |
| **24비트 초과를 요구하는 사이트** | **0** | **0** | **0** |
| **LayerNorm 포맷 포화** | **0** | **0** | **0** |

(포화로 잡히는 것은 LayerNorm이 아니라 ReLU의 INT8 입력 1~2건(8,800만 개 중 1개
수준, 모든 INT8 설계에 있는 일반적인 outlier 클리핑)과, ASL_DVS 첫 GELU의 Q4.11
클리핑 0.64%입니다 — 후자는 §3 ⑥에서 다룹니다.)

즉 이 설정에서는 **"비선형 내부를 빼면 INT8 / INT32 / 고정소수점만 쓴다"가
시뮬레이터 한계 없이 그대로 성립**합니다. 정확도는 99.960% / 97.348% / 95.833%.

---

## 1. Linear / GEMM 유닛

모든 `nn.Linear`와 묶여 있는 `Wq|Wk|Wv` projection:

```
x_int[k]   = clamp(round(x[k] / s_x), -128, 127)          INT8
w_int[c,k] = clamp(round(W[c,k] / s_w[c]), -128, 127)     INT8      (오프라인)
acc[c]     = SUM_k x_int[k] * w_int[c,k]                  INT32
acc[c]    += b_int[c]                                     INT32     <-- ①
y[c]       = acc[c] * (s_x * s_w[c])                      Qm.n x Qm.n
```

**① INT32 bias.** `b_int[c] = round(b[c] / (s_x * s_w[c]))`를 오프라인에서 계산해
INT32로 저장합니다. bias가 fp32 곱셈 뒤의 fp32 덧셈이 아니라 누산기 위의 정수
덧셈 하나가 되고, rescale 곱수는 GEMM이 어차피 필요로 하는 `s_x * s_w[c]`
그대로입니다. 비용은 누산기 LSB의 절반 이하 오차 (`test_hw_quant.py`에서 검증).

실측 `max |b_int|`는 6,701 (DVS128_10) / 8,228 (DVS128_11) / 16,342 (ASL_DVS) —
최악 14비트라 INT32는 넉넉하고, 누산기가 float32의 정확 정수 범위를 벗어나지
않습니다 (`int8_linear`이 런타임에 assert).

`s_x * s_w[c]`는 int16 × int16 → int32 곱셈에 shift 하나입니다. 두 Q포맷의 shift
값이 더해집니다.

여기서 reduction 길이는 64–160이고 `127 * 127 * 160 = 2.6e6`이라 22비트에 들어가므로
INT32 누산기는 9비트 여유가 있습니다.

## 2. Attention

헤드당 (`head_dim = 32`, 4 heads, `embed_dim = 128`):

```
Q,K,V      = 위 GEMM, s_x = 캘리브레이션된 projection 전 Q/K/V scale
bias_k/v   = projection 후 K/V 도메인의 INT32 addend
scores_int = SUM_d Q_int[d] * K_int[d]                    INT32   (d = 32)
scores     = scores_int * (s_q * s_k) * INV_SQRT_D        Qm.n    <-- ②
attn       = softmax(scores)                              Qm.n -> Q1.14
attn_int   = clamp(round(attn * 127), 0, 127)             UINT8 (항상 >= 0)
out_int    = SUM_j attn_int[j] * V_int[j]                 INT32   (j = key 수)
out        = out_int * (1/127 * s_v)                      Qm.n
out_proj   = 위 GEMM, s_x = 캘리브레이션된 out_proj 입력 scale
```

**② `1/sqrt(head_dim)`도 fp32 나눗셈이 아니라 Q포맷 상수**로, score rescale 곱수에
흡수됩니다.

`softmax`는 고정 포맷 Qm.n 값을 받아(cross-attention은 보통 **Q5.10**, latent
attention은 **Q7.8** — `hw_quant_distributions.csv` 참조) **Q1.14**를 내보냅니다.
LUT/구간선형 exp 유닛이 설계 대상 입력 포맷을 하나로 고정할 수 있습니다. 출력
범위는 해석적으로 [0,1]이라 INT8 step은 캘리브레이션 없이 1/127로 고정입니다.

`attn_int`는 음수가 없으므로 `attn @ V` 어레이는 unsigned × signed 곱셈기를 쓸 수
있습니다.

## 3. 비선형 유닛

각 유닛의 입출력은 캘리브레이션 패스에서 정해진 각자의 Qm.n 격자 위에 놓입니다
(`RequantSite`, `hw_quant_distributions.csv`):

| 유닛 | 입력 포맷 | 출력 포맷 | 비고 |
|---|---|---|---|
| **GELU** | **Q4.11 (고정)** | **Q4.11 (고정)** | 네트워크 전체가 **LUT 하나** — ⑥ 참조 |
| LayerNorm | ③ 참조 | Q2.13 – Q4.11 | 정규화 중간값도 Qm.n |
| softmax | Q5.10 / Q6.9 / Q7.8 | Q1.14 | 블록마다 다름, §2 참조 |
| log_softmax | Q5.10 | — | argmax 불변, 출력은 재양자화 안 함 |
| **ReLU** | **INT8** | **INT8** | 부호만 보면 되므로 8비트로 충분 |

### ⑥ GELU는 Q4.11 하나로 고정 (`gelu_frac_bits=11`)

사이트마다 포맷을 따로 고르면(Q3.12 / Q4.11이 섞임) LUT을 여러 개 만들거나 포맷을
전환해야 합니다. 배포 설정은 **모든 GELU 입출력을 Q4.11로 못박습니다**
(1 부호 + 4 정수 + 11 소수 = 16비트, 범위 ±16, LSB 2⁻¹¹ = 0.000488). 실측 결과
정확도는 오히려 **좋아졌습니다** — 세 데이터셋 모두 자동 포맷보다 같거나 위:

| | ASL_DVS | DVS128_10 | DVS128_11 |
|---|---|---|---|
| 사이트별 자동 포맷 (`deploy_nosplit`) | 99.955% | 97.348% | 95.833% |
| **Q4.11 고정 (`deploy_gelu_q411`)** | **99.960%** | **97.348%** | **95.833%** |

**단, ASL_DVS의 첫 GELU에서 클리핑이 있습니다.** `event_projection`의 GELU 입력이
최대 63.45까지 올라가서 Q4.11의 ±16을 넘고, 원소의 **0.64%가 잘립니다**
(DVS128은 최대 13.3으로 클리핑 0%). 그런데 정확도는 오히려 올랐습니다. 이유:

- GELU는 x > 4에서 사실상 항등함수라, 큰 값을 자르는 것은 activation clipping과
  같습니다.
- 그 출력은 곧바로 concat되어 `preproc_block_events`의 INT8 양자화기를 통과하는데,
  그 step은 0.138이고 **INT8 자체가 17.6에서 잘립니다**. 즉 Q4.11이 16에서 자르는
  것과 원래 17.6에서 잘리던 것의 차이뿐입니다.

클리핑을 0으로 만들고 싶으면 Q6.9(±64)를 쓰면 되지만 해상도를 2비트 잃고, 실측
정확도는 Q4.11이 더 좋으므로 **Q4.11 유지를 권합니다.**

세 모델 전체에서 비선형 데이터패스가 쓰는 포맷은 GELU **Q4.11 하나**, softmax 입력
Q5.10 / Q6.9 / Q7.8, LayerNorm 출력 Q2.13 – Q4.11, LayerNorm 입력 Q16.7(24비트)
입니다.

LayerNorm 계산:

```
x        = requant(acc)                                   Qm.n     <-- ③
mu       = mean(x)
centered = requant(x - mu)                                Qm.n     <-- ④
var      = mean(centered^2)
xhat     = requant(centered * rsqrt(var + eps))            Q3.12/Q4.11 (16b)
y        = requant(xhat * gamma + beta)                    Q3.12/Q4.11 (16b)
```

`gamma`/`beta`는 Q포맷 int16으로 저장합니다(고정소수점 유닛의 상수이므로, fp32로
두면 fp32 곱셈기가 데이터패스로 되돌아옵니다). latent memory 테이블
`memory_vertical`도 같은 이유로 Q포맷 int16입니다 — 곧바로 `layer_norm_1`에
들어갑니다.

### ③ LayerNorm 입력에는 런타임 exponent가 필요합니다 — 단, 타임스텝이 여러 개일 때만

EvT는 `latent_vectors = inp_q + latent_vectors`를 타임스텝마다 누적합니다. 그래서
`layer_norm_1` / `layer_norm_att` / `latent_attentions.*.layer_norm_x` /
`proc_embs_block.layer_norm`에 들어가는 residual stream이 첫 스텝에서 마지막
스텝까지 여러 자릿수만큼 커집니다.

| 모델 | 타임스텝 | residual-stream LN 호출 간 진폭비 | 나머지 사이트 |
|---|---|---|---|
| DVS128_10 | 20 | 192× – **32,768×** | ≤ 2.34× |
| DVS128_11 | 20 | 18× – **32,768×** | ≤ 1.98× |
| ASL_DVS | **1** | 1.0× (누적 자체가 없음) | ≤ 4.67× |

DVS128_10의 `cross_attention.layer_norm_1.in`은 스텝당 거의 정확히 2배씩 커집니다
(저장된 site book의 `per_call_max`):

```
step  0     1     2     3     4     5     6     7     8     9
max  0.7  12.6  15.1  18.9  20.0  23.4  25.8  47.4  94.8  189.5
step 10    11    12    13     14      15      16      17      18      19
max 379   758  1516  3033   6065   12130   12133   12136   12137   12138
```

static Q16 하나로 잡으면 `Q14.1`(LSB = 0.5)이 되어 step 0에 레벨이 2개만 남습니다.
빠져나갈 방법이 네 가지 있고 **네 가지 모두 측정했습니다**
(`hw_quant_layernorm_options.png`):

| 옵션 | requantizer의 RTL 비용 | DVS128_10 | DVS128_11 | ASL |
|---|---|---|---|---|
| static Q16 | 상수 shift | −0.758 pp | −0.347 pp | 0 |
| Q16 + 런타임 exponent | 128-way max tree + leading-zero count + barrel shifter, **게다가 shift 전에 벡터 전체를 버퍼링해야 함** | +0.379 pp | 0 | 0 |
| Q16, 타임스텝별 포맷 | 스텝 카운터로 인덱싱하는 20×4비트 ROM + barrel shifter | 0 | 0 | 0 |
| **static Q32** | **32비트 레지스터로 상수 shift — 추가 로직 없음, 추가 pass 없음** | **0** | **0** | **0** |

(INT8 reference 대비 pp. 테스트 샘플 1개 = 0.379 / 0.347 / 0.005 pp.)

셋 다 정확도가 같으므로 하드웨어 비용으로 고르면 **static Q32가 이깁니다**. 그리고
residual stream은 **어차피 INT32 누산기에서 나옵니다** — 16비트로 좁힌 것이
인위적이었던 것이고, Q32 레지스터로 rescale하면 같은 곱셈기에 shift 값만 달라지므로
추가 로직이 문자 그대로 0입니다. 선택된 Q32 포맷은 누적되는 LayerNorm이 `Q14.17`,
`proc_embs_block`이 `Q17.14`입니다.

### ④ variance 경로도 같은 폭이 필요합니다

`(x - mean)`은 같은 스텝별 진폭 변화를 그대로 물려받습니다. 그래서 centered 값을
16비트로 좁혀서 제곱하면 Q32 입력으로 회복한 것을 그대로 다시 잃습니다
(`all_ln32_ctr16`: −0.758 / −0.347 pp, static Q16과 동일). 32비트면 다시 깨끗합니다
(`all_ln32_ctr32`가 `all_ln32`와 동일).

정리하면 LayerNorm 유닛은: **입력 레지스터·centered 값·제곱기를 모두 같은 폭으로**
두고, 정규화 **이후**에야 16비트로 내려옵니다(`.hat`, `.out`).

**필요한 최소 폭은 24비트입니다.** 32비트는 안전하지만 float32 시뮬레이터가 온전히
담지 못해 하한만 잽니다(§0의 3번). 24비트는 float32가 정확히 표현하므로 그 행은
정확한 값이고, 실측 결과 32비트와 사실상 같습니다(ASL 99.940% vs 99.955% = 20,157개
중 3개 차이). 따라서 RTL은 **24비트 입력 레지스터, 24×24→48 제곱기, 약 31비트
mean/variance 누산기**로 충분하고, 여유를 두고 싶으면 32비트로 올리면 됩니다.

### ⑤ 캘리브레이션 범위에 가드 밴드가 필요합니다

`audit_inference_datapath.py`가 잡아낸 문제입니다. 학습셋 100배치로 캘리브레이션한
LayerNorm 입력 포맷이 `frac_bits = 16`(범위 ±32,768)으로 정해졌는데, **테스트셋의
residual stream은 48,523까지 올라갑니다.** 고정소수점 포맷은 완만히 나빠지지 않고
**포화(saturate)** 하므로, 8개 사이트에서 원소의 0.00034%가 클리핑됩니다.

지금은 정확도에 영향이 없지만(백만 개 중 3개), 캘리브레이션 데이터에 의존하는
위험 지점입니다. 대책은 소수부 비트를 1–2비트 줄여 표현 범위를 2–4배 넓히는
것입니다(`ln_headroom_bits`). 조용한 스텝에서 정밀도를 1–2비트 잃는 대신 클리핑이
사라집니다.

### 런타임 exponent를 GELU/softmax에는 쓸 수 없는 이유

둘 다 scale-invariant가 아니라서, LUT은 입력 포맷이 고정되어 있어야 합니다.
LayerNorm만 scale-invariant이므로 ③의 방법들이 성립합니다. 그리고 GELU/softmax는
애초에 필요가 없습니다 — 호출 간 진폭비가 2.4배 미만입니다.

## 4. 저장되는 파라미터

| 텐서 | 자료형 | 이유 |
|---|---|---|
| Linear / in_proj / out_proj weight | INT8 | 출력 채널별 symmetric |
| weight scale | int16 + 레이어당 shift 1개 | 레이어별 공유 exponent — int16 곱수 벡터 하나 + shift 하나, 곧 rescale 유닛 그 자체 |
| activation scale | int16 + shift | Linear 입력마다 하나, 여기에 attention Q/K/V/out_proj scale |
| bias (`bias_k`/`bias_v` 포함) | INT32 | ① 참조 |
| positional encoding | INT8 + scale 1개 | 아래 참조 |
| LayerNorm `gamma`/`beta` | int16 Qm.n | 고정소수점 유닛의 상수 |
| `memory_vertical` | int16 Qm.n | LayerNorm에 직접 들어감 |

### positional encoding을 INT8로 저장해도 손해가 없는 이유

이 테이블은 `event_projection` 출력에 concat되고, 그 결과가
`preproc_block_events`의 입력 양자화기에서 step 0.120(DVS128) / 0.138(ASL)로 INT8
재양자화됩니다 — concat된 텐서의 최댓값이 15–18이고 projection 쪽이 지배하기
때문입니다. pos-enc 테이블 자체의 INT8 step은 0.0099–0.0060으로, **어차피 곧
재양자화될 step보다 12배(DVS128)–23배(ASL) 더 촘촘합니다.** fp32로 두어도 다음
레이어가 즉시 버리는 정밀도를 사는 것이고, 체크포인트에서 weight 다음으로 가장 큰
텐서입니다(DVS128 28,224개, ASL 76,800개 = ASL 모델 828 KiB 중 300 KiB).

### 그런데 인코딩이 손상되는데 왜 정확도가 안 떨어지나

`analyze_pos_encoding.py` 실측 (`pos_encoding_analysis.csv`):

| | DVS128_10 | DVS128_11 | ASL_DVS |
|---|---|---|---|
| fp32에서 `pos_encoding = 0`으로 두면 | 39.4% (**−58.7 pp**) | 31.9% (**−63.9 pp**) | 22.2% (**−77.7 pp**) |
| 해당 레이어 출력 norm 중 pos-enc 기여 | 58% | 60% | 46% |
| pos-enc가 실제로 쓰는 INT8 레벨 수 | 27 / 256 | 36 / 256 | **12 / 256** |
| pos-enc SQNR, 공유 scale | 28.7 dB | 30.7 dB | **17.6 dB** |
| pos-enc SQNR, 분리 scale | 48.3 dB | 47.7 dB | 44.1 dB |
| **위치 고유 식별률** | **100%** | **100%** | **100%** |
| 분리 마진 (최근접 코드쌍 거리 / 양자화 노이즈) | 21.5× | 20.0× | **2.33×** |

모델은 인코딩에 의존합니다 — 지우면 59–78 pp 떨어집니다. 공유 scale이 SQNR 관점에서
심하게 망가뜨리는 것도 사실입니다. 하지만 인코딩의 역할은 441–1200개 격자 위치를
**구별 가능하게** 만드는 것이지 정밀한 값을 나르는 게 아니고, 양자화가 64차원 위치
코드를 움직이는 거리는 서로 다른 두 코드 사이 거리보다 훨씬 작습니다. 모든 위치가
여전히 고유하게 식별되며, 그래서 정확도가 움직이지 않습니다.

**봐야 할 것은 SQNR이 아니라 마진입니다.** ASL_DVS가 2.33×인 이유는 위치 격자가 가장
촘촘하고(40×30 = 1200개 코드) 공유 step이 가장 거칠기 때문입니다. 지금은 안전하지만
모델을 한 번만 바꿔도 아닐 수 있습니다 — `downsample_pos_enc`를 줄이거나, 센서를
키우거나, Fourier band를 늘리면 최근접 코드쌍 거리만 줄고 노이즈는 그대로라
실패가 점진적이 아니라 **갑자기** 옵니다.

### 선택지: concat에 scale 두 개 (K-split) — **배포 설정에서는 사용하지 않음**

`split_posenc_scale`은 `cat([projection, pos_enc])`의 두 절반에 각각 INT8 scale을
줍니다. 하드웨어로는 한 레이어의 **K-split GEMM**입니다: K=160 루프를 96에서 쪼개
INT32 부분 누산기 두 개를 두고 rescale 곱수 두 개로 합치거나 — `r = s_B / s_A`가
상수 하나이므로 `acc = acc_A + ((acc_B * R) >> k)` 뒤에 기존 rescale 하나면 됩니다.
누산기 하나와 DSP 하나가 한 레이어에 추가됩니다.

실측: 분리 마진 2.33× → **49.6×** (ASL), 21.5× → 202× (DVS128). 현재 정확도는
테스트 샘플 ±1개 이내로 변화 없습니다. "마진이 이미 충분했다"는 결론과 정확히
일치하므로, **현재 손실의 수정이 아니라 보험**입니다. 배포 설정
(`deploy_gelu_q411`)은 하드웨어를 단순하게 유지하기 위해 이 옵션을 **끕니다**.

**두 절반이 "섞이는" 것과 "섞이면 안 되는 것"의 구분** — Transformer가 의도하는
것은 위치 정보와 content feature가 하나의 embedding으로 융합되는 것이고, EvT는
concat 방식이라 **이 Linear가 바로 그 융합 지점**입니다. 그 융합은 두 설계 모두에서
그대로 일어납니다: 출력 채널 `c`는 언제나 두 절반 모두에서 기여를 받습니다.

```
y[c] = Σ_{k<96} x[k]·W[c,k]  +  Σ_{k≥96} x[k]·W[c,k]        ← 결합법칙, 값은 동일
     = s_A·s_w[c]·accA[c]    +  s_B·s_w[c]·accB[c]
```

하면 안 되는 것은 **LSB가 다른 두 정수 누산기를 공통 단위로 바꾸기 전에 더하는
것**(`accA + accB`)뿐입니다. K-split은 rescale **이후**에 더합니다. 두 설계의 진짜
차이는 "단위를 언제 맞추느냐"입니다:

| | 단위 변환 시점 | pos-enc 입력 해상도 |
|---|---|---|
| 단일 scale (배포 설정) | GEMM **입력**에서 — 두 절반을 같은 INT8 격자에 강제 | INT8 레벨 12–27개 |
| K-split | GEMM **출력**에서 — 누산기 두 개를 각자 rescale 후 합산 | INT8 레벨 256개 |

정확한 산술에서는 둘이 **완전히 같은 `y[c]`** 를 냅니다. 차이는 입력 양자화 오차가
어디에 얹히느냐뿐이고, GEMM을 통과한 뒤에는 융합된 embedding 하나에 scale 하나라
이후 네트워크는 pos-enc가 별도로 있었다는 사실을 알지 못합니다.

공짜처럼 보이는 대안은 **동작하지 않습니다**: pos-enc 테이블을 오프라인에서 `c`배
키우고 대응하는 weight **열**을 `c`로 나누면 곱은 그대로지만, 그 weight 열이 같은
출력 행별 INT8 weight scale 안에서 projection 열보다 21배쯤 작아집니다 — 같은
불균형이 activation에서 weight로 옮겨갈 뿐입니다.

## 5. 전역 Q포맷 하나가 아니라 텐서별 공유 exponent를 쓰는 이유

모델의 모든 scale(weight 채널별 scale + activation scale) 실측: 최대 0.115–0.138,
최소 3.7e-4, 진폭비 156–372배. 전역 포맷 하나로는 **Q-3.18** / **Q-2.17**이 되고
상대오차 최대 0.22–0.91%, 평균 0.053–0.077%가 듭니다. 레이어별 공유 exponent(소수부
20–24비트)는 같은 16비트 저장으로 **최대 6e-5 – 2.5e-4, 평균 약 2e-5**입니다. 각
레이어의 채널별 scale이 내부적으로 1.7–8배밖에 안 벌어지기 때문입니다. rescale
유닛은 어차피 레이어당 shift 값이 필요하므로 레이어별 exponent는 공짜입니다. 둘 다
`hw_quant_distributions.csv`에 있습니다.

scale이 전부 1보다 훨씬 작아서 Q포맷의 정수 비트 수가 음수로 나옵니다(`Q-8.23` =
16비트 코드 × 2^-23). 하드웨어에서는 그냥 고정소수점 곱수 + shift이고, 상위 정수
비트가 항상 0이라 저장하지 않는다는 뜻일 뿐입니다.

## 6. 측정 결과

각 패킹 체크포인트에서 **다시 불러온** 모델의 테스트 정확도입니다
(`hw_quant_summary.csv`, 그림 `hw_quant_accuracy_drop.png`). `reference`는
`int8_w8a8_static+qkv_mac`을 명시적 정수 데이터패스 위에 다시 만든 것으로, 기존에
보고된 수치를 정확히 재현합니다 — 재작성이 바꾸면 안 되는 것을 바꾸지 않았다는
확인입니다.

| config | ASL_DVS (20,157) | DVS128_10 (264) | DVS128_11 (288) | 패킹 크기 |
|---|---|---|---|---|
| fp32 baseline | 99.965% | 98.106% | 95.833% | 2,123 / 1,927 KiB |
| reference (INT8 W/A + INT8 attention MAC) | 99.955% | 97.348% | 95.486% | 828 / 636 KiB |
| ① INT32 bias | 99.945% | 97.348% | 95.486% | 828 / 636 KiB |
| ② INT8 positional encoding | 99.960% | 97.348% | 95.833% | 603 / 553 KiB |
| ③ FX16 scale | 99.955% | 97.348% | 95.486% | 821 / 629 KiB |
| ④ FX16 비선형 I/O, LayerNorm 입력 static Q16 | 99.945% | 96.591% | 95.486% | 797 / 606 KiB |
| ④ᵃ LayerNorm 입력 Q16 + 런타임 exponent | 99.945% | 97.348% | 95.486% | 797 / 606 KiB |
| ④ᵇ LayerNorm 입력 Q16, 타임스텝별 포맷 | 99.940% | 97.348% | 95.486% | 797 / 606 KiB |
| ④ᶜ LayerNorm 입력 static Q32 | 99.950% | 97.348% | 95.486% | 797 / 606 KiB |
| ①+②+③+④ (LayerNorm 입력 static Q16) | 99.955% | 96.591% | 95.139% | **565 / 516 KiB** |
| ①+②+③+④ᵃ | 99.955% | 97.727% | 95.486% | 565 / 516 KiB |
| ①+②+③+④ᵇ | 99.955% | 97.348% | 95.486% | 565 / 516 KiB |
| ①+②+③+④ᶜ | 99.955% | 97.348% | 95.486% | 565 / 516 KiB |
| ①+②+③+④ᶜ, variance 경로 16비트 | 99.955% | 96.591% | 95.139% | 565 / 516 KiB |
| ①+②+③+④ᶜ, variance 경로 32비트 | 99.955% | 97.348% | 95.486% | 565 / 516 KiB |
| ①+②+③+④ᶜ + split pos-enc scale | 99.950% | 97.727% | 95.833% | 565 / 516 KiB |
| ①+②+③+④ + split pos-enc, LayerNorm 경로 24비트 | 99.940% | 97.727% | 96.181% | 565 / 516 KiB |
| ①+②+③+④ + split pos-enc, 24비트 + 2비트 가드 | 99.940% | 97.348% | 95.833% | 565 / 516 KiB |
| ①+②+③+④, split 없음, 24비트 + 가드 (`deploy_nosplit`) | 99.955% | 97.348% | 95.833% | 565 / 516 KiB |
| **위 + GELU를 Q4.11로 고정 (`deploy_gelu_q411`, 배포 설정)** | **99.960%** | **97.348%** | **95.833%** | **565 / 516 KiB** |

### LayerNorm 경로 폭을 좁혀 보면 (전부 2비트 가드 적용)

| LayerNorm 폭 | ASL_DVS | DVS128_10 | DVS128_11 |
|---|---|---|---|
| 16비트 (가드 없음) | 99.955% | 96.591% | 95.139% |
| 18비트 | 99.940% | 97.348% | **93.403%** ← 무너짐 |
| 20비트 | 99.940% | 97.727% | 95.139% |
| 22비트 | 99.950% | 97.727% | 95.486% |
| **24비트** | **99.940%** | **97.348%** | **95.833%** |

§3 ⑤의 "동적 범위에 16.1비트가 필요하다"는 계산과 일치합니다 — 18비트는 해상도에
쓸 비트가 2비트도 안 남아 DVS128_11에서 −2.4 pp로 무너지고, 20–22비트에서 회복해
24비트에서 안정됩니다. **24비트가 여유를 가진 최소 폭**입니다.

DVS128 두 열은 조심해서 읽어야 합니다. 테스트셋이 264개와 288개라 **샘플 1개가
0.379 pp / 0.347 pp**입니다. 표의 모든 DVS128 차이는 1–2 샘플입니다. 그보다 미세한
것을 분해할 수 있는 열은 20,157개인 ASL_DVS 하나이고, 거기서는 네 변경을 다 합쳐도
INT8 reference 대비 **0.000 pp**입니다(양쪽 다 99.955%).

테스트셋 커버리지: 각 `test/` 폴더의 264 / 288 / 20,160개 파일 중 264 / 288 /
20,157개를 평가했습니다(DVS128 수치는 `classes_to_exclude` 적용 후). 빠진 ASL 3개는
EvT 자신의 collate가 버리는 것으로(`data_generation.py`, 유효 패치가 없는 샘플),
fp32 baseline과 모든 양자화 케이스에 동일합니다. val loader는 `validation=True`
(augmentation 없음) + `shuffle=False`라 여기 있는 모든 수치가 결정론적입니다.

크기: fp32 체크포인트의 26.6–26.8%이고, `verify_hw_model_sizes.py`가 아키텍처로부터
독립적으로 계산한 기대값과 그룹별로 일치하는지 확인합니다. `.pt` 파일에는 payload
위에 torch zip/pickle 컨테이너가 약 53 KiB 붙습니다. FPGA용 raw binary는 payload
수치입니다.

## 7. 실제 FPGA에 올릴 때: `.pt`의 값을 어디에 넣는가

`export_fpga_constants.py`가 패킹된 `.pt`를 읽어 RTL이 바로 쓸 수 있는 정수 상수와
raw binary로 바꿔 줍니다. **그리고 그 정수 공식이 시뮬레이터 출력과 일치하는지
실제 배치로 검증**합니다.

```
python export_fpga_constants.py --dataset DVS128_10 --config deploy_gelu_q411
→ fpga_export/DVS128_10__deploy_gelu_q411/{manifest.json, *.bin, verification.csv}
```

### 7.1 `.pt` 안의 자료 구조

| payload 키 | 내용 | 자료형 / 모양 |
|---|---|---|
| `layers[<k>.weight]['q']` | GEMM weight | `int8 [E_out, E_in]` |
| `layers[...]['scale']` | weight scale, **레이어당 shift 하나 공유** | `{codes: int16[E_out], frac_bits: int}` |
| `layers[...]['x_scale']` | 입력 activation scale | `{codes: int16[1], frac_bits: int}` |
| `layers[...]['x_scale_b']`, `['split_at']` | K-split GEMM의 두 번째 입력 scale과 분할 지점 | `int16[1]`, `int` |
| `layers[...]['bias']['q']` | folded bias | `int32 [E_out]` |
| `layers[<mha>.in_proj_weight]` | `Wq\|Wk\|Wv`가 묶인 것 | `int8 [3E, E]`, scale `int16[3E]` |
| `... ['bias_k'], ['bias_v']` | K/V에 붙는 추가 토큰 | `int32 [E]` |
| `pos_encoding['q']`, `['scale']` | 위치 인코딩 LUT | `int8 [H, W, 64]` + scale 1개 |
| `fx_params[<ln>.weight/.bias]` | LayerNorm gamma/beta | `{codes: int16[E], frac_bits}` |
| `fx_params['backbone.memory_vertical']` | latent 초기값 테이블 | `int16 [96, 128]` |
| `attention_scales[...]` | Q/K/V/out_proj/softmax scale | 각 `{codes: int16[1], frac_bits}` |
| `sites[<name>]` | 비선형 경계의 Q포맷 | `{kind, total_bits, frac_bits}` 또는 `{kind:'int8', scale}` |

**scale 하나 읽는 법**: `값 = codes * 2^(-frac_bits)`. 예를 들어
`cross_attention.linear1`의 `x_scale = {codes:[22141], frac_bits:19}` →
`22141 / 2^19 = 0.042232...`

### 7.2 GEMM 연산기에 넣는 것

```
입력  A : x_int[k]        int8   ← 앞 단계가 이미 int8 코드로 내보냄
입력  B : w_int[c,k]      int8   ← <layer>.W.int8.bin,  [E_out][E_in] row-major
입력  C : b_int[c]        int32  ← <layer>.B.int32.bin
출력    : acc[c] = SUM_k x_int[k]*w_int[c,k] + b_int[c]     int32
```

`x_int`를 만들기 위한 나눗셈은 **필요 없습니다.** 앞 단계의 requantizer가 이미
그 layer의 `input.step`에 맞춘 int8 코드를 내보내도록 상수가 계산되어 있습니다.
manifest의 `input.step`은 참고값(디버깅·검증용)입니다.

reduction 길이는 64–160, 최대 관측 `|acc|`는 252,240 (18비트) → **INT32 누산기면
13비트 여유**가 있습니다.

### 7.3 Requantizer에 넣는 것

**모든 requantizer가 곱셈기 하나 + shift 하나입니다.**

```
out_code[c] = sat( (acc[c] * M[c] + (1 << (k-1))) >> k )
```

| 입력 | 어디서 | 자료형 |
|---|---|---|
| `M[c]` | `<layer>.M.int32.bin` | `int32 [E_out]` |
| `k` | manifest `requant.shift` | 상수 (레이어당 1개) |
| 포화 한계 | manifest `output.qmax` | 상수 |

`M`과 `k`는 `.pt`의 값에서 오프라인으로 이렇게 만들어집니다:

```
s_x    = x_scale.codes[0] * 2^-x_scale.frac_bits          # 입력 activation scale
s_w[c] = scale.codes[c]   * 2^-scale.frac_bits            # 채널별 weight scale
lsb_out= 소비자 포맷의 LSB (Qm.n이면 2^-n, INT8이면 그 step)
ratio[c] = s_x * s_w[c] / lsb_out
k        = max(k) s.t. |round(ratio*2^k)| < 2^31          # int32에 딱 맞게
M[c]     = round(ratio[c] * 2^k)
```

즉 `.pt`가 들고 있는 `(int16 code, frac_bits)` 쌍 두 개(입력 scale, weight scale)를
소비자의 LSB로 나눠 **하나의 int32 곱수 + shift**로 접는 것이고, 런타임에는 그
둘만 있으면 됩니다.

실제 예 (DVS128_10, `cross_attention.linear1`):

```
input.step  = 0.04223251        (앞 GELU가 내보내는 int8 step)
requant     : M = <128개 int32>, shift k = 32
output      : consumer = gelu1.in, Q3.12, lsb = 2^-12 = 0.000244, qmax = 32767
```

### 7.4 각 GEMM의 출력이 가야 할 포맷 (`output.*`)

소비자는 둘 중 하나입니다.

| GEMM | 소비자 | 출력 포맷 |
|---|---|---|
| `event_projection.seq_init.0` | GELU | Q4.11 (16b) |
| `preproc_block_events.seq_init.0` | GELU | Q4.11 (16b) |
| `proc_event_blocks.0.seq_init.1 / .4` | **ReLU** | **INT8** |
| `<blk>.attention.in_proj` | Q·Kᵀ / attn·V 연산기 | **INT8** (행 구간별 3개 step) |
| `<blk>.attention.out_proj` | residual stream → LayerNorm | Q16.7 (24b) |
| `<blk>.linear1`, `linear2` | GELU | **Q4.11 (16b)** |
| `<blk>.linear3` | residual stream → LayerNorm | Q16.7 (24b) |
| `proc_embs_block.linear1` | **ReLU** | **INT8** |
| `models_clf.0.linear_1` | **ReLU** | **INT8** |
| `models_clf.0.linear_2` | log_softmax (argmax만) | Q5.10 (16b) |

핵심: **residual stream 전체가 하나의 Q포맷(24비트 Q16.7)** 을 씁니다. 거기에
합류하는 GEMM(`out_proj`, `linear3`)은 그 포맷으로 직접 rescale하므로, residual
덧셈 `z_att + z_input`은 **같은 포맷끼리의 평범한 고정소수점 덧셈**이고 별도
정렬이 필요 없습니다.

### 7.5 in_proj의 행 구간별 requantize

`Wq|Wk|Wv`가 하나의 `int8 [384,128]`로 묶여 있고, 세 구간이 각각 다른 입력 scale과
다른 출력 INT8 step을 씁니다. `M[c]`가 이미 행마다 다르므로 **연산기는 하나면
되고**, manifest의 `requant.bands`가 행 범위와 각 구간의 step을 알려줍니다:

```
rows [0,128)   : Q,  input_step 0.05511  → 출력 int8 (q_proj step)
rows [128,256) : K,  input_step ...      → 출력 int8 (k_proj step)
rows [256,384) : V,  input_step ...      → 출력 int8 (v_proj step)
shift k = 39 (세 구간 공통)
```

`bias_k` / `bias_v`는 K/V의 **projection 후** 도메인에 있는 INT32 값입니다. K/V
시퀀스 끝에 토큰 하나로 붙습니다(`Lk`가 1 늘어남). 하드웨어에서는 이 둘을 미리
INT8 코드로 바꿔 저장하면 덧셈 없이 그냥 붙이면 됩니다.

### 7.6 Attention 두 MAC

manifest `attention[i]`에 블록별로 들어 있습니다 (실제 값: DVS128_10 cross_attention):

```
Q·Kᵀ  : A = Q_int (int8),  B = K_int (int8),  reduce = head_dim = 32
        acc는 int32
        out_code = (acc * 1192500936) >> 31        ← 1/sqrt(32)가 이미 곱수에 포함됨
        출력 = softmax.in, Q5.10 (lsb 2^-10)

softmax: 입력 Q5.10 → 출력 Q1.14 → int8 코드로 round(attn / 0.00787402)
         출력은 항상 0..127 (unsigned) → 곱셈기를 unsigned x signed로 축소 가능

attn·V: A = attn_int (uint8 0..127), B = V_int (int8), reduce = key 개수
        out_code = (acc * 1189884913) >> 37
        출력 = out_proj의 int8 입력 (step 0.05562)
```

softmax 출력 범위가 해석적으로 [0,1]이라 그 int8 step은 캘리브레이션 없이
`1/127`로 고정입니다.

### 7.7 비선형 유닛에 넣는 것

`manifest.nonlinear_formats`가 사이트마다 `(width, frac_bits, q_format, lsb)`를
줍니다. LUT 설계에 필요한 건 **입력 Q포맷 하나와 출력 Q포맷 하나**입니다.

```
GELU        : 입력 Q4.11 (16b) → LUT → 출력 Q4.11 (16b)   ← 네트워크 전체 공통, LUT 하나
softmax     : 입력 Q5.10 (16b) → 출력 Q1.14 → int8
LayerNorm   : 입력 Q16.7 (24b)
              mu = mean(x)                          (24b 입력, ~31b 누산기)
              centered = sat_Q16.7(x - mu)          (24b)
              var = mean(centered^2)                (24x24→48 제곱기)
              xhat = sat_Q3.12(centered * rsqrt(var+eps))   (16b)
              y = sat_Q3.12(xhat * gamma + beta)     gamma/beta는 int16 Q1.14
              (LayerNorm 출력 포맷은 사이트별로 Q2.13~Q4.11)
ReLU        : 입력 int8 → 출력 int8 (부호 비트만 보면 됨)
log_softmax : 입력 Q5.10, 출력은 argmax만 쓰므로 계산 불필요
```

### 7.8 위치 인코딩 LUT

```
pos_encoding.int8.bin : int8 [21][21][64]   (ASL은 [40][30][64])
step                  : 0.01007318
인덱싱                : pos[y / 6][x / 6]   ← downsample_pos_enc = 6
```

`event_projection`의 GELU 출력(Q4.11)과 이 LUT에서 읽은 int8을 concat해
`preproc_block_events` GEMM에 넣습니다. **배포 설정은 K-split을 쓰지 않으므로**
두 절반이 하나의 입력 scale(0.0822)로 함께 INT8 양자화됩니다 — 즉 pos-enc 코드는
자기 step(0.01007)에서 공유 step(0.0822)으로 한 번 재양자화됩니다. GEMM은
`K=160` 하나짜리 평범한 형태이고 `M` 벡터도 하나입니다.

정밀도 여유가 더 필요해지면 §4의 K-split(누산기 1개 + DSP 1개 추가)을 켜면 되고,
그때는 manifest에 `k_split`, `M_A`, `M_B`가 대신 생깁니다.

### 7.9 정수 공식이 시뮬레이터와 같은지 검증

`export_fpga_constants.py`가 실제 배치에서 위 정수 공식(`(acc*M)>>k`)의 출력을
시뮬레이터 출력과 비교합니다 (`verification.csv`):

| 데이터셋 | 정확히 일치한 레이어 | 1 LSB 이내 | 최대 불일치 |
|---|---|---|---|
| DVS128_10 | 7 / 16 | **100.0000%** | 1 LSB |
| DVS128_11 | 5 / 16 | **100.0000%** | 1 LSB |
| ASL_DVS | 3 / 16 | **100.0000%** | 1 LSB |

INT8을 소비자로 갖는 레이어는 **전부 완전 일치**합니다. Qm.n을 소비자로 갖는
레이어에서 ±1 LSB가 나오는데, 이는 "float 곱한 뒤 반올림"과 "정수 곱한 뒤
round-to-nearest shift"의 차이로 예상된 값입니다.

**남는 위험**: 보고된 정확도는 시뮬레이터 경로에서 나온 것이고, 실제 RTL은
레이어당 최대 1 LSB 다를 수 있습니다. Q3.12의 1 LSB는 2.4e-4인데 그 값을 받는
다음 단의 INT8 step은 0.02–0.1이므로, 섭동이 다음 양자화 step의 1/100 이하라 거의
항상 흡수됩니다. 그래도 **보드에서 나온 logit을 이 경로와 한 번 대조**하는 것을
권합니다.

### 7.10 내보내진 파일

```
fpga_export/<dataset>__deploy_gelu_q411/
  manifest.json                  ← 위의 모든 상수·포맷·파일 매핑
  verification.csv               ← 레이어별 정수공식 대조 결과
  <layer>.W.int8.bin             ← weight            int8  [E_out][E_in]
  <layer>.B.int32.bin            ← folded bias       int32 [E_out]
  <layer>.M.int32.bin            ← requant 곱수       int32 [E_out]
  <mha>.bias_k/.bias_v.int32.bin
  pos_encoding.int8.bin
  <layernorm>.weight/.bias.int16.bin    (gamma/beta)
  backbone.memory_vertical.int16.bin
```

raw blob 합계는 DVS128 522.6 KiB, ASL 571.9 KiB입니다(§6의 payload 수치와 일치하며,
`M[c]` 벡터가 추가된 만큼만 큽니다 — `M`은 `.pt`에 들어 있는 두 scale에서 파생된
값이라 둘 중 하나만 보드에 올리면 됩니다).

## 8. 파일

| 파일 | 내용 |
|---|---|
| `quant_lib/fixed_point.py` | Qm.n 프리미티브, 포맷 선택 |
| `quant_lib/hw_quant.py` | 정수 데이터패스 + pack/unpack |
| `run_hw_quantization.py` | config별 calibrate → pack → reload → evaluate |
| `audit_inference_datapath.py` | 실제 forward를 계측해 피연산자/누산기/격자 검증 |
| `export_fpga_constants.py` | `.pt` → RTL용 정수 상수(M, shift) + raw binary, 그리고 정수공식 대조 검증 |
| `verify_hw_model_sizes.py` | 독립적인 바이트 수 검증 + dtype 감사 |
| `analyze_pos_encoding.py` | 모델이 positional encoding을 얼마나 쓰는지, 공유 INT8 scale에서 얼마나 살아남는지 |
| `test_hw_quant.py` | 프리미티브 단위 검증 |
| `plot_hw_quantization.py` | 그림 생성 |
| `<dataset>/hw_quantized_models/hw__<config>.pt` | 패킹된 체크포인트 |
