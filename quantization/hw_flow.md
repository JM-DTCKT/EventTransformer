# EvT 정수 데이터패스 (ZCU102) — 유닛 레퍼런스

`quant_lib/hw_quant.py`가 실제로 무엇을 계산하는지를 **유닛별로** 적은 문서입니다.
실행 방법과 정확도는 `README.md`, 상수 실측값은
`fpga_export/<dataset>/manifest.json`에 있습니다.

아래 숫자는 전부 **DVS128_10 실측**입니다(다른 데이터셋은 상수만 다르고 포맷은 동일).

## 표기

```
Qm.n     16비트 2의 보수 정수를 code · 2^-n 으로 해석 (부호1 + 정수m + 소수n, m+n=15)
         n이 음수면 LSB 가중치가 1보다 큼 (Q16.-1 → lsb 2.0)
s_x      activation scale (텐서당 스칼라)      s_w[c]   채널 c의 weight scale
acc      INT32 누산기                          M, sh    requant 곱수(int32) / 시프트(uint6)
```

**requantizer** — 데이터패스에 단 한 종류뿐입니다:

```
out_code[c] = sat( (acc[c] · M[c] + (1 << (sh-1))) >> sh )
```

`M[c] = round(s_x · s_w[c] / lsb_out · 2^sh)`, `sh`는 `|M| < 2^31`을 만족하는 최대값.
`s_x`·`s_w`·`lsb_out`은 전부 컴파일 타임 상수라 **역양자화와 재양자화가 곱수 하나로
접힙니다.** 런타임에 나눗셈도, 부동소수점도 없습니다. 실측 `sh` = 31~46 → **uint6**.

예외가 하나 있습니다: 소비자가 bf16 LayerNorm인 6개 레이어는 정수 시프트 대신
`out = bf16(acc · s_x·s_w[c])`로 나갑니다 (§2.6).

---

## 1. 자료형 한눈에 보기

| 자료형 | 어디에 |
|---|---|
| `int8` / `uint8` | 모든 MAC 피연산자, ReLU 전후, softmax 출력, pos-enc LUT |
| `INT32` | 누산기, folded bias, requant 곱수 `M` |
| `Q4.11` (16b) | **GELU 입출력 — 전 네트워크 고정, LUT 하나** |
| `Q6.9` (16b) | **softmax 입력 — 전 네트워크 고정, exp LUT 하나** |
| `Q1.14` (16b) | softmax 출력 |
| `Q4.11` (16b) | **LayerNorm `xhat`·출력 — 전 사이트 고정** |
| `Q1.14` / `Q4.11` (16b) | LayerNorm `gamma` / `beta` |
| **`bfloat16`** | **LayerNorm 앞단(입력~rsqrt) + residual stream** |

---

## 2. 유닛별 명세

### 2.1 Linear / GEMM

모든 `nn.Linear`. reduction 길이 K = 128 / 144 / 160.

| | |
|---|---|
| **입력** | `act: int8` (앞 단 requantizer가 이미 그 step으로 내보냄 — 입력 나눗셈 없음) |
| **weight** | `int8 [E_out][E_in]` + 채널별 scale, `bias`는 INT32로 접힘 |
| **계산** | `acc[c] = Σ_k x_int[k]·w_int[c,k] + b_int[c]` → `INT32` |
| **출력** | requantizer가 소비자 포맷으로 직접 rescale (아래 표) |

```
x_int  int8 ─┐
             ├─ MAC ─→ acc:INT32 ─→ (·M[c] +2^(sh-1)) >>sh ─→ sat ─→ 소비자 포맷
w_int  int8 ─┘         +b_int:INT32
```

**레이어별 입출력** (실측):

| 레이어 | 입력 step | `sh` | 출력 |
|---|---|---|---|
| `event_projection.seq_init.0` | 0.030147 | 33 | Q4.11 → GELU |
| `preproc_block_events.seq_init.0` | 0.084709 | 31 | Q4.11 → GELU |
| `proc_event_blocks.0.seq_init.1` | 0.038420 | 39 | int8 (step 0.036535) → ReLU |
| `proc_event_blocks.0.seq_init.4` | 0.020563 | 39 | int8 (step 0.019991) → ReLU |
| `<blk>.attention.in_proj` | 행 3구간 | 39/40 | **int8 ×3** (§2.2) |
| `<blk>.attention.out_proj` | 0.057617 | — | **bf16** → LayerNorm |
| `<blk>.linear1` | 0.042250 | 33 | Q4.11 → GELU |
| `<blk>.linear2` | 0.071793 | 32 | Q4.11 → GELU |
| `<blk>.linear3` | 0.037500 | — | **bf16** → residual |
| `proc_embs_block.linear1` | 0.040701 | 40 | int8 (0.066265) → ReLU |
| `models_clf.0.linear_1` | 0.062143 | 40 | int8 (0.119465) → ReLU |
| `models_clf.0.linear_2` | 0.115334 | — | **argmax** (§2.8) |

**INT32 bias.** `b_int[c] = round(b[c] / (s_x·s_w[c]))`를 오프라인에서 계산합니다.
누산기 LSB가 곧 `s_x·s_w[c]`이므로 bias가 **누산기 위 정수 덧셈 하나**가 되고,
rescale 곱수는 GEMM이 어차피 필요로 하던 값 그대로입니다. 비용은 누산기 LSB의
절반 이하 오차. 실측 최대 `|b_int|` = 6,204.

**비트폭.** `x·w`는 int16(최악 15b), `acc`는 실측 최대 232,898(19b) / 이론 상한 23b
→ INT32에 여유. `acc·M`은 23b × 31b + 부호 = **54비트** (DSP48E2의 48비트 누산기를
넘으므로 캐스케이드 필요).

### 2.2 Attention

`heads=4`, `head_dim=32`, `E=128`. `AttentionBlock` 3개(cross 1 + latent 2),
회로는 하나이고 상수만 다릅니다.

| | cross_attention | latent_attentions.0 / .1 |
|---|---|---|
| Q 출처 / K·V 출처 | `layer_norm_1(latent 96)` / `layer_norm_x(이벤트)` | 둘 다 latent 96 |
| `Lq` / `Lk` | 96 / 이벤트수+1 (실측 94, 112) | 96 / **97** |

#### ① `in_proj` — GEMM 하나, 행 3구간

`Wq|Wk|Wv`가 `int8[384][128]` 하나로 묶여 있고 `M[c]`가 행마다 다르므로 연산기는
하나입니다. `sh = 39`(cross) / `40`(latent)를 세 구간이 **공유**합니다.

| 구간 | 입력 텐서 | 입력 step (cross 실측) | 출력 |
|---|---|---|---|
| Q rows [0,128) | `layer_norm_1` 출력 | 0.04330635 | `int8`, q_proj step |
| K rows [128,256) | `layer_norm_x` 출력 | **0.06583405** | `int8`, k_proj step |
| V rows [256,384) | `layer_norm_x` 출력 | **0.06583405** | `int8`, v_proj step |

**K와 V는 같은 텐서를 읽고 step도 같습니다** (`attention_forward_hw(z, x_ln, x_ln, …)`).
따라서 HW는 `x_ln`을 **한 번만 int8로 인코딩해 두 구간이 공유**하면 됩니다 — 소스
텐서는 2개, 입력 인코딩도 2개입니다.

> **`bias_k` / `bias_v`는 bias가 아니라 토큰입니다.** `add_bias_kv=True`가 만드는
> `[1,1,128]` 파라미터로, `torch.cat([k, bias_k], dim=0)` — 시퀀스 길이를 1 늘립니다
> (`Lk += 1`). 더하면 효과가 없습니다: `Q·(K[j]+bias_k) = Q·K[j] + 상수`이고
> `softmax(x+c) = softmax(x)`로 상쇄되기 때문입니다. 토큰으로 붙여야 attention이
> 그 자리에 확률질량을 **선택적으로** 줄 수 있습니다(attention sink).
> `key_padding_mask`에 `False` 한 칸이 함께 붙으므로 **절대 마스킹되지 않습니다.**
> HW 비용은 오프라인에서 int8로 바꿔둔 **ROM 128 B × 2**가 전부입니다.

#### ② `Q·Kᵀ` — weight 없는 MAC

| | |
|---|---|
| **입력** | `act:int8 × act:int8` (둘 다 activation), reduce = `head_dim` = 32 |
| **계산** | `acc = Σ_d Q_int[d]·K_int[d]` → INT32 → requant |
| **출력** | **Q6.9** (softmax 입력) |

`M`에 `s_q·s_k / lsb_out` **과 `1/√32`가 함께** 접혀 있어 스케일링 곱셈이 따로
없습니다. 여기서 `M`은 채널 벡터가 아니라 **블록당 스칼라 1개**입니다.

| | `M` | `sh` |
|---|---|---|
| cross_attention | 1,165,286,311 | 32 |
| latent_attentions.0 | 1,457,637,920 | 32 |
| latent_attentions.1 | 1,328,161,171 | 32 |

#### ③ `attn·V`

| | |
|---|---|
| **입력** | `act:uint8 [0,127] × w:int8`, reduce = `Lk` (실측 94/97/112) |
| **계산** | `acc = Σ_j attn_int[j]·V_int[j]` → INT32 → requant |
| **출력** | `int8` (step 0.057617 = `out_proj`의 입력 step) |

피연산자에 음수가 없으므로 **unsigned × signed 곱셈기**로 축소됩니다.
누산기 상한 `127·127·112 = 1,806,448`(21b).
cross: `M = 1,170,898,206`, `sh = 37`.

#### ④ `out_proj`

`§2.1`의 평범한 GEMM. 입력 `int8`, 출력 **bf16** (residual stream).

### 2.3 GELU

| | |
|---|---|
| **입력** | **Q4.11** (16b, lsb 4.883e-4, 범위 ±16) |
| **계산** | LUT — **네트워크 전체가 하나** |
| **출력** | **Q4.11** → 다음 GEMM 입력 `int8`로 재양자화 (스칼라 `M` 하나) |

사이트마다 포맷을 따로 고르면 LUT을 여러 개 만들거나 포맷 전환이 필요합니다.
Q4.11로 못박으면 LUT 하나로 끝나고, 실측 정확도는 오히려 같거나 좋았습니다.

ASL_DVS의 첫 GELU만 입력이 ±16을 넘어 원소의 0.64%가 클리핑되는데, GELU는 x>4에서
사실상 항등함수이고 그 출력이 어차피 다음 INT8 양자화기(step 0.138 → 17.6에서 클리핑)를
통과하므로 손실이 흡수됩니다.

### 2.4 Softmax

| | |
|---|---|
| **입력** | **Q6.9** (16b, lsb 1.953e-3, 범위 ±64) — **전 블록 통일, exp LUT 하나** |
| **계산** | `exp LUT → Σ(Lk항) → × 1/Σ` |
| **출력** | **Q1.14** (16b) → `uint8 [0,127]`, step `1032 · 2^-17 = 0.00787401` |

- 출력 범위가 해석적으로 `[0,1]`이라 int8 step은 **캘리브레이션 없이 `1/127` 고정**
  이고 세 블록이 동일합니다.
- **마스킹에 `-inf`가 필요 없습니다.** `-inf`의 목적은 `exp(-inf)=0`뿐이므로
  HW에서는 exp LUT 출력을 마스크로 게이팅하면 됩니다:
  `exp_out[j] = mask[j] ? 0 : LUT(score[j])`.
- **여기가 유일하게 나눗셈이 남는 곳입니다.** `Σexp`는 런타임 값이라 상수로 접을 수
  없어 reciprocal 유닛(LUT + Newton-Raphson)이 필요하고, 정규화를 하려면 행 전체를
  먼저 봐야 합니다 → score 행 버퍼 `16b × 112 = 224 B` 또는 online-softmax.

### 2.5 ReLU

| | |
|---|---|
| **입력** | `int8` | 
| **계산** | 부호 비트만 보면 됨 |
| **출력** | `int8`, 같은 step → **requant 없음** |

### 2.6 LayerNorm — bf16 앞단 + Qm.n 뒷단

| | |
|---|---|
| **입력** | **bfloat16** — 생산 GEMM이 `bf16(acc · s_x·s_w[c])`로 내보냄 |
| **계산** | 정규화까지 bf16, `xhat` 이후 Qm.n |
| **출력** | **Q4.11** (16b) → 다음 GEMM 입력 `int8`로 재양자화 |

```
x        = bf16(입력)                          ─┐
mu       = bf16( mean(x) )                      │ 동적 범위 5자릿수
centered = bf16( x - mu )                       │ → bf16의 8비트 지수가 필요
var      = bf16( mean(centered²) )              │
rstd     = bf16( rsqrt(var + eps) )            ─┘
─────────────────────────────────────────────────────────────────
xhat     = Q4.11( centered · rstd )            ─┐ 정규화 후 |xhat| < 9.02
y        = Q4.11( xhat · gamma + beta )         │ → 지수 불필요, 가수가 전부
                     Q1.14    Q4.11            ─┘
```

**왜 앞단만 bf16인가.** 관측된 동적 범위가 두 구간에서 완전히 다릅니다:

| 사이트 | `max\|x\|` | 포맷 |
|---|---|---|
| `layer_norm.in` (proc_embs) | **242,606** | bf16 — 지수 범위가 꼭 필요 |
| `layer_norm_att.in` (residual) | **48,523** | bf16 |
| `layer_norm_*.hat` (정규화 후) | **4.80 – 9.02** | **Q4.11** |
| `layer_norm_*.out` | **5.10 – 9.45** | **Q4.11** |

`xhat`은 정규화 결과라 범위가 ±9로 묶입니다. 거기서 bf16의 8비트 지수는 아무 일도
하지 않고 남은 8비트 가수가 정밀도를 결정합니다 — `|xhat|≈1`에서 ulp `3.9e-3`.
같은 16비트를 Q4.11로 쓰면 lsb `4.88e-4`로 **8배 정밀**하고 범위 ±16도 넉넉합니다
(관측 최대 9.45 → 1.7배 여유). Q3.12(±8)는 `layer_norm_2`(9.45)에서 클리핑되므로
**Q4.11이 전 사이트를 통일할 수 있는 유일한 16비트 포맷**입니다.

**affine 단계는 곱셈기 하나 + 고정 시프트입니다.**

```
xhat  = code_x · 2^-11   int16      (|code_x| ≤ 18,473,  15b)
gamma = code_g · 2^-14   int16      (|code_g| ≤  2,572,  12b)
beta  = code_b · 2^-11   int16

① code_x · code_g                    15b × 12b → 26b       ← DSP48E2 1개
   값 = (code_x·code_g) · 2^-25
② beta 정렬:  code_b << 14           같은 2^-25 스케일로
③ acc = ① + ②                       int32 (26b 사용)
④ y_code = sat16( (acc + 2^13) >> 14 )   →  Q4.11
```

`shift = n_xhat + n_gamma − n_out = 11 + 14 − 11 = 14`, **전 사이트 공통 상수**라
`M` 곱셈기가 필요 없습니다.

**gamma는 Q1.14, beta는 Q4.11 — 이유가 다릅니다.**

| | 포맷 | 근거 |
|---|---|---|
| `gamma` | **Q1.14** (`max\|γ\|` = 1.256) | `xhat`에 **곱해지므로** 오차가 증폭됨. Q4.11로 내리면 출력 오차가 `9.02 × 2.44e-4 ≈ 4.5 LSB`, Q1.14면 `0.55 LSB` |
| `beta` | **Q4.11** | 출력에 **더해지기만** 함 → 출력 LSB(4.88e-4)보다 정밀해도 해상 불가. 이전 `Q-4.19`는 256배 과잉 |

gamma를 Q1.14로 두어도 하드웨어는 안 복잡해집니다 — 위 shift가 11이 아니라 14가
될 뿐입니다.

**부수 효과: requantizer가 한 종류로 통일됩니다.** LayerNorm 출력이 Qm.n이므로
다음 GEMM 입력(`int8`)으로 가는 경로가 다른 모든 곳과 같은 정수 `M`+shift입니다.
bf16 출력이었다면 그 지점에만 별도 bf16 곱셈기가 필요했습니다.

### 2.7 skip connection

**원칙: 덧셈기에서 정렬하지 않고, 생산자가 같은 포맷으로 내보내게 만듭니다.**

| 덧셈 | 두 피연산자 | 정렬 |
|---|---|---|
| `z_att = out_proj ⊞ z_input` | bf16 + bf16 | 불필요 |
| `출력 = linear3 ⊞ z_att` | bf16 + bf16 | 불필요 |
| `latent_vectors ⊞ inp_q` | bf16 + bf16 | 불필요 |
| `x ⊞ x_input` (MLPBlock) | int8@s1 + int8@s0 | **곱셈기 필요** |

앞 세 개는 `out_proj`/`linear3`의 requantizer가 bf16을 타깃으로 하므로 그냥 bf16
덧셈입니다. 마지막 하나만 두 항의 step이 임의값(비율이 2의 거듭제곱이 아님)이라
정렬에 곱셈이 듭니다 — `seq_init.4`의 출력 step을 `s0`와 같게 강제하면
9비트 정수 덧셈으로 줄일 수 있습니다(미적용, 확인 필요).

### 2.8 분류 헤드 — log_softmax 없이 argmax

| | |
|---|---|
| **입력** | `act:int8` (step 0.115334), `w:int8 [C][128]` |
| **계산** | `acc[c] = Σ x_int·w_int + b_int` → **`argmax_c( acc[c] · M[c] )`** |
| **출력** | 클래스 인덱스 |

추론은 최댓값의 인덱스만 쓰고 `log_softmax`는 **단조증가**이므로
`argmax(log_softmax(z)) = argmax(z)`입니다. 따라서 하드웨어에는
**log_softmax도, 출력 Q포맷도, 시프트도, 포화도 없습니다.**

```
acc[c] : INT32  ──→  × M[c]  ──→  argmax over c  ──→  클래스 인덱스
                     (54b 곱, 시프트 불필요)
```

> **`M[c]`는 생략할 수 없습니다.** `logit[c] = acc[c] · s_x · s_w[c]`에서 `s_x`는
> 전 클래스 공통이라 순서에 영향이 없지만, **`s_w[c]`는 클래스마다 다릅니다.**
> `acc[c]`만으로 argmax를 하면 클래스 순서가 뒤바뀝니다.
>
> 반면 `sh`는 전 클래스 공통이라 `>> sh`가 순서를 보존합니다 — **생략해도 됩니다.**
> 포화도 생략하는 편이 낫습니다(두 클래스가 같이 포화하면 순서 정보를 잃습니다).

**비용**: 곱셈기 1개 + 비교 트리(C = 11~24개). LUT도, 나눗셈도, 지수 함수도 없습니다.
검증은 코드값이 아니라 **결정 자체**를 대조합니다 — 세 데이터셋 모두 argmax
**100.0000% 일치**.

---

## 3. 파라미터 저장 맵

### 3.1 `.pt` — `<dataset>/quantized/model_int8.pt`

```
payload
├─ config              {name:'fpga_int8', gelu_frac_bits:11, softmax_frac_bits:9}
├─ layers[<sd_key>]    GEMM 하나당 1개, 22개
│    kind              'linear' | 'in_proj'
│    q                 int8 [E_out][E_in]              ← weight
│    scale             {codes:int16[E_out], frac_bits} ← 채널별 s_w, 레이어당 shift 공유
│    x_scale           {codes:int16[1],     frac_bits} ← s_x   (linear만)
│    bias              {q: int32[E_out]}               ← folded bias
│    bias_k, bias_v    {q: int32[E]}                   ← in_proj만, K/V 토큰
├─ fx_params[<sd_key>] {codes:int16[·], frac_bits, shape}   LN gamma/beta 26개 + memory_vertical
├─ pos_encoding        {q:int8[21][21][64], scale, shape}
├─ attention_scales    q_in/k_in/v_in/qk/v/out_proj/softmax, 전부 (int16 code, frac)
├─ sites[<name>]       비선형 경계 79개: {kind, frac_bits|scale, total_bits, role, max_abs}
└─ fp32_other          추론에 안 쓰이는 나머지 (criterion.weight 1개)
```

**scale 읽는 법**: `값 = codes · 2^-frac_bits`.
예) `x_scale = {codes:[22142], frac_bits:19}` → `22142/2^19 = 0.042232513`.

### 3.2 `.bin` — `fpga_export/<dataset>/`

`export_fpga.py`가 `.pt`의 `(code, frac_bits)` 쌍들을 RTL이 바로 쓰는 형태로 접습니다.

| 파일 | 개수 | dtype | 내용 |
|---|---|---|---|
| `<layer>.W.int8.bin` | 22 | `int8 [E_out][E_in]` | weight, row-major |
| `<layer>.B.int32.bin` | 22 | `int32 [E_out]` | folded bias |
| `<layer>.M.int32.bin` | **16** | `int32 [E_out]` | requant 곱수 (`sh`는 manifest) |
| `<layer>.S.bf16.bin` | **6** | `bf16 [E_out]` | bf16 소비자용 채널 scale (`M` 대신) |
| `<mha>.bias_k/.bias_v.int32.bin` | 3+3 | `int32 [E]` | K/V 토큰 |
| `<ln>.weight/.bias.int16.bin` | 13+13 | `int16 [E]` | LayerNorm gamma/beta |
| `pos_encoding.int8.bin` | 1 | `int8 [21][21][64]` | step 0.010073, 인덱스 `[y//6][x//6]` |
| `memory_vertical.int16.bin` | 1 | `int16 [96][128]` | latent 초기값 |
| `manifest.json` | 1 | — | 위 파일 매핑 + 모든 상수·포맷 |
| `verification.csv` | 1 | — | 레이어별 정수공식 대조 결과 |

raw blob 합계 522.6 KiB (DVS128) / 571.9 KiB (ASL).

> `M`은 `.pt`의 두 scale에서 파생된 값이라 **둘 중 하나만 보드에 올리면 됩니다.**

---

## 4. 필요한 회로

| 회로 | 하는 일 |
|---|---|
| **MAC 어레이** | `int8 × int8 → INT32` (한 곳은 `uint8 × int8`) |
| **requantizer** | 곱셈기 1 + 덧셈 1 + 시프트 + 클램프. GEMM 뒤는 채널 벡터 `M`, 비선형 뒤는 스칼라 `M` |
| **LUT** | GELU 1개(Q4.11→Q4.11), exp 1개(Q6.9 입력), rsqrt |
| **비교 트리** | 분류 헤드의 argmax (C = 11~24) — log_softmax 없음 |
| **bf16 유닛** | LayerNorm 데이터패스 + residual 덧셈 |

`1/√32`, `1/127`, `1/96`, 그리고 다음 단 격자로의 나눗셈까지 **전부 `M`에 흡수**되어
있습니다. 런타임 나눗셈은 softmax의 `1/Σ`와 LayerNorm의 `rsqrt` 둘뿐입니다.

---

## 5. 검증

`quantize.py`의 정확도는 **저장된 `.pt`에서 모델을 다시 불러와** 이 데이터패스로
forward를 돌린 결과입니다. 그 위에 두 가지 독립 검증이 있습니다.

**`audit.py`** — 실제 배치를 흘리며 텐서 단위로 계측 (DVS128_10):

| 확인 항목 | 결과 |
|---|---|
| 모든 MAC 피연산자가 [-128,127]의 정확한 정수 | ✅ |
| 모든 누산기가 정확한 정수 | ✅ |
| 최대 \|누산기\| | 232,898 (INT32 한계 2.1e9) |
| 최대 \|folded INT32 bias\| | 6,204 |
| 비선형 입력이 선언된 격자 위 | ✅ (전 사이트) |
| 캘리브레이션 범위 포화 | 0건 |

**`export_fpga.py`** — 내보낸 정수 상수로 각 레이어를 다시 계산해 시뮬레이터와 대조.
**22개 GEMM 전부**를 덮습니다: attention 내부 6개(`in_proj` ×3, `out_proj` ×3)는
모듈 forward 훅으로 볼 수 없어 `hw_quant.ATTN_TAP`으로 I/O를 잡습니다.

| | DVS128_10 | DVS128_11 | ASL_DVS |
|---|---|---|---|
| 완전 일치 레이어 | 12 / 22 | 9 / 22 | 9 / 22 |
| **1 LSB 이내** | **100.0000%** | **100.0000%** | **100.0000%** |
| 최악 불일치 | 1 LSB | 1 LSB | 1 LSB |

bf16 소비자 6개(`out_proj` ×3, `linear3` ×3)는 **0.00 ulp — 완전 일치**입니다.
`in_proj`는 Q/K/V 세 구간을 합쳐 한 행으로 보고합니다.

### 남은 확인 항목

1. **`[5]`의 `x ⊞ x_input` 정렬**을 step 통일로 없앨 수 있는지 (§2.7). 지금은 두 항의
   int8 step이 임의값이라 정렬에 곱셈이 듭니다.
2. **캘리브레이션 데이터에 augmentation이 걸려 있습니다.** train loader가
   `validation=False`라 랜덤 crop/flip/drop이 적용된 분포에서 범위를 잽니다.
   배포 분포와 맞추려면 augmentation을 끈 train 데이터를 쓰는 것이 맞지만,
   현재는 원 파이프라인 동작을 그대로 유지하고 있습니다.
