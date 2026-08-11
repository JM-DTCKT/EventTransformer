# real_dvs_script — DVS128_10 실데이터 → 보드 입력 + 골든

`nonlinear_script` 은 MNIST 로 **비선형 유닛과 16비트 포맷**을 덮었습니다. 여기는
**실제 EvT 를 DVS128 제스처 데이터로** 보드에 올리기 위한 호스트 쪽 전부입니다 —
전처리 · 골든 · RTL 대조용 tap.

양자화는 새로 하지 않습니다. `quantization/` 이 이미 끝내 둔 `model_int8.pt` 와
`fpga_export/DVS128_10/` 을 그대로 씁니다.

이어지는 RTL: [`04_basic_rtl/fpga_dvs128_10/`](../../../04_basic_rtl/fpga_dvs128_10/)

---

## 1. 결과

| | |
|---|---|
| fp32 체크포인트 | 98.1061 % |
| 정수 데이터패스 (`quantization/`) | **97.3485 %** |
| **여기 전처리 + fp32 입력** | **97.3485 %** (257/264) |
| **여기 전처리 + int8 입력** | **97.3485 %** (257/264) |

두 줄이 소수점까지 같다는 것이 이 폴더의 검증 결과입니다. 뜻하는 바:

1. 전처리가 학습/평가 때와 **한 글자도 다르지 않다**
2. 유효 토큰만 보내고 `n_tok` 로 마스크를 표현한 것이 원래 패딩 표현과 **등가**다
3. 보드에는 **int8 코드만** 보내면 되고 fp32 를 실어 나를 필요가 없다

---

## 2. 입력이 이미지가 아닙니다

MNIST 는 이미지를 그대로 넣으면 됐습니다. EvT 입력은 이벤트 스트림을 패치 토큰으로
바꾼 것이라 전처리 자체가 파이프라인입니다:

```
aedat → 12 ms sparse frame → chunk(24 ms = 2 frame) → 시간 bins 2
      → 6×6 패치 → 활성 패치만(≥ 3픽셀) → log(1+p) → 토큰 144차원
```

`144 = 6×6 패치 × 2 bins × 2 극성`.

**이걸 새로 구현하지 않았습니다.** `data_generation.py::Event_DataModule` 을
**validation 모드**로 그대로 돌립니다. 새로 짰다면 그 구현 자체가 새 검증 대상이
됐을 겁니다.

### 결정론

`data_generation.py` 는 `__getitem__` 안에서 `np.random` 으로 증강합니다(랜덤 시간창,
크롭, 플립, 토큰 드롭). `validation=True` 면 그 경로가 전부 꺼지므로 재실행해도 같은
텐서가 나옵니다. `quantization/README.md` 가 지적한 "어느 배치를 봤는지가 결과의
일부" 문제가 여기서는 생기지 않습니다.

### 실측 (테스트셋 264 샘플)

| | min | 중앙 | 평균 | max |
|---|---|---|---|---|
| T (timestep) | 3 | 20 | 19.6 | **20** |
| 활성 토큰 / 스텝 | 16 | 43 | 45.6 | **123** |

0토큰 스텝은 없습니다. 이 두 숫자가 RTL 메모리 사이징을 정합니다 — 토큰 128(32×4
타일), latent 96(32×3 타일).

---

## 3. 마스크를 `n_tok` 하나로

원 파이프라인은 `(T, B, Npad, 144)` 로 **앞쪽 0 패딩**을 하고
`mask = kv.sum(-1) == 0` 으로 마스크를 만듭니다. 보드는 패딩을 아예 받지 않고
타임스텝마다 **유효 토큰 수 `n_tok` 하나**만 받습니다 — `j < n_tok` 이 곧 마스크입니다.

전체 토큰의 **63%가 패딩**이라 그만큼 DMA 를 안 씁니다.

### 등가성 근거

정확도 재현(위)이 종단 증거이고, 원인 쪽도 직접 봤습니다. 샘플 0 의 step 1 은
유효 49 / 패딩 3 인데:

```
패딩 열의 softmax 출력 |max| = 0      유효 열 |max| = 125
```

**마스크된 키는 정확히 0 을 기여합니다.** 그러니 아예 빼도 결과가 같습니다.
(`bias_k`/`bias_v` 는 예외 — bias 가 아니라 **토큰**이고 절대 마스킹되지 않습니다.
`Lk = n_tok + 1`.)

---

## 4. 골든 tap

레이어 출력만 보면 유닛이 많을 때 어디가 틀렸는지 못 짚습니다. EvT 는 GEMM 22개 +
attention MAC 6개라 더 심합니다. `quant_lib/hw_quant.py` 가 이미 갖고 있는 관측점
세 가지를 합쳐 씁니다:

| | 무엇을 주나 | |
|---|---|---|
| 모듈 forward 훅 | Linear/LayerNorm/GELU 입출력 | 이름 기반 |
| `ATTN_TAP` | attention 의 q_in/k_in/v_in, q/k/v, out_proj | 이름 기반 |
| **`MAC_PROBE`** | **모든 정수 MAC** 의 (a, b, acc) — attention 내부 포함 | 순서 기반 |

`MAC_PROBE` 가 결정적입니다. `attention_forward_hw` 는 `int8_linear` 를 직접 부르므로
모듈 훅으로는 안 보이는데 여기로는 보입니다. 그리고 **`attn_AV` 의 첫 피연산자가 곧
softmax 출력(uint8)** 이라, RTL attention 테스트벤치에 그대로 들어갑니다.

샘플 하나 2 타임스텝 = **사이트 151개 / MAC 63개**:

```
linear_K128     49회      linear_K144      1회   ← event_projection (T 전체 한 번)
attn_QK^T_K32    6회      linear_K160      1회   ← preproc        (마찬가지)
attn_AV_K97      4회      latent attention, Lk = 96 + 1
attn_AV_K53      2회      cross  attention, Lk = n_tok + 1
```

`event_projection` 과 `preproc` 이 타임스텝당이 아니라 **한 번**인 것에 주의하세요 —
`EvT.forward` 가 `(T, ...)` 텐서 전체에 먼저 적용하고 그다음 타임스텝 루프를 돕니다.

---

## 5. 네트워크

```
이벤트 → 6×6 패치 토큰 144차원 (활성 패치만)
  ├─ event_projection   144→96   GELU
  ├─ ⊕ fourier pos enc  64        → 160
  ├─ preproc            160→128  GELU
  ├─ proc_events MLP    128→128 ReLU ×2, + x
  └─ proc_memory TransformerBlock
       cross_attention    Q = latent 96,  K/V = 토큰   ← 마스크
       latent_attentions  Q = K = V = latent 96        ← 마스크 없음
       (각 블록: in_proj → QKᵀ → softmax → ·V → out_proj → linear1/2/3)
  … 위를 T(≤20) 타임스텝 반복, latent 를 누적 …
→ LayerNorm → Linear → ReLU → latent 96개 평균 → CLF → argmax
```

`heads=4`, `head_dim=32`, `E=128`, `latent=96`, `클래스=10`.

**마스킹이 필요한 곳은 cross_attention 하나**입니다 — latent 끼리는 96개가 항상
전부 유효합니다.

---

## 6. 실행

```bash
EVT=/hai/home/sgh/.conda/envs/evt_new/bin/python

$EVT preprocess.py            # 테스트셋 → data/    (264 샘플, 34 MB)
$EVT golden.py                # 정확도 97.3485 % 재현 확인
$EVT golden.py --int8         # 보드가 받는 int8 코드로도 동일한지
$EVT taps.py --sample 0 --steps 2    # RTL 대조용 tap → taps/
```

---

## 7. 출력

### `data/` — 보드 입력

| | |
|---|---|
| `tokens.int8.bin` | 유효 토큰만 이어붙임, 토큰 하나 = 144 B (34.0 MB) |
| `tokens.fp32.bin` | 골든 대조용 (136 MB, 보드에는 안 감) |
| `pos_idx.int16.bin` | 토큰별 `(y//6)*21 + (x//6)` |
| `index.int32.bin` | 타임스텝마다 `[tok_off, n_tok, 0, 0]` — **여기 `n_tok` 이 마스크** |
| `samples.int32.bin` | 샘플마다 `[step_off, T, 0, 0]` |
| `labels.int32.bin` | 정답 |
| `pred_golden.int32.bin` | 골든 예측 (보드 대조용) |
| `meta.json` | 위 전부의 형상·dtype·`input_step` |

int8 인코딩은 첫 Linear 의 입력 step 을 그대로 씁니다
(`fpga_export/DVS128_10/manifest.json`, `0.03014659882`):

```
x_int = clamp(round(log(1+p) / step), -128, 127)
```

### `taps/` — RTL 대조용

사이트별 `.bin` + `manifest.json` (dtype·shape·MAC 호출 순서).

---

## 8. 파일

| | |
|---|---|
| `preprocess.py` | 테스트셋 → 결정론적 보드 입력. 전처리는 `Event_DataModule` 재사용 |
| `golden.py` | 전처리 결과만으로 정수 모델을 돌려 정확도 재현 확인 |
| `taps.py` | 포맷 경계 tap 수집 (훅 + `ATTN_TAP` + `MAC_PROBE`) |

정수 데이터패스는 여기서 **구현하지 않습니다** — `quant_lib/hw_quant.py` 를 부릅니다.
레퍼런스가 두 벌이 되면 어느 쪽이 맞는지부터 다퉈야 합니다.

---

## 9. 이어지는 곳

| | |
|---|---|
| [`04_basic_rtl/fpga_dvs128_10/`](../../../04_basic_rtl/fpga_dvs128_10/) | attention 포함 EvT 가속기 |
| [`04_basic_rtl/fpga_nl/`](../../../04_basic_rtl/fpga_nl/) | 비선형 유닛 검증 (보드 통과) |
| [`../nonlinear_script/`](../nonlinear_script/) | MNIST 기반 비선형 골든 |
| [`../quantization/`](../quantization/) | 양자화 · `hw_flow.md` |
