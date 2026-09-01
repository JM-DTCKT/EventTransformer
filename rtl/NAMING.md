# 명명 규칙 사전

이 문서가 **정본**입니다. `rtl/` · `tb/` · `sw/` · `data/` · `tcl/` · `board/` · README 가
모두 여기 용어를 따릅니다.

| 무엇을 하려는가 | 볼 곳 |
|---|---|
| 새 신호 이름을 짓는다 | [§2 접두사·접미사](#2-접두사--접미사) |
| `op_ostr` 같은 이름이 무슨 뜻인지 본다 | [§5 신호 사전](#5-신호-사전) |
| 명령어 워드 비트를 해석한다 | [§4 명령어 워드](#4-명령어-워드-256비트) |
| 옛 코드·git 이력·옛 로그와 대조한다 | [§7 옛 이름 대조표](#7-옛-이름-대조표) |
| 고친 뒤 동작이 그대로인지 확인한다 | [§8 회귀 검증](#8-회귀-검증) |

> 리팩토링 기준일 **2026-08-31**. 동작은 바뀌지 않았습니다 — `tb/run_vcs.sh` 회귀
> 로그가 리팩토링 직전과 **비트 단위로 동일**합니다.

---

## 1. 용어

### 1.1 `step` 은 이제 안 씁니다

한 낱말이 세 가지 뜻으로 쓰이고 있어서 갈라 놓았습니다.

| 옛 낱말 | 새 용어 | 뜻 | 대표 신호 |
|---|---|---|---|
| step | **`inst`** | 프로그램 한 줄 (256비트 명령어) | `inst_ptr` `inst_addr` `inst_word` `Inst_Mem` |
| step | **`tstep`** | 타임스텝 (time-window) | `tstep` `n_tstep` `tstep_idx` `ST_TSTEP` |
| step | **`scale`** | 양자화 눈금 (스케일 상수) | `Requant_Bf16.scale`, RES 의 정수 scale |

### 1.2 이 설계 고유 용어

| 용어 | 뜻 | 값·범위 |
|---|---|---|
| `tok` | 토큰 | `n_tok` = 이번 타임스텝의 토큰 수 (실측 16~123) |
| `lane` | 워드 안의 슬롯 | 32개. GEMM 에서 A 는 행, B 는 열 |
| `mt` / `nt` | 행 타일 / 열 타일 번호 | 타일 = 32행 또는 32열 |
| `col` | GEMM 이 뱉는 컬럼 | 출력채널 하나 × 32행 |
| `Lk` | attention 의 키 개수 | `n_tok + 1` (마지막 하나가 bias 토큰) |
| `bkv` | 학습된 bias_k / bias_v 상수 토큰 | 별도 BKV 영역 |
| `af` | affine — LayerNorm 의 gamma/beta | → `Affine_Mem` |
| `rq` | requantize — 재양자화 | → `Requant_Mem` |
| `fca` | `Format_Cast_Act` (컬럼 → 출력 포맷) | 구 `Col_Post` |

---

## 2. 접두사 · 접미사

### 2.1 접두사

| 접두사 | 뜻 | 예 |
|---|---|---|
| `op_*` | 디코드된 명령어 필드 (`ST_DECODE` 에서 래치, 명령어 내내 고정) | `op_kind` `op_ain` |
| `a_ra_*` / `a_rb_*` | A_Mem 읽기 포트 A / B (같은 내용을 미러링한 두 벌) | `a_ra_addr` |
| `a_wr_*` / `a_we_*` | A_Mem 쓰기 — `a_wr_*` 는 로더와 먹싱된 최종단 | `a_wr_en` |
| `rq_*` | Requant_Mem 및 그 출력 상수 | `rq_scale_q` |
| `af_*` | Affine_Mem 및 그 출력 상수 | `af_gamma` |
| `gemm_*` / `col_*` | GEMM 코어 제어 / 코어가 내는 컬럼 | `gemm_start` `col_valid` |
| `fca_*` | `Format_Cast_Act` 출력과 그 지연 파이프 | `fca_data` |
| `tr_*` | `Transpose32` 채우기 · 쏟기 | `tr_drain` |
| `ln_* smax_* mean_* argmax_* rs_* pos_*` | 해당 유닛 전용 | `ln_valid` |
| `ld_*` / `LD_*` | AXI-Stream 로더 경로 / 목적지 상수 | `ld_sel` `LD_RQ` |
| `ST_*` | FSM 상태 상수 (레지스터 이름은 항상 `state`) | `ST_DECODE` |
| `OP_*` / `FMT_*` | 명령어 KIND / 출력 포맷 상수 | `OP_GEMM` `FMT_Q69` |
| `g_*` | generate 블록 라벨 (소문자) | `g_res_lane` |
| `u_*` | 모듈 인스턴스 | `u_gemm` `u_rq_mem` |

### 2.2 접미사

| 접미사 | 뜻 | 예 |
|---|---|---|
| `_q` | 레지스터 한 단 (타이밍용 컷) | `rq_scale_q` |
| `_d1` `_d2` | N 사이클 지연 — 숫자가 곧 단수 | `col_valid_d2` |
| `_c` | 조합 값 (같은 이름의 `_q` 와 짝) | `diff_c` |
| `_early` | 한 사이클 **앞당긴** 탭 | `fca_n_early` |
| `_last` | 마지막 인덱스 | `row_tile_last` `beat_last` |
| `_en` `_addr` `_data` | 메모리 포트 3종 | `rd_en` `we_addr` |

### 2.3 쓰지 말 것

| 금지 | 이유 | 대신 |
|---|---|---|
| `q_*` 접두사 | `_q` 접미사(레지스터)와 뜻이 충돌 | 명령어 필드는 `op_*` |
| 한두 글자 약칭 (`sp` `ti` `gc` `sf` `mn_` `am_`) | 뜻이 안 드러남 | `inst_ptr` `tstep` `mean_*` `argmax_*` |
| 파일명과 다른 모듈명 | 찾기가 안 됨 | 파일 하나 = 모듈 하나 = 같은 이름 |
| 내용을 안 드러내는 약칭 (`PB` `PG`) | 무슨 메모리인지 모름 | `Requant_Mem` `Affine_Mem` |

---

## 3. 메모리

| 메모리 | 인스턴스 | `ld_sel` | 담는 것 | 주소폭 | 이미지 파일 |
|---|---|---|---|---|---|
| W_Mem | `u_w_mem` | `LD_W` (0) | 가중치 int8 | `AW_W` = 14 | `data/wmem.bin` |
| A_Mem | `u_a_mem0` `u_a_mem1` | `LD_A` (1) | 활성값 (미러 2벌 = 읽기 2포트) | `AW_A` = 13 | 런타임 |
| **Requant_Mem** | `u_rq_mem` | `LD_RQ` (2) | 채널별 `{mult, bias}` | `AW_RQ` = 10 | `data/rqmem.bin` |
| **Affine_Mem** | `u_af_mem` | `LD_AF` (3) | 특징별 `{gamma, beta}` | `AW_AF` = 8 | `data/afmem.bin` |
| **Inst_Mem** | `u_inst_mem` | `LD_INST` (4) | 명령어 프로그램 | `AW_INST` = 8 | `data/instmem.bin` |
| POS 표 | `u_pos.pos_tbl` | `LD_POS` (5) | positional encoding 441행 | `AW_T` = 9 | `data/posmem.bin` |

`ld_sel` 인코딩은 `EvT_Engine.LD_*` · `Axis_Loader.LD_*` · `Top.v` 의 `LOAD_SEL`
레지스터가 **모두 같은 값**입니다.

---

## 4. 명령어 워드 (256비트)

```
[ 31: 0] KIND[3:0] FMT[5:4] ACT[7:6] VAR[11:8] FLAG[15:12]
         SHIFT[21:16] SHIFT2[27:22] FLAG2[31:28]
[ 63:32] M       [ 95:64] K       [127:96] NOUT
[159:128] AIN    [191:160] BIN    [223:192] AOUT
[255:224] RQ_BASE[15:0] | OSTR[31:16]
```

### 4.1 필드

| 필드 | 레지스터 | 뜻 |
|---|---|---|
| KIND | `op_kind` | `OP_GEMM` `OP_LN` `OP_SMAX` `OP_RES` `OP_MEAN` `OP_ARGMAX` `OP_POS` |
| FMT | `op_fmt` | 출력 포맷 `FMT_INT8` `FMT_Q411` `FMT_BF16` `FMT_Q69` |
| ACT | `op_act` | 활성함수 선택 (`Activation.v` 인코딩) |
| VAR | — | `n_tok` 으로 채울 필드 표시. `ST_DECODE` 가 `inst_word` 비트를 직접 봅니다 |
| FLAG | `op_flag` | 아래 §4.2 |
| FLAG2 | `op_flag2` | 아래 §4.2 |
| SHIFT | `op_shift` | 1차 재양자화 시프트 |
| SHIFT2 | `op_shift2` | **다의어** — §4.3 |
| M · K · NOUT | `op_m` `op_k` `op_nout` | GEMM 형상 |
| AIN · BIN · AOUT | `op_ain` `op_bin` `op_aout` | 피연산자 · 결과 베이스 |
| RQ_BASE | `op_rq_base` | **다의어** — §4.3 |
| OSTR | `op_ostr` | **다의어** — §4.3 |

### 4.2 FLAG 비트

| 비트 | FLAG (`op_flag`) | FLAG2 (`op_flag2`) |
|---|---|---|
| [0] | GEMM 출력을 `Transpose32` 경유로 (in_proj 의 V) | LayerNorm 입력이 Q4.11 정수 코드 |
| [1] | GEMM 출력을 head-major 주소로 (in_proj 의 Q/K) | RES 두 피연산자가 정수 코드 |
| [2] | B 피연산자를 A_Mem 에서 (Q·Kᵀ, attn·V) | bias_k / bias_v 토큰 끼워넣기 |
| [3] | Q4.11 을 재양자화 없이 16b 그대로 (raw16) | 활성함수 뒤 2차 재양자화 |

### 4.3 다의어 필드 — 쓰는 자리마다 별칭을 답니다

`EvT_Engine` 안에 이름 붙인 wire 를 두었습니다.
**새 용도를 추가하면 별칭도 같이 추가하십시오.**

| 필드 | 어느 KIND 에서 | 별칭 (읽는 자리의 이름) | 뜻 |
|---|---|---|---|
| OSTR | GEMM | `op_ostr` | 출력 stride (보통 NOUT, `event_projection` 은 160) |
| OSTR | GEMM + FLAG[1] | `op_ostr` | head 간 간격 |
| OSTR | GEMM + Q6.9 | `ostr_as_smax_base` | softmax 출력 베이스 |
| OSTR | GEMM (attn·V) | `ostr_as_bkv_addr` | bias_v 워드 주소 |
| OSTR | LN | `ostr_as_af_base` | Affine_Mem 베이스 |
| OSTR | RES | `ostr_as_scale_b` | 피연산자 B 의 정수 scale |
| OSTR | POS | `op_ostr` | PIN stride (160) |
| RQ_BASE | GEMM · ARGMAX | `op_rq_base` | Requant_Mem 채널 테이블 베이스 |
| RQ_BASE | RES | `rq_base_as_scale_a` | 피연산자 A 의 정수 scale |
| SHIFT2 | GEMM (Q4.11) | `op_shift2` | GELU 뒤 2차 재양자화 시프트 |
| SHIFT2 | LN | `op_shift2` | 코어의 고정소수점 창 (`in_shift`) |
| SHIFT2 | MEAN | `op_shift2` | 재양자화 시프트 |

> GELU 뒤 2차 재양자화 **곱수**는 별도 필드 없이 `RQ_BASE + NOUT` 칸에 있습니다.

---

## 5. 신호 사전

### 5.1 제어 · 명령어

| 신호 | 뜻 |
|---|---|
| `state` | 메인 FSM 레지스터 |
| `ST_*` | `IDLE`→`TLOAD`→`FETCH`→`DECODE`→`CONST`→`RUN`→`WAIT`→`NEXT`→`TSTEP`/`DONE` |
| `inst_ptr` | 명령어 포인터 (프로그램 카운터) |
| `inst_addr` / `inst_word` | Inst_Mem 주소 / 읽어온 256비트 명령어 |
| `n_body` / `n_tail` | 타임스텝당 명령어 수 (118) / 끝에 한 번 도는 수 (5) |
| `in_tail` | body 를 다 돌고 tail 을 도는 중 |
| `const_ph` | `ST_CONST` 의 3사이클 위상 (Requant_Mem · bias_k 읽기) |
| `wait_ack` | 하위 코어의 `done` 이 한 번 0 으로 내려간 것을 본 뒤부터 인정 |

### 5.2 타임스텝 핸드셰이크

| 신호 | 방향 | 뜻 |
|---|---|---|
| `tstep` / `n_tstep` | 내부 / in | 현재 타임스텝 / 전체 타임스텝 수 T |
| `tstep_idx` | out | 호스트가 채워야 할 타임스텝 번호 |
| `tok_req` / `tok_ack` | out / in | 멈춤 신호 / 적재 완료 통보 |
| `tok_n` → `n_tok` | in | 이번 타임스텝의 토큰 수 (`tok_ack` 에서 래치) |

### 5.3 GEMM 경로

| 신호 | 뜻 |
|---|---|
| `gemm_start` / `gemm_done` | 코어 기동 / 완료 |
| `col_valid` `col_data` `col_n` `col_mt` | 코어가 내는 컬럼 (`_d1` `_d2` 는 상수 정렬용 지연) |
| `gemm_a_rd_*` / `gemm_b_rd_*` | 코어가 내는 A / B 읽기 주소 |
| `b_src_amem` | B 를 A_Mem 에서 읽는가 (`op_flag[2]`) |
| `gemm_b_from_w` → `gemm_b_from_a` → `gemm_b_patched` → `gemm_b_data` | B 피연산자 먹싱 4단계 |
| `gemm_we_addr_q` | 한 사이클 앞서 계산해 둔 A_Mem 쓰기 주소 |

### 5.4 Format_Cast_Act (컬럼 후처리)

| 신호 | 뜻 |
|---|---|
| `fca_valid` `fca_data` `fca_q69` | 출력 (Q6.9 는 softmax 직결) |
| `fca_n` `fca_mt` | 그 컬럼의 채널 / 행타일 |
| `fca_n_early` `fca_mt_early` | 쓰기 주소 계산용 **한 칸 앞선** 탭 |
| `fca_busy` `fca_pipe` | 파이프가 아직 값을 들고 있는가 |
| `FCA_LAT_BF16` `_INT8` `_GELU` `_REQ2` | 출력 포맷별 지연 = **2 / 3 / 9 / 6** |
| `rq_scale` → `rq_scale_q` | **1차** 재양자화 scale — 컬럼마다 바뀜 (`RQ_BASE + n`) |
| `rq_bias` → `rq_bias_q` | 채널별 bias (scale 과 같은 워드) |
| `rq_scale2_q` | **2차** 재양자화 scale — 명령어당 하나 (`RQ_BASE + NOUT`) |

### 5.5 유닛별

| 유닛 | 신호 |
|---|---|
| LayerNorm | `ln_start` `ln_done` `ln_valid` `ln_mt` `ln_k` `ln_af_addr` `af_gamma` `af_beta` |
| Softmax | `smax_start` `smax_done` `smax_valid` `smax_col` `smax_in_valid` `smax_in_data` |
| Transpose32 | `tr_we` → `tr_last` → `tr_last_d1` → `tr_last_d2` → `tr_drain` (채우기 → 쏟기 파이프), `tr_row` `tr_head` `tr_mt` |
| RES (잔차) | `rs_run` `rs_ph` `rs_a` `rs_b` `rs_mt` `res_is_int` |
| MEAN | `mean_run` `mean_k` `mean_mt` `mean_ph` `mean_acc` `mean_out` |
| ARGMAX | `argmax_valid_q` `argmax_acc_q` `argmax_val_q` `argmax_best` `argmax_any` |
| bias 토큰 | `bkv_is_qk` `bkv_is_av` `qk_hit` `av_hit` `bias_k_word` `b_key_lim` |
| 공통 | `row_tile_last` = `(op_m-1) >> 5` — RES · MEAN 이 공유 |

---

## 6. 모듈 · 파일

### 6.1 규칙

| 규칙 | 내용 |
|---|---|
| 파일 ↔ 모듈 | 파일 하나에 모듈 하나, **이름이 같습니다** |
| 표기 | `PascalCase_Snake` — 헤더(`.vh`)도 같습니다 |
| 최상위 | `Top` (`rtl/Top.v`). `tcl/build.tcl` 이 `-reference Top` 으로 참조 |
| 래퍼 / 코어 | `*_Top` = 엔진 인터페이스, `*_Unit` = 연산 코어 |
| 파라미터 | 전부 대문자 (`LANE` `PSUM_W` `AW_A` `N_CLASS`) |
| FSM | `ST_*` + `state`. 스테이지별 독립 FSM 인 `LayerNorm_Unit` `Softmax_Unit` 만 `SS_*`/`MS_*`/`ES_*` |
| generate 라벨 | 소문자 `g_*` |
| 파일 머리 | 전부 `` `timescale 1ns/1ps ``. 주석 틀은 §6.2 |

### 6.2 주석 규약

모든 파일이 같은 틀을 씁니다.

```
// -----------------------------------------------------------------------------
// Module_Name : 한 줄 요약
//
// 본문 (무엇을 / 왜 이렇게)
//
// ## 소제목
//
// ...
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
```

| 요소 | 규칙 |
|---|---|
| 머리말 상자 | `// ---` 79자 규칙선, 소제목은 `## ` |
| 본문 구획 | 코드 안에서는 들여쓴 `// ===` 상자 |
| 언어 | 한국어. 수식·포맷 표기는 원문 그대로 |
| **`[타이밍]`** | 파이프라인 단을 왜 넣었는지. **지우면 Fmax 가 깨집니다** |
| **`[함정]`** | 실제로 물렸던 버그. 되돌리면 재발합니다 |

쓰지 않는 것:

| 안 씁니다 | 이유 |
|---|---|
| 이 저장소에 없는 프로젝트 참조 | 따라갈 수 없는 링크 |
| "예전에는 …였습니다" 서술 | git 이력에 있습니다. 남길 것은 **제약**이지 경위가 아닙니다 |
| 특정 툴 실행의 실측 수치 나열 | 경로 이름만 남기고 수치는 한 개까지 |
| 진행 상태 ("아직 검증 안 함") | 금방 낡습니다 |

### 6.3 모듈 목록 (29개)

| 디렉토리 | 모듈 = 파일 | 역할 |
|---|---|---|
| `rtl/` | `Top` | 보드 최상위 — AXI4-Lite + AXI-Stream |
| `rtl/core/` | `EvT_Engine` | 명령어 프로그램 실행기 |
| | `Format_Cast_Act` | 컬럼 → 출력 포맷 4종 + 활성함수 |
| | `Pos_Gather` | positional encoding 온칩 게더 |
| | `Transpose32` | 32×32 corner-turn (V 전용) |
| | `Bram_Sdp` | simple dual-port BRAM |
| `rtl/gemm_core/` | `Gemm_Core` | 32×32 output-stationary systolic GEMM |
| | `PE_Array` | ping-pong mesh |
| | `PE_OS` | DSP48E2 + shadow 누산기 |
| | `Skew_Buf` | systolic 정렬용 삼각 지연 |
| `rtl/axi/` | `Axis_Loader` | DMA → 온칩 메모리 6종 |
| | `Axis_Dump` | A_Mem → DMA |
| `rtl/layernorm/` | `LayerNorm_Top` | 래퍼 — A_Mem 읽기 + affine + 재양자화 |
| | `LayerNorm_Unit` | 코어 — 32행 Tile 단위 정규화 |
| | `Rsqrt_Unit` | 1/sqrt(var) |
| | `LayerNorm_Affine` | gamma/beta 적용 |
| `rtl/softmax/` | `Softmax_Top` | 래퍼 — 길이 세기 + Q1.14 → uint8 |
| | `Softmax_Unit` | 코어 — 32행 Tile 단위 softmax |
| | `Exp2_Unit` `Recip_Unit` | exp / 역수 |
| `rtl/gelu/` | `Gelu_Pwl` | 64세그먼트 PWL GELU |
| `rtl/requant/` | `Requant_Int` `Requant_Bf16` | 정수 / bf16 재양자화 |
| `rtl/vector_alu/` | `Bf16_To_Fix` `Q411_To_Fix` | → 고정소수점 |
| | `Int32_To_Bf16` `Fp32_To_Bf16` `Fp32_Add` | bf16 / fp32 산술 |
| `rtl/activation/` | `Activation` | ReLU / Leaky / Clamp |

### 6.4 포함 헤더 (include)

| 헤더 | 쓰는 모듈 |
|---|---|
| `gelu/Gelu_Lut.vh` | `Gelu_Pwl` |
| `layernorm/Rsqrt_Lut.vh` | `Rsqrt_Unit` |
| `softmax/Exp2_Lut.vh` `Recip_Lut.vh` | `Exp2_Unit` `Recip_Unit` |

> `tcl/build.tcl` 이 이 넷을 **Verilog Header 파일 타입**으로 프로젝트에 넣습니다.
> 빠지면 `create_bd_cell` 이 통째로 실패합니다.

---

## 7. 옛 이름 대조표

`rtl/` · `tb/` · `sw/` · `data/` · `tcl/` · `board/` · README 까지 전부 맞춰 두었습니다.
아래는 git 이력이나 옛 로그를 읽을 때 쓰는 표입니다.

### 7.1 용어

| 옛 | 새 |
|---|---|
| `step` (프로그램 한 줄) | `inst` |
| `step` (타임스텝) | `tstep` |
| `step` (양자화 눈금) | `scale` |
| `CONS` (consumer) | `FMT` |
| `GSH` | `SHIFT2` |
| `PB` / `PG` | `RQ_BASE` / (Affine — 베이스는 `OSTR` 이 실어 옴) |

### 7.2 RTL 신호

| 옛 | 새 |
|---|---|
| `sp` `dbg_step` | `inst_ptr` `dbg_inst` |
| `ti` `n_time` | `tstep` `n_tstep` |
| `q_kind` `q_cons` `q_M` `q_AIN` `q_PB` `q_OSTR` … | `op_kind` `op_fmt` `op_m` `op_ain` `op_rq_base` `op_ostr` … |
| `cp_v` `cp_d` `cp_n` `CP_LAT_*` | `fca_valid` `fca_data` `fca_n` `FCA_LAT_*` |
| `ar_*` / `br_*` | `a_ra_*` / `a_rb_*` |
| `gm_start` `gm_done` `col_v` `col_d` | `gemm_start` `gemm_done` `col_valid` `col_data` |
| `sm_*` (`u_sm` `sm_ov` `sm_c`) | `smax_*` (`u_smax` `smax_valid` `smax_col`) |
| `mn_*` / `am_*` | `mean_*` / `argmax_*` |
| `st` / `S_*` | `state` / `ST_*` |
| `rq_mult` `rq_mult_q` `rq_mult2_q` | `rq_scale` `rq_scale_q` `rq_scale2_q` |
| `tr_run` `tr_arm` `tr_go` | `tr_drain` `tr_last_d1` `tr_last_d2` |

### 7.3 모듈 · 파일

| 옛 | 새 |
|---|---|
| `top` `top.v` (소문자) | `Top` `Top.v` |
| `Col_Post` | `Format_Cast_Act` |
| `PE_Array_Pp` `PE_OS_Pp` | `PE_Array` `PE_OS` |
| `layernorm_top` `layernorm_unit` | `LayerNorm_Top` `LayerNorm_Unit` |
| `softmax_top` `softmax_unit` | `Softmax_Top` `Softmax_Unit` |
| `rsqrt_unit` `exp2_unit` `recip_unit` | `Rsqrt_Unit` `Exp2_Unit` `Recip_Unit` |
| `bf16_to_fix` `gelu_pwl` `LN_Affine` | `Bf16_To_Fix` `Gelu_Pwl` `LayerNorm_Affine` |
| `gelu_lut.vh` `rsqrt_lut.vh` `exp2_lut.vh` `recip_lut.vh` | `Gelu_Lut.vh` `Rsqrt_Lut.vh` `Exp2_Lut.vh` `Recip_Lut.vh` |
| `tb_colpost_ev` | `tb_format_cast_act` |

### 7.4 포트 · 파라미터

| 옛 | 새 |
|---|---|
| `Softmax_Top.C` / `.out_c` | `.n_col` / `.out_n` |
| `Softmax_Unit.Tile_M` | `.LANE` |
| `LayerNorm_Top.p_addr` `.p_gamma` `.p_beta` | `.af_addr` `.af_gamma` `.af_beta` |
| `EvT_Engine.GELU_LUT_FILE` | (삭제 — `Gelu_Pwl` 은 `` `include `` 로 상수를 들고 있음) |

### 7.5 호스트 · 데이터

| 옛 | 새 |
|---|---|
| `data/stepmem.bin` `.hex` | `data/instmem.bin` `.hex` |
| `data/pbmem.*` `data/pgmem.*` | `data/rqmem.*` `data/afmem.*` |
| `sw/golden_steps.py` `--step_t` | `sw/golden_insts.py` `--tstep` |
| config.json `pb_base` `pg_base` `attn_pb` `ln_pb` | `rq_base` `af_base` `attn_rq` `ln_rq` |
| config.json `input_steps` `input_step` | `input_scales` `input_scale` |
| config.json `pos_encoding.step` `.table_step` | `.scale` `.table_scale` |
| schedule.json `steps` | `insts` |
| schedule.json `CONS` `GSH` `PB` `GPB` | `FMT` `SHIFT2` `RQ_BASE` `GRQ_BASE` |
| main_evt.c `SEL_PB/PG/S` `DDR_PB/PG/STEP` | `SEL_RQ/AF/INST` `DDR_RQ/AF/INST` |
| AXI 레지스터 `N_TIME` | `N_TSTEP` |

---

## 8. 회귀 검증

RTL 을 고친 뒤에는 **반드시** 돌리고 변경 전 로그와 diff 하십시오. TB 들이 내부
신호를 그대로 찍기 때문에 **로그가 같으면 사이클 단위로 같습니다.**

```
tb/run_vcs.sh <tb>          # 로그는 tb/build/<tb>.run.log
```

### 8.1 기준 결과

| TB | 검사 수 | 기대 |
|---|---|---|
| `tb_evt` | 140,170 | **2건 실패가 정상** (TFFN embs.linear1 #10021, TCLF1 clf.linear_1 #105) |
| `tb_evt_head` | 53,888 | PASS |
| `tb_transpose32` | 205,824 | PASS |
| `tb_gemm_ev` | 8,160 | PASS |
| `tb_softmax_attn` | 1,696 | PASS |
| `tb_accel_evt` | 13 | PASS (86,780 사이클) |
| `tb_format_cast_act` | 4 | PASS |
| `tb_ln_wrap` `tb_smx_wrap` `tb_evt_compile` | — | PASS |
| `tb_evt_t` `tb_evt_prof` | — | 10분을 넘어 상시 회귀에서 제외 |

> **TB 들이 `dut.<내부신호>` 를 다수 참조합니다** (`tb_evt` `tb_evt_head`
> `tb_evt_prof` 가 특히 많이). RTL 신호명을 바꾸면 `tb/` 도 같이 고쳐야 합니다.

### 8.2 무엇을 고쳤을 때 무엇을 돌리나

| 고친 곳 | 확인 방법 |
|---|---|
| `rtl/` `tb/` | 위 회귀 전부 + 변경 전 로그와 diff |
| `sw/` | 이미지를 다시 만들어 `data/` 와 `cmp` (§8.3) |
| 문법만 빠르게 | `xvlog` + `xelab` (§8.4) |
| `tcl/build.tcl` · 모듈명 | Vivado 프로젝트 폴더를 지우고 BD 부터 다시 (§8.5) |

### 8.3 `sw/` — 이미지가 그대로인지

외부 manifest 가 있어야 합니다.

```
cd sw
python3 pack_evt.py --dst /tmp/d
python3 schedule_evt.py --dst /tmp/d
cmp /tmp/d/rqmem.bin ../data/rqmem.bin   # instmem / afmem / wmem / … 전부
```

### 8.4 문법 · 엘라보레이션만 (셸에서 `v22`)

```
xvlog -i rtl/gelu -i rtl/softmax -i rtl/layernorm rtl/*.v rtl/*/*.v
xvlog $XILINX_VIVADO/data/verilog/src/glbl.v      # 안 넣으면 xelab 이 glbl 을 못 찾습니다
xelab -L unisims_ver -L unisim Top glbl
```

### 8.5 Vivado 재빌드

`tcl/build.tcl` 이 BD 를 매번 새로 만듭니다 (`-reference Top`).
**기존 프로젝트 폴더는 지우고 돌리십시오.**
