# GELU Unit — FP32 interface, Residual + PWL-LUT core (RTL)

TSMC28 급 하드웨어용 GELU 활성화 함수 유닛.
**입출력은 FP32(IEEE-754 single)**, 내부는 **Residual + PWL-LUT 고정소수점(Q5.10)** 으로 처리한다.
GELU 를 직접 근사하지 않고 잔차 `R(a)` 만 PWL-LUT 로 근사하는 것이 핵심.

```
x_fp32 ─▶[FP32→Q5.10]─▶ gelu_pwl(core, Q5.10) ─▶[Q5.10→FP32]─▶ y_fp32
```

## 1. 수학적 배경

GELU: `G(x) = x·Φ(x)`,  `Φ` = 표준정규분포 CDF.

`a = |x| (≥0)` 에 대해 **잔차(residual)** 를 정의:

```
R(a) = a − G(a) = a·(1 − Φ(a)) = (a/2)·erfc(a/√2)
```

`Φ(−a) = 1 − Φ(a)` 성질로부터:

```
x ≥ 0 :  G(x) =  a − R(a)   ( = x − R(x) )
x < 0 :  G(x) =    − R(a)   ( = −R(|x|)  )
```

즉 **양수 입력에 대한 `R(a)` 하나만** LUT+PWL 로 구현하면 되고,
부호에 따라 `x−R` 또는 `−R` 을 선택한다.

`R(a)` 는 `0 ≤ R ≤ 0.17` 로 유계이고 `a≳4` 에서 0 으로 소멸하는
작은 bump → 동적범위가 작아 PWL 근사가 매우 정확·저비용이다.
(GELU 전체를 근사하는 것보다 훨씬 유리 — 이것이 residual 방식의 핵심)

## 2. 데이터 포맷

| 신호 | 포맷 | 비고 |
|------|------|------|
| 입력 `x_fp32`, 출력 `y_fp32` | **IEEE-754 FP32** (32-bit) | 유닛 인터페이스 |
| 내부 처리 `x`, `y` | **signed Q5.10** (16-bit) | 범위 [−32, 32), LSB = 2⁻¹⁰ ≈ 9.77e-4 |
| 내부 잔차 `R` | Q?.14 (`base14`,`delta14`) | 정밀도 확보용 |
| PWL 구간 | `a ∈ [0, 4)` 를 **64 세그먼트** 균일분할 | 폭 w = 2⁻⁴ = 0.0625 |
| `a ≥ 4` | `R = 0` 으로 clamp | 양수측 G=x, 음수측 G=0 자연포화 |

**FP32 ↔ Q5.10 변환**
- `FP32→Q5.10` : `round(value·2^10)`, [−32, 32) 포화. `e==0`(zero/subnormal)→0, `e==255`(Inf/NaN)→포화.
- `Q5.10→FP32` : 유효비트 ≤16 < 23 이라 **무손실**(반올림 없음). 정규화 후 `exp = msb+117`.

**코어 세그먼트 인덱싱** (a 는 Q?.10):
```
seg  = a[11:6]                          (0..63)
frac = a[5:0]                           (0..63, 세그먼트 내 위치)
R14  = base14[seg] + (delta14[seg]·frac >>> 6)     // 선형보간
R10  = (R14 + 8) >>> 4                              // Q14→Q10 반올림
G    = sign ? (−R10) : (x − R10)
```

## 3. 아키텍처 (기본 latency = 3 clk, 파라미터로 5까지)

```
 x_fp32 ─[A: FP32→Q5.10]─▶[S1: |x|,seg/frac,LUT조회]─▶[S2: 보간→R]─▶[S3: x−R / −R]─[E: Q5.10→FP32]─ y_fp32
         └ 조합(top) ┘   └──────────────────── gelu_pwl (core, 3단) ───────────────────┘  └ 조합(top) ┘
```

- 코어: 곱셈기 1개(`delta×frac`), 가산기 몇 개, LUT 128워드(64×base + 64×delta, 16-bit). **3-stage 파이프라인**.
- 변환기 Stage A/E는 **조합 로직**: `FP32→Q5.10`(가변 시프트+반올림), `Q5.10→FP32`(leading-one 검출+정규화).
- **latency = 3 (기본)**. `top` 파라미터 `REG_IN`/`REG_OUT` 로 Stage A/E 뒤에 레지스터를 넣으면 latency = 3+REG_IN+REG_OUT (최대 5). 레지스터는 Fmax를 높이고 조합은 latency를 줄이는 트레이드오프.
- **throughput 은 어느 경우든 1 결과/clock** (파이프라인). `in_valid`/`out_valid` 로 스트리밍, 순서 보존.

## 4. 디렉토리 구성

```
GELU/
├── verilog/          # RTL 소스 (합성 대상)
│   ├── top.v            ★top★ FP32 인터페이스 유닛 (변환기+코어 통합)
│   ├── fp32_to_q510.v   FP32 → Q5.10 변환 (조합)
│   ├── q510_to_fp32.v   Q5.10 → FP32 변환 (조합, 무손실)
│   ├── gelu_pwl.v       Residual-PWL GELU 코어 (Q5.10 in/out)
│   └── gelu_lut.vh      자동생성 LUT 계수(base14,delta14) — gelu_pwl.v 에서 include
├── testbench/        # 검증
│   └── tb_gelu_fp32.v   FP32 벡터 스윕, 모델 비트일치 대조, DUT 덤프
├── data/             # 테스트 데이터/레퍼런스 (gen_lut.py 가 생성, 시뮬 입력)
│   ├── vectors_fp32.hex   FP32 입력 벡터
│   ├── model_fp32_out.hex bit-accurate SW 모델 출력(골든)
│   ├── nvec.txt           벡터 개수
│   └── np_xf.npy/np_gtf.npy  analyze/plot 용 (입력 실수 / 실제 GELU)
├── build/            # ★모든 결과물★ (run_vcs.sh 실행 시 생성)
│   ├── simv, csrc/, simv.daidir/   VCS 빌드
│   ├── compile.log / sim.log       컴파일/시뮬 로그
│   ├── dut_out_fp32.hex            VCS DUT 출력(FP32)
│   ├── error_report.txt            analyze.py 오차 리포트
│   └── gelu_result.png             결과 플롯 (true vs DUT 오버레이)
├── gen_lut.py        # LUT 계수 + FP32 테스트벡터 + bit-exact 파이프라인 모델 생성
├── analyze.py        # DUT(FP32) vs 실수 GELU 정밀 오차 리포트
├── plot_gelu.py      # true GELU vs DUT 오버레이 플롯 → build/gelu_result.png
├── explore_pwl.py    # 세그먼트 수 결정용 사전 오차 탐색
├── run_vcs.sh        # 전체 플로우 일괄 실행
└── README.md
```

## 5. 실행

```bash
./run_vcs.sh          # 생성 → VCS 컴파일 → 시뮬 → 오차분석 → 플롯 (결과물 전부 build/)
```

- `run_vcs.sh` 는 **GELU 디렉토리에서 실행**한다 (`cd ~/EventTransformer/RTL/GELU && ./run_vcs.sh`). 실행 시 **RUN DIR** 를 출력한다.
- **모든 결과물은 `build/` 에 저장**된다: `simv`, `dut_out_fp32.hex`, `compile.log`/`sim.log`, `error_report.txt`, `gelu_result.png`.
- 컴파일: `vcs -full64 -sverilog +incdir+verilog verilog/fp32_to_q510.v verilog/q510_to_fp32.v verilog/gelu_pwl.v verilog/top.v testbench/tb_gelu_fp32.v -Mdir=build/csrc -o build/simv`
- `simv` 는 GELU 루트에서 실행한다 (testbench 가 `data/`·`build/` 를 상대경로로 접근). `run_vcs.sh` 가 처리.
- VCS: `W-2024.09-SP2-3`

## 6. 검증 결과 (FP32 벡터 32,018개: 스윕 [−16,16] + 특수값)

- **RTL(top) ↔ bit-accurate 파이프라인 모델 : 0 mismatch** (변환기+코어가 모델과 완전 일치)
- **latency = 3 cycle** (실측: 첫 in_valid→첫 out_valid), **throughput = 1 출력/clock, bubble 없음** (out_valid가 32,018 사이클 연속 high)
- **Python FP32 GELU ↔ HW(top.v) GELU 오차** (절대 오차값, |x|≤31.9):

| 지표 | 절대 오차값 | (참고) LSB=2⁻¹⁰ |
|------|-----|-----|
| **최대 절대오차** | **1.15e-3** | 1.179 LSB |
| 평균 절대오차 | 1.67e-4 | 0.171 LSB |
| RMS 오차 | 2.51e-4 | 0.257 LSB |
| worst-case | x = 0.521 (GELU 기울기 최대 구간): true=0.364083, hw=0.365234 |

**오차 구성**: `입력 FP32→Q5.10 양자화(≤0.5 LSB·기울기) + PWL 근사(~0.4 LSB) + 출력 반올림`.
FP32↔Q5.10 중 **출력 복원은 무손실**이고, 고정소수점 코어 자체 오차는 ~0.85 LSB.
FP32 입력을 격자로 반올림하는 단계가 기울기가 큰 `x≈0.5` 부근에서 오차를 최대로 만든다.
표현범위 [−32, 32) 밖은 설계상 포화.

`build/gelu_result.png` — (상단) 실제 GELU(검정) vs DUT 출력(빨강 점선) 오버레이,
(하단) `base(R)`/`delta`가 `|x|`보다 값 범위가 훨씬 작아(1.0 미만, 정수비트 불필요)
소수부를 넓힌 **Q?.14**를 쓰는 이유 시각화:

![result](build/gelu_result.png)
