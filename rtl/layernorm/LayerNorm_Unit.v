// -----------------------------------------------------------------------------
// LayerNorm_Unit : 32행 Tile 단위 D축 LayerNorm  (BF16 in / signed Q4.11 out)
//
// EvT 의 nn.LayerNorm([embed_dim]) 을 그대로 구현한다.  정규화 축은 **마지막
// 차원(embed_dim = D = 128)** 이고, 토큰(latent embedding) 96개가 "몇 번
// LayerNorm 하느냐" 다.  이를 LANE = 32 행씩 3 Tile 로 나눠 처리한다.
//
//    for r in 0..95:  y[r][0..D-1] = (x[r][.] - mu[r]) / sqrt(var[r] + eps)
//    mu[r]  = (1/D) * sum_d x[r][d]
//    var[r] = (1/D) * sum_d x[r][d]^2 - mu[r]^2          (biased, PyTorch 와 동일)
//
// ## 입력
//
// 매 clk "한 열" = 32행 각각의 원소 1개 (LANE x 16b = 512b, BF16)
// · 열 하나가 32행에 1원소씩이므로 부분 beat 가 없다 -> keep 마스크 불필요
// · D 는 컴파일타임 상수(2의 거듭제곱)라 1/D 가 시프트로 끝난다. in_last 불필요.
//
// ## 수신하면서 sum / sumsq 를 같이 구한다
//
// 데이터가 어차피 버스를 지나가므로 **제곱기 32개 + 누산기 2벌**만 얹으면
// 추가 사이클·추가 메모리 읽기 없이 행별 통계가 확정된다.  softmax 가 max 를
// 수신 중에 구한 것과 같은 구조다.
//
// ## 분산을 **정수 그대로** 구해 상쇄오차를 원천 제거
//
// X 를 24b 정수 표현(x = X*2^-IF)이라 하면
//       SX = sum X            (31b signed, 정확)
//       SQ = sum X*X          (54b unsigned, 정확)
//       var = SQ*2^-(2IF+DLOG) - (SX*2^-(IF+DLOG))^2
//           = 2^-(2IF+2DLOG) * ( (SQ << DLOG) - SX*SX )
// 괄호 안이 **오차 없는 정수식**이다.  E[x^2]-mu^2 형태는 보통 mu >> sigma 일 때
// 치명적 상쇄가 나지만, 여기서는 두 항 모두 정확한 정수라 뺄셈도 정확하고
// Cauchy-Schwarz 에 의해 결과가 **절대 음수가 되지 않는다**.
// (mu 는 SX 를 소수점만 바꿔 읽은 값이라 나눗셈이 아예 없다: mu = SX * 2^-MUF)
//
// ## 곱하기 전에 시프트 (DSP 1개/lane)
//
// y = (x - mu) * rstd,  rstd = r * 2^-RF * 2^-e   (r in (0.5,1], e 는 행마다 다름)
// d = x - mu 는 33b 라 곱셈기가 33x18 (DSP 2개) 이 되어버린다.  대신 e 시프트를
// **곱셈 앞으로** 옮기면 ds = d*2^-e 의 크기가 |ds| = |y|/r < 2|y| <= 32 로 묶여
// 25b(Q6.18) 면 충분해진다 -> 25x18 = DSP48 **1개**에 정확히 들어간다.
// (softmax 처럼 곱 뒤에 시프트하면 44b 를 시프트해야 해서 더 비싸다)
//
// ## 3-stage Tile 파이프라인
//
// 한 Tile 안에서는 mu/var 가 전역 의존성이라 RECV 와
// NORM 을 겹칠 수 없다.  대신 **서로 다른 Tile** 을 세 단계가 동시에 처리한다.
//
//    Tile k+2 :  [S1 RECV  D=128 clk ]
//    Tile k+1 :                [S2 STAT  1+2+LANE+3 = 38 clk ]
//    Tile k   :                        [S3 NORM  1+D+L+3 = 133 clk ]
//               주기 P = max(128, 38, 133) = 133 clk        (L = SRAM_LAT = 1)
//
// ## 슬롯을 몇 개 두는가
//
// xbuf 는 S1 이 쓰고 **S3 가** 읽는다.  그 사이에 S2 가
// 끼어 있으므로 한 Tile 의 버퍼 점유는 S1+S2+S3 = 299 clk 이고, 이는 2P=266 보다
// 크다.  따라서 **슬롯 3개** 가 필요하다 (3P = 399 >= 299).
// 슬롯 하나 = D word x (LANE*16b) = 64 kbit  ->  합계 196 kbit.
// · 버퍼에는 **변환 전 BF16(16b)** 을 그대로 담는다.  고정소수점(24b)으로
//     담으면 98 kbit 가 더 드는데, 대신 S3 쪽 변환기 32개(~5k gate)면 끝난다.
//     SRAM 이 훨씬 비싸므로 BF16 저장이 이득이다.  두 변환기는 같은 모듈이라
//     S1 이 누산한 X 와 S3 가 재생성한 X 가 **비트 단위로 동일**하다.
//
// ## 고정소수점 포맷
//
// in_col   BF16              16b   입력 (FP32 상위 16b)
// X (내부) signed Q8.15       24b   범위 [-256, 256), LSB 3.05e-5
// SX       signed             31b   sum X                  (정확)
// SQ       unsigned           54b   sum X*X                (정확)
// V        unsigned           61b   var*2^44 + eps*2^44    (정확)
// mu       signed Q8.22       31b   = SX 를 소수점만 바꿔 읽은 값
// r        unsigned UQ1.17    18b   (m*2^par)^-0.5,  범위 (0.5, 1]
// e        signed              6b   rstd = r*2^-17 * 2^-e
// d        signed Q10.22      33b   x - mu
// ds       signed Q6.18       25b   d * 2^-e            (포화)
// out_col  **signed Q4.11**   16b   범위 [-16, 16), LSB 4.88e-4
//
// y_out = (ds * r + 2^23) >> 24        (24 = DSF + RF - OF)
//
// ## affine(gamma/beta) 는 왜 없는가
//
// EvT 의 LayerNorm 은 **전부 뒤에 Linear 가 붙는다** (layer_norm_x -> attention
// 의 Q/K/V projection, layer_norm_2 -> linear2, ...).  따라서
//       W*(gamma (*) xhat + beta) + b = (W*diag(gamma))*xhat + (W*beta + b)
// 로 gamma/beta 를 **다음 Linear 의 가중치·바이어스에 접어 넣을 수 있다**.
// 하드웨어에 넣으면 lane 당 곱셈기 1개 + 파라미터 메모리가 더 드는데,
// 오프라인 상수 접기로 공짜가 되므로 코어는 정규화만 한다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module LayerNorm_Unit #(
    parameter integer LANE   = 32,   // Tile 행 수 (= GEMM 코어 타일 행 수)
    parameter integer D      = 128,  // 정규화 축 길이 (= embed_dim, 2^DLOG)
    parameter integer DLOG   = 7,    // log2(D)
    parameter integer IW     = 24,   // 내부 고정소수점 폭 (signed Q8.15)
    parameter integer IF     = 15,   // 내부 고정소수점 소수부
    parameter integer OW     = 16,   // 출력 폭 (signed Q4.11)
    parameter integer OF     = 11,   // 출력 소수부
    parameter integer RW     = 18,   // rstd 가수 폭 (UQ1.17, = RF+1)
    parameter integer RF     = 17,   // rstd 가수 소수부
    parameter integer DSW    = 25,   // ds 폭 (signed Q6.18) — DSP A 포트 25b
    parameter integer DSF    = 18,   // ds 소수부
    parameter integer XSW    = 6,    // in_shift 폭 (signed)
    // eps * 2^VF (VF = 2*IF+2*DLOG = 44).  기본 = round(1e-5 * 2^44) = PyTorch 기본값
    parameter integer EPS_INT = 175921860,
    // MAXSHL : 좌시프트 여유.  e 최소값이 floor((floor(log2 EPS_INT) - VF)/2) = -9
    //          이므로 MAXSHL = -(e_min + MUF - DSF) = 9 - 4 = 5
    parameter integer MAXSHL = 5,
    // SRAM read latency : 1 = 코어만, 2 = 출력 레지스터까지 (고Fmax 권장)
    parameter integer SRAM_LAT = 1,
    parameter integer NB     = 3     // Tile 슬롯 수 (>= 3)
)(
    input                        clk,
    input                        rst_n,
    // ---- 입력 : 매 clk 한 열(LANE원소, BF16), D 회 ----
    input                        in_valid,
    output                       in_ready,
    input       [LANE*16-1:0]    in_col,     // BF16 x LANE (행 r = in_col[r*16 +: 16])
    input  signed [XSW-1:0]      in_shift,   // 이 Tile 의 고정소수점 창 이동 (첫 beat 에서 래치)
    input                        in_q411,    // 이 Tile 의 입력 포맷 : 1=Q4.11 코드, 0=bf16
                                             //   (첫 beat 에서 래치, in_shift 와 같은 규약)
    // ---- 출력 : 매 clk 한 열(LANE원소), D 회 ----
    output reg                   out_valid,
    output reg  [LANE*OW-1:0]    out_col,    // signed Q4.11 x LANE
    output reg                   out_last,
    // ---- 상태 ----
    output reg                   ovf         // 포화 발생 (sticky, rst_n 으로만 클리어)
);
    // ---- 유도 파라미터 -----------------------------------------------------
    localparam integer SXW   = IW + DLOG;              // = 31, sum X
    localparam integer SQW   = 2*(IW-1) + DLOG + 1;    // = 54, sum X*X
    localparam integer VW    = SQW + DLOG;             // = 61, (SQ<<DLOG) - SX^2
    localparam integer VF    = 2*IF + 2*DLOG;          // = 44, var = V * 2^-VF
    localparam integer MUF   = IF + DLOG;              // = 22, mu = SX * 2^-MUF
    localparam integer DDW   = SXW + 2;                // = 33, d = (X<<DLOG) - SX
    localparam integer DQW   = DDW + MAXSHL;           // = 38, d 를 미리 좌시프트
    localparam integer EWD   = 6;                      // e 폭 (signed)
    localparam integer QW    = 6;                      // leading-one 위치 폭
    localparam integer EMAXV = (VW-1-VF)/2;            // = 8,  e 최대값
    localparam integer MAXU  = EMAXV + (MUF-DSF) + MAXSHL;  // = 17, 우시프트 최대
    localparam integer USH   = 5;                      // u 폭 (0..MAXU)
    localparam integer YSH   = DSF + RF - OF;          // = 24, 최종 반올림 시프트
    localparam integer PDW   = DSW + RW + 1;           // = 44, ds*r 곱 폭
    localparam integer MW    = 5;                      // $clog2(LANE)
    localparam integer NBW   = 2;                      // $clog2(NB)
    localparam integer AW    = NBW + DLOG;             // = 9, xbuf 주소 폭
    localparam integer NB2   = 2;                      // sum_x2 / rstd_man / rstd_exp 슬롯 수

    // 포화 한계값 — 비교식이 전부 signed 가 되도록 폭까지 맞춰 상수로 둔다
    // (signed wire 와 unsigned concat 을 비교하면 Verilog 가 unsigned 비교로 바꾼다)
    localparam signed [DSW-1:0]  DS_MAX  = {1'b0, {(DSW-1){1'b1}}};   // +2^24-1 (대칭)
    localparam signed [DSW-1:0]  DS_MIN  = -DS_MAX;
    localparam signed [DQW-1:0]  DSQ_MAX = (1 <<< (DSW-1)) - 1;
    localparam signed [DQW-1:0]  DSQ_MIN = -DSQ_MAX;
    localparam signed [OW-1:0]   Y_MAX   = {1'b0, {(OW-1){1'b1}}};
    localparam signed [OW-1:0]   Y_MIN   = {1'b1, {(OW-1){1'b0}}};
    localparam signed [PDW-1:0]  YP_MAX  = (1 <<< (OW-1)) - 1;
    localparam signed [PDW-1:0]  YP_MIN  = -(1 <<< (OW-1));
    localparam        [VW-1:0]   EPSV    = EPS_INT;

    localparam [1:0] SL_FREE = 2'd0, SL_RECVD = 2'd1, SL_STATED = 2'd2;

    genvar  gl;
    integer i;

    // ======================================================================
    //  슬롯 상태 (단일 드라이버 — 각 FSM 은 done 펄스만 낸다)
    // ======================================================================
    reg  [1:0]     slot_state [0:NB-1];       // FREE -> RECVD -> STATED -> FREE
    reg  [NBW-1:0] slot_recv, slot_stat, slot_norm;

    // RXP 누산 파이프라인 (본체는 S1 절 참조) — slot_state 가 slp_p1 을 쓴다
    reg                   hs_p0,  hs_p1;
    reg                   fst_p0, fst_p1;
    reg                   lst_p0, lst_p1;
    reg  [NBW-1:0]        slp_p0, slp_p1;
    reg                   x2s_p0, x2s_p1;
    reg  signed [IW-1:0]  x_p    [0:LANE-1];
    reg  signed [SXW-1:0] xe_p   [0:LANE-1];
    reg  [SQW-1:0]        sq_p   [0:LANE-1];


    wire recv_done, stat_done, norm_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NB; i = i + 1) slot_state[i] <= SL_FREE;
        end else begin
            if (recv_done) slot_state[slp_p1]   <= SL_RECVD;    // S1 -> S2 (파이프라인이 든 슬롯)
            if (stat_done) slot_state[slot_stat] <= SL_STATED;   // S2 -> S3
            if (norm_done) slot_state[slot_norm] <= SL_FREE;     // S3 반납
        end
    end

    // ======================================================================
    //  S1 : 수신 + 행별 sum / sumsq   (뒤 단계와 독립적으로 굴러감)
    // ======================================================================
    reg  [DLOG-1:0]      recv_col;
    reg  signed [SXW-1:0] sum_x  [0:NB-1][0:LANE-1];   // sum X   (= mu),  S1 -> S2,**S3**
    reg  [SQW-1:0]        sum_x2  [0:NB2-1][0:LANE-1];  // sum X*X,          S1 -> S2 뿐
    reg  signed [XSW-1:0] xshift_slot   [0:NB-1];             // Tile 별 고정소수점 창
    reg                   q411_slot     [0:NB-1];             // Tile 별 입력 포맷 (S3 에서도 씀)
    // sum_x2 는 S2 에서 소비가 끝나므로 수명이 S1+S2 = 166 clk (< 2P) -> 슬롯 2개면 된다.
    // 슬롯 수가 다르므로 mod-2 전용 포인터를 따로 돌린다 (slot_recv/slot_stat 와 같은 시점에 토글).
    reg  x2_wslot, x2_rslot;

    wire recv_hs    = in_valid & in_ready;
    wire recv_first = (recv_col == {DLOG{1'b0}});
    wire recv_last  = (recv_col == D-1);

    assign in_ready  = (slot_state[slot_recv] == SL_FREE);
    // recv_done 은 **누산 파이프라인 끝**에서 낸다 (아래 RXP 단 참조).
    // 변환→제곱→누산을 한 사이클에 두면 100 MHz 에서 -3.1 ns 로 실패한다.

    // 첫 열에서는 입력값을, 이후에는 저장된 값을 쓴다
    wire signed [XSW-1:0] xshift_cur = recv_first ? in_shift : xshift_slot[slot_recv];
    wire                  q411_cur   = recv_first ? in_q411  : q411_slot[slot_recv];

    // ------------------------------------------------------------------
    //  xsh / q411 을 **레인 그룹별로 복제한 레지스터**로 넘긴다
    // ------------------------------------------------------------------
    //  `xshift_cur` 는 64개 변환기(32레인 x 2종)에 뿌려져 `recv_first` 팬아웃이
    //  163 까지 올라갔다.  187.5 MHz 에서 `recv_col -> recv_first -> mux ->
    //  변환기` 가 최악 경로였다 (5.600 ns, 그중 배선이 65 %).
    //
    //  `in_shift`(= EvT_Engine 의 `q_gsh`) 와 `in_q411`(= `q_flag2[0]`) 는
    //  **명령어 내내 고정**이고 `xshift_slot[]` 도 그 값을 담은 것이라, 먹스의 두
    //  갈래가 항상 같은 값이다.  따라서 한 사이클 늦춰도 값이 안 바뀐다
    //  (첫 beat 도 안전 : q_gsh 는 S_DEC 에서 정해져 S_GCONST 3 사이클 뒤에야
    //   ln_start 가 뜬다).
    //
    //  그룹마다 사본을 두면 배치기가 각 사본을 자기 8레인 근처에 놓아 배선이
    //  짧아진다.  DONT_TOUCH 가 없으면 합성이 다시 하나로 합쳐버린다.
    localparam integer NG  = 4;                 // 8 레인씩 4 그룹
    localparam integer GSZ = LANE / NG;
    (* DONT_TOUCH = "yes" *) reg signed [XSW-1:0] xsh_g [0:NG-1];
    (* DONT_TOUCH = "yes" *) reg                  q4_g  [0:NG-1];
    integer gi;
    always @(posedge clk) begin
        for (gi = 0; gi < NG; gi = gi + 1) begin
            xsh_g[gi] <= xshift_cur;
            q4_g[gi]  <= q411_cur;
        end
    end

    wire signed [IW-1:0]  x_cvt     [0:LANE-1];
    wire [LANE-1:0]       x_cvt_ovf;
    wire [SQW-1:0]        x_cvt_sq_p [0:LANE-1];   // x_p 기준 (1단 늦음)
    generate
        for (gl = 0; gl < LANE; gl = gl + 1) begin : g_rx
            localparam integer G = gl / GSZ;      // 이 레인이 속한 그룹
            // [타이밍] 입력 포맷 두 갈래를 **병렬**로 변환하고 마지막에 고릅니다.
            // Q4.11 을 bf16 으로 올려서 넣으면 정규화(LZC+좌시프트)와
            // 역정규화(우시프트)가 한 사이클에 직렬로 놓입니다. 각자
            // Q(IW-1-IF).IF 까지 가서 mux 로 합치면 깊이가 절반입니다.
            wire signed [IW-1:0] xc_bf, xc_q4;
            wire                 ov_bf, ov_q4;
            Bf16_To_Fix #(.IW(IW), .IF(IF), .XSW(XSW)) u_cvt (
                .bf(in_col[gl*16 +: 16]), .xsh(xsh_g[G]),
                .x(xc_bf), .ovf(ov_bf));
            Q411_To_Fix #(.IW(IW), .IF(IF), .QF(11), .XSW(XSW)) u_cvt_q (
                .code(in_col[gl*16 +: 16]), .xsh(xsh_g[G]),
                .x(xc_q4), .ovf(ov_q4));
            assign x_cvt[gl]     = q4_g[G] ? xc_q4 : xc_bf;
            assign x_cvt_ovf[gl] = q4_g[G] ? ov_q4 : ov_bf;
            // X*X : X 가 대칭포화(|X| <= 2^23-1)라 곱은 항상 2^46 미만 -> 정확
            //  **등록된 x_p 를 쓴다** — DSP48 의 A/B 입력 레지스터로 흡수되어
            //  곱셈이 파이프라인 한 단을 통째로 차지하게 된다.
            wire signed [2*IW-1:0] sq_full = x_p[gl] * x_p[gl];
            assign x_cvt_sq_p[gl] = {{(SQW-(2*IW-1)){1'b0}}, sq_full[2*IW-2:0]};
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            recv_col   <= {DLOG{1'b0}};
            slot_recv <= {NBW{1'b0}};
            x2_wslot   <= 1'b0;
            for (i = 0; i < NB; i = i + 1) xshift_slot[i] <= {XSW{1'b0}};
            for (i = 0; i < NB; i = i + 1) q411_slot[i]   <= 1'b0;
        end else if (recv_hs) begin
            if (recv_first) xshift_slot[slot_recv] <= in_shift;
            if (recv_first) q411_slot[slot_recv]   <= in_q411;
            recv_col <= recv_col + 1'b1;
            if (recv_last) begin
                recv_col   <= {DLOG{1'b0}};
                slot_recv <= (slot_recv == NB-1) ? {NBW{1'b0}} : slot_recv + 1'b1;
                x2_wslot   <= ~x2_wslot;                      // mod-2
            end
        end
    end

    // ------------------------------------------------------------------
    //  RXP : 수신 누산 파이프라인 (2단)
    // ------------------------------------------------------------------
    //  BRAM 출력 -> Bf16_To_Fix(33b 배럴 시프트) -> X*X(DSP) -> 47b 누산 을
    //  **한 사이클에 두면 논리 40단, 12.8 ns** 가 되어 100 MHz 를 못 맞춘다
    //  (ZU9EG -2 실측 WNS -3.083 ns).  그래서 두 곳을 끊는다 :
    //
    //     P0 : BRAM -> 변환            -> x_p       (여기서 자름)
    //     P1 : x_p  -> 제곱/부호확장   -> sq_p/xe_p (DSP 내부 레지스터로 흡수)
    //     P2 : 누산                    -> sum_x/sum_x2
    //
    //  **처리량은 그대로**다 (매 beat 1열).  지연만 2 clk 늘고, 그만큼
    //  recv_done 도 늦춰서 S2 가 확정된 합만 읽게 한다.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hs_p0 <= 1'b0; hs_p1 <= 1'b0;
        end else begin
            hs_p0 <= recv_hs;      hs_p1 <= hs_p0;
        end
    end
    always @(posedge clk) begin
        fst_p0 <= recv_first;  fst_p1 <= fst_p0;
        lst_p0 <= recv_last;   lst_p1 <= lst_p0;
        slp_p0 <= slot_recv;   slp_p1 <= slp_p0;
        x2s_p0 <= x2_wslot;    x2s_p1 <= x2s_p0;
        for (i = 0; i < LANE; i = i + 1) begin
            x_p[i]  <= x_cvt[i];
            xe_p[i] <= {{(SXW-IW){x_p[i][IW-1]}}, x_p[i]};
            sq_p[i] <= x_cvt_sq_p[i];
        end
    end

    // 누산기는 폭이 넓어 리셋을 걸지 않는다 (첫 beat 에서 무조건 로드된다)
    always @(posedge clk) begin
        if (hs_p1)
            for (i = 0; i < LANE; i = i + 1) begin
                sum_x[slp_p1][i]  <= fst_p1 ? xe_p[i] : sum_x[slp_p1][i]  + xe_p[i];
                sum_x2[x2s_p1][i] <= fst_p1 ? sq_p[i] : sum_x2[x2s_p1][i] + sq_p[i];
            end
    end

    assign recv_done = hs_p1 & lst_p1;

    // ==================================================================
    //  xbuf : simple dual-port SRAM   (write = S1, 등록형 read = S3)
    //    SRAM 은 0-latency 조합 읽기가 물리적으로 불가능하므로 읽기를
    //    레지스터 뒤로 뺀다.  SRAM_LAT 은 총 읽기 지연 단수다 (0 은 불가능).
    //      1 = 코어 출력 레지스터만,  2 = + 선택적 출력 레지스터 (고Fmax 용)
    // ==================================================================
    reg  [LANE*16-1:0] xbuf [0:NB*D-1];
    reg  [DLOG:0]      norm_col;
    reg                norm_state;
    localparam MS_IDLE = 1'b0, MS_RUN = 1'b1;

    wire        norm_issue = (norm_state == MS_RUN) && (norm_col < D);
    wire [AW-1:0] wr_addr = {slot_recv, recv_col};
    wire [AW-1:0] rd_addr = {slot_norm, (norm_issue ? norm_col[DLOG-1:0] : {DLOG{1'b0}})};

    reg [LANE*16-1:0] sram_rd_c;      // SRAM 코어 출력 레지스터 (= 읽기 1단째, 필수)
    reg               sram_vld_c, sram_last_c;
    always @(posedge clk) begin
        if (recv_hs) xbuf[wr_addr] <= in_col;
        sram_rd_c <= xbuf[rd_addr];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sram_vld_c <= 1'b0; sram_last_c <= 1'b0; end
        else begin
            sram_vld_c <= norm_issue;
            sram_last_c <= norm_issue && (norm_col == D-1);
        end
    end

    wire [LANE*16-1:0] sram_rd;
    wire               sram_vld, sram_last;
    generate
        if (SRAM_LAT >= 2) begin : g_oreg   // 선택적 출력 레지스터 -> 총 2단
            reg [LANE*16-1:0] srd_r;
            reg               srv_r, srl_r;
            always @(posedge clk) srd_r <= sram_rd_c;
            always @(posedge clk or negedge rst_n)
                if (!rst_n) begin srv_r <= 1'b0; srl_r <= 1'b0; end
                else        begin srv_r <= sram_vld_c;  srl_r <= sram_last_c;  end
            assign sram_rd = srd_r; assign sram_vld = srv_r; assign sram_last = srl_r;
        end else begin : g_oreg             // 코어 1단만
            assign sram_rd = sram_rd_c; assign sram_vld = sram_vld_c; assign sram_last = sram_last_c;
        end
    endgenerate

    // ======================================================================
    //  S2 : 행별 var -> 1/sqrt        (sum_x/sum_x2[slot_stat] -> rstd_man/rstd_exp[slot_stat])
    // ======================================================================
    localparam SS_IDLE = 1'b0, SS_RUN = 1'b1;

    reg              stat_state;
    reg  [MW:0]      stat_row_i, stat_row_c;                     // 행 투입 / 수집 포인터
    // rstd_man/rstd_exp 는 S2 -> S3 로 수명 S2+S3 = 170 clk (< 2P) -> 슬롯 2개.
    // 단 정상상태에서 여유가 3 clk 뿐이라, 파이프 단수를 건드리면 Tile 을 연속으로
    // 밀어 넣는 경우로 반드시 재검증할 것.
    reg  [RW-1:0]        rstd_man [0:NB2-1][0:LANE-1];  // 행별 rstd 가수
    reg  signed [EWD-1:0] rstd_exp [0:NB2-1][0:LANE-1]; // 행별 rstd 지수
    reg  rstd_wslot, rstd_rslot;

    wire stat_issue = (stat_state == SS_RUN) && (stat_row_i < LANE);
    wire signed [SXW-1:0] sum_x_rd = sum_x[slot_stat][stat_row_i[MW-1:0]];
    wire        [SQW-1:0] sum_x2_rd = sum_x2[x2_rslot][stat_row_i[MW-1:0]];

    // ---- 전단 2 stage : SX*SX (31x31, **행당 1개**) 와 뺄셈을 나눠 넣는다 ----
    reg             var1_vld, var2_vld;
    reg  [VW-1:0]   var1_sumsq_sh, var1_sumx_sq, var_v;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            var1_vld <= 1'b0; var2_vld <= 1'b0;
            var1_sumsq_sh <= {VW{1'b0}}; var1_sumx_sq <= {VW{1'b0}}; var_v <= {VW{1'b0}};
        end else begin
            var1_vld    <= stat_issue;
            var1_sumsq_sh  <= {sum_x2_rd, {DLOG{1'b0}}};        // SQ << DLOG
            var1_sumx_sq <= sum_x_rd * sum_x_rd;                // SX^2 (<= 2^60, 무손실)
            var2_vld    <= var1_vld;
            var_v  <= var1_sumsq_sh - var1_sumx_sq + EPSV;      // >= 0 보장 (Cauchy-Schwarz)
        end
    end

    wire                  rsq_vld;
    wire [RW-1:0]         rsq_man;
    wire signed [EWD-1:0] rsq_exp;
    Rsqrt_Unit #(.VW(VW), .VF(VF), .RW(RW), .RF(RF), .QW(QW), .EW(EWD)) u_rsqrt (
        .clk(clk), .rst_n(rst_n),
        .in_valid(var2_vld), .v(var_v),
        .out_valid(rsq_vld), .r(rsq_man), .e(rsq_exp)
    );

    assign stat_done = (stat_state == SS_RUN) && rsq_vld && (stat_row_c == LANE-1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stat_state <= SS_IDLE; slot_stat <= {NBW{1'b0}};
            x2_rslot <= 1'b0; rstd_wslot <= 1'b0;
            stat_row_i <= 0; stat_row_c <= 0;
        end else begin
            case (stat_state)
            SS_IDLE: if (slot_state[slot_stat] == SL_RECVD) begin
                stat_row_i <= 0; stat_row_c <= 0; stat_state <= SS_RUN;
            end
            SS_RUN: begin
                if (stat_row_i < LANE) stat_row_i <= stat_row_i + 1'b1;
                if (rsq_vld) begin
                    stat_row_c <= stat_row_c + 1'b1;
                    if (stat_row_c == LANE-1) begin
                        slot_stat <= (slot_stat == NB-1) ? {NBW{1'b0}} : slot_stat + 1'b1;
                        x2_rslot   <= ~x2_rslot;              // mod-2
                        rstd_wslot   <= ~rstd_wslot;              // mod-2
                        stat_state <= SS_IDLE;
                    end
                end
            end
            endcase
        end
    end

    always @(posedge clk) begin
        if (rsq_vld) begin
            rstd_man[rstd_wslot][stat_row_c[MW-1:0]] <= rsq_man;
            rstd_exp[rstd_wslot][stat_row_c[MW-1:0]] <= rsq_exp;
        end
    end

    // ======================================================================
    //  S3 : y = (x - mu) * rstd  ->  출력          (xbuf[slot_norm] -> out_col)
    // ------------------------------------------------------------------
    //   B1 : BF16 재변환 + d = (X<<DLOG) - SX
    //   B2 : ds = round(d * 2^-e)  (미리 <<MAXSHL 해두고 단방향 배럴 시프트)
    //   C  : y = round(ds * r >> YSH)  + 포화     -> out_col
    // ======================================================================
    reg                   norm_v1, norm_l1, norm_v2, norm_l2;
    reg  signed [DDW-1:0] diff_r  [0:LANE-1];
    reg  signed [DSW-1:0] dscaled_r [0:LANE-1];
    reg  [LANE-1:0]       sat_ds;
    wire [LANE*OW-1:0]    y_pack;
    wire [LANE-1:0]       xo_ovf, sat_y;

    generate
        for (gl = 0; gl < LANE; gl = gl + 1) begin : g_norm
            // ---- B1 : 재변환 + mu 뺄셈 ------------------------------------
            // xbuf 는 **원본 포맷 그대로** 담으므로 S3 도 같은 분기가 필요합니다.
            wire signed [IW-1:0]  xo_bf, xo_q4;
            wire                  ovb, ovq;
            Bf16_To_Fix #(.IW(IW), .IF(IF), .XSW(XSW)) u_cvt (
                .bf(sram_rd[gl*16 +: 16]), .xsh(xshift_slot[slot_norm]),
                .x(xo_bf), .ovf(ovb));
            Q411_To_Fix #(.IW(IW), .IF(IF), .QF(11), .XSW(XSW)) u_cvt_q (
                .code(sram_rd[gl*16 +: 16]), .xsh(xshift_slot[slot_norm]),
                .x(xo_q4), .ovf(ovq));
            wire signed [IW-1:0]  xo = q411_slot[slot_norm] ? xo_q4 : xo_bf;
            assign xo_ovf[gl] = q411_slot[slot_norm] ? ovq : ovb;
            // d = x - mu 를 Q(MUF) 로 맞춘다 : (X << DLOG) - SX
            wire signed [DDW-1:0] x_aligned = {{(DDW-IW-DLOG){xo[IW-1]}}, xo, {DLOG{1'b0}}};
            wire signed [DDW-1:0] mu_aligned  = {{(DDW-SXW){sum_x[slot_norm][gl][SXW-1]}},
                                             sum_x[slot_norm][gl]};
            wire signed [DDW-1:0] diff_c     = x_aligned - mu_aligned;

            // ---- B2 : e 시프트 (곱셈 앞으로 옮긴 배럴 시프트) ---------------
            wire signed [EWD+2:0] u_raw = {{3{rstd_exp[rstd_rslot][gl][EWD-1]}}, rstd_exp[rstd_rslot][gl]}
                                          + (MUF - DSF) + MAXSHL;
            wire [USH-1:0] u = (u_raw <= 0)     ? {USH{1'b0}} :
                               (u_raw >= MAXU)  ? MAXU[USH-1:0] : u_raw[USH-1:0];
            wire signed [DQW-1:0] diff_shl   = {diff_r[gl], {MAXSHL{1'b0}}};
            // 반올림 상수 2^(u-1) 은 (1<<u)>>1 로 만든다 (u=0 에서 시프트량 음수 회피)
            wire signed [DQW-1:0] half = ({{(DQW-1){1'b0}}, 1'b1} <<< u) >>> 1;
            wire signed [DQW-1:0] diff_rnd  = (diff_shl + half) >>> u;
            wire ds_hi = (diff_rnd > DSQ_MAX);
            wire ds_lo = (diff_rnd < DSQ_MIN);
            wire signed [DSW-1:0] dscaled_c = ds_hi ? DS_MAX : ds_lo ? DS_MIN : diff_rnd[DSW-1:0];

            // ---- C : ds * r  -> Q4.11 ------------------------------------
            //  ds(25b signed) x r(18b unsigned -> 19b signed) = DSP48 1개
            wire signed [PDW-1:0] prod = dscaled_r[gl] * $signed({1'b0, rstd_man[rstd_rslot][gl]});
            wire signed [PDW-1:0] y_round  = (prod + (1 <<< (YSH-1))) >>> YSH;
            wire y_hi = (y_round > YP_MAX);
            wire y_lo = (y_round < YP_MIN);
            assign sat_y[gl] = y_hi | y_lo;
            assign y_pack[gl*OW +: OW] = y_hi ? Y_MAX : y_lo ? Y_MIN : y_round[OW-1:0];

            always @(posedge clk) begin
                diff_r[gl]     <= diff_c;
                dscaled_r[gl]    <= dscaled_c;
                sat_ds[gl]  <= ds_hi | ds_lo;
            end
        end
    endgenerate

    assign norm_done = (norm_state == MS_RUN) && norm_v2 && norm_l2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            norm_state <= MS_IDLE; slot_norm <= {NBW{1'b0}};
            rstd_rslot <= 1'b0;
            norm_col <= 0;
            norm_v1 <= 1'b0; norm_l1 <= 1'b0; norm_v2 <= 1'b0; norm_l2 <= 1'b0;
            out_valid <= 1'b0; out_last <= 1'b0;
            out_col   <= {(LANE*OW){1'b0}};
        end else begin
            norm_v1 <= sram_vld;  norm_l1 <= sram_last;
            norm_v2 <= norm_v1;  norm_l2 <= norm_l1;
            out_valid <= norm_v2;
            out_last  <= norm_v2 & norm_l2;
            out_col   <= y_pack;

            case (norm_state)
            MS_IDLE: if (slot_state[slot_norm] == SL_STATED) begin
                norm_col <= 0; norm_state <= MS_RUN;
            end
            MS_RUN: begin
                if (norm_issue) norm_col <= norm_col + 1'b1;
                if (norm_v2 && norm_l2) begin                     // norm_done 펄스
                    slot_norm <= (slot_norm == NB-1) ? {NBW{1'b0}} : slot_norm + 1'b1;
                    rstd_rslot   <= ~rstd_rslot;                      // mod-2
                    norm_state <= MS_IDLE;
                end
            end
            endcase
        end
    end

    // ---- 포화 sticky 플래그 ------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) ovf <= 1'b0;
        else if ((recv_hs && (|x_cvt_ovf)) || (sram_vld && (|xo_ovf))
                 || (norm_v2 && (|sat_ds)) || (norm_v2 && (|sat_y)))
            ovf <= 1'b1;
    end

    // ---- 파라미터 정합성 체크 (합성 무관) ---------------------------------
    initial begin
        if (D != (1 << DLOG))
            $display("ERROR: LayerNorm_Unit D(%0d) must be 2^DLOG(%0d)", D, DLOG);
        if (NB < 3 || NB > (1 << NBW))
            $display("ERROR: LayerNorm_Unit NB(%0d) must be 3..%0d (버퍼 점유 > 2P)",
                     NB, (1 << NBW));
        if (RW != RF+1)
            $display("ERROR: LayerNorm_Unit RW(%0d) must be RF+1", RW);
        if (YSH < 1)
            $display("ERROR: LayerNorm_Unit DSF+RF must be > OF");
        if (VF % 2 != 0)
            $display("ERROR: LayerNorm_Unit VF(%0d) must be even", VF);
        if (LANE != (1 << MW))
            $display("ERROR: LayerNorm_Unit LANE(%0d) must be 2^%0d", LANE, MW);
    end
endmodule
