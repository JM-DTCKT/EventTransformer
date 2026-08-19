// -----------------------------------------------------------------------------
// EvT_Engine : EvT(DVS128_10) 실행기 — step 프로그램을 순서대로 돌립니다
//
// `fpga_nl/FFN_Engine` 과 역할은 같지만 규모가 다릅니다:
//
//                     FFN_Engine        EvT_Engine
//   step 수           12                **타임스텝당 150 + 끝 5**
//   재귀              없음              **타임스텝 T(≤20) 루프, latent 누적**
//   attention         없음              **3블록 x head 4**
//
// step 이 150개라 `fpga_nl` 처럼 레지스터 파일에 못 담습니다. **Step_Mem(BRAM)**
// 에 담고 하나씩 읽어 실행합니다. AXI-Stream 로더가 다른 메모리와 같은 방식으로
// 채웁니다.
//
// ## step 프로그램은 정적입니다
//
// `n_tok` 은 타임스텝마다 다릅니다(실측 16~123). 영역 크기를 `n_tok` 에 맞추면
// 베이스가 매번 바뀌어 프로그램을 5,182벌 만들어야 합니다. 그래서 **모든 영역
// stride 를 최악치(토큰 128)로 고정**했습니다(`sw/schedule_evt.py`). 그러면
// 베이스가 전부 상수가 되고, `n_tok` 에 따라 바뀌는 것은 네 필드뿐입니다:
//
//   VAR[0]  M    <- n_tok        (토큰 행을 도는 step)
//   VAR[1]  NOUT <- n_tok+1      (Q·Kᵀ 의 Nout = Lk)
//   VAR[2]  K    <- n_tok+1      (attn·V 의 reduce = Lk)
//   VAR[3]  C    <- n_tok+1      (softmax 의 클래스 수 = Lk)
//
// 발행 시점에 `n_tok` 레지스터로 채워 넣습니다. latent 쪽 step 은 `Lk = 96+1` 이
// 상수라 VAR 을 안 씁니다.
//
// ## step 워드 (256비트 = Step_Mem 한 워드)
//
//   [ 31: 0] KIND[3:0] CONS[5:4] ACT[7:6] VAR[11:8] FLAG[15:12]
//            SHIFT[21:16] GSH[27:22] FLAG2[31:28]
//   [ 63:32] M      [ 95:64] K      [127:96] NOUT
//   [159:128] AIN   [191:160] BIN   [223:192] AOUT
//   [255:224] PB[15:0] | OSTR[31:16]
//
//   OSTR = GEMM 출력 stride. 보통 NOUT 과 같지만 **다를 수 있습니다** —
//   `event_projection` 은 96채널을 내면서 `preproc` 입력(160 = 96 + pos enc 64)의
//   앞쪽에 써야 하므로 stride 가 160 입니다. FLAG[1](head-major)일 때는 head 간
//   간격으로 쓰입니다.
//
//   GELU 뒤 int8 재양자화 곱수는 **PB + NOUT** 자리에 둡니다 (그 레이어의 채널별
//   항목 바로 뒤). 그래서 별도 필드가 필요 없습니다.
//
//   KIND  0 GEMM  1 LN  2 SMAX  3 RES  4 MEAN  5 ARGMAX
//   CONS  0 int8(+ACT)  1 Q4.11→GELU→int8  2 bf16  3 Q6.9(softmax 직결)
//   FLAG (s_rd[15:12] → q_flag[3:0])
//     [0] GEMM 출력을 **Transpose32 경유**로            (in_proj 의 V)
//     [1] GEMM 출력을 **head-major 주소**로             (in_proj 의 Q/K)
//         이때 OSTR 은 head 간 간격으로 쓰입니다
//     [2] B 피연산자를 **A_Mem** 에서                    (Q·Kᵀ, attn·V)
//     [3] 예약
//
// ## A_Mem 포트 중재
//
// 읽기 1포트를 GEMM(A) · GEMM(B) · LN · RES · MEAN 이 나눠 씁니다. 동시에 도는
// 유닛이 없도록 스케줄이 보장하지만, **GEMM 만은 A 와 B 를 동시에** 읽습니다.
// B 가 W_Mem 인 경우(Linear)는 충돌이 없고, B 가 A_Mem 인 경우(Q·Kᵀ, attn·V)만
// 두 포트가 필요합니다. 그래서 A_Mem 을 **읽기 2포트**로 둡니다 (Bram_Sdp 두 벌에
// 같은 내용을 미러링 — BRAM 이 남으므로 이게 가장 단순합니다).
//
// ## 검증 상태
//
// **아직 통합 TB 를 안 돌렸습니다.** `fpga_nl` 의 12 step 엔진에서도 위상·주소
// 버그가 5개 나왔고, 여기는 step 이 150개입니다. `tb_evt` 로 tap 대조를 하기 전에는
// 동작한다고 볼 수 없습니다.
// -----------------------------------------------------------------------------
module EvT_Engine #(
    parameter N      = 32,
    parameter ACT_W  = 8,
    parameter PSUM_W = 32,
    parameter DIM_W  = 16,
    parameter E      = 128,
    parameter LAT    = 96,
    parameter HEADS  = 4,
    parameter HD     = 32,
    parameter TOKMAX = 128,
    parameter AW_W   = 14,       // W_Mem 워드 주소폭 (14,000 워드)
    parameter AW_A   = 14,       // A_Mem  (8,261 워드)
    parameter AW_PB  = 10,       // PB_Mem (860 워드)
    parameter AW_PG  = 8,        // PG_Mem (208 워드)
    parameter AW_S   = 8,        // Step_Mem (155 워드)
    parameter GELU_LUT_FILE  = "gelu.hex",
    parameter EXP_LUT_FILE   = "exp.hex",
    parameter RCP_LUT_FILE   = "recip.hex",
    parameter RSQRT_LUT_FILE = "rsqrt.hex"
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 start,
    output reg                  done,
    output wire                 busy,
    output wire [3:0]           dbg_state,
    output wire [AW_S-1:0]      dbg_step,

    // ---- 실행 파라미터 ----
    input  wire [AW_S-1:0]      n_body,        // 타임스텝당 step 수
    input  wire [AW_S-1:0]      n_tail,        // 끝에 한 번 도는 step 수
    input  wire [5:0]           n_time,        // 타임스텝 T
    input  wire [31:0]          eps,           // LayerNorm eps (fp32 비트)

    // ---- 타임스텝 입력 요청 ----
    // X/PIN 영역은 **한 타임스텝분**만 들어갑니다 (20 타임스텝을 다 담으면
    // 24k 워드로 A_Mem 을 넘습니다). 그래서 타임스텝마다 호스트가 새로 채워야
    // 하고, 엔진은 채워질 때까지 기다립니다.
    //
    //   tok_req = 1 로 멈춤  →  호스트가 tok_rd_idx 번째 타임스텝의
    //   X / PIN(pos enc) 를 적재하고 tok_ack 를 한 번 올림  →  진행
    //
    // `tok_rd_n` 은 그 타임스텝의 토큰 수입니다 (ack 시점에 래치).
    output wire [5:0]           tok_rd_idx,
    input  wire [DIM_W-1:0]     tok_rd_n,
    output reg                  tok_req,
    input  wire                 tok_ack,

    // ---- 메모리 적재 ----
    input  wire                 ld_we,
    input  wire [2:0]           ld_sel,        // 0 W  1 A  2 PB  3 PG  4 STEP  5 POS
    input  wire [AW_W-1:0]      ld_addr,
    input  wire [N*16-1:0]      ld_data,

    // ---- 결과 ----
    output reg  [3:0]           res_class,     // argmax
    output reg  [N*PSUM_W-1:0]  res_logits,    // 클래스별 acc (디버그)

    // ---- A_Mem 리드백 (IDLE 일 때만) ----
    input  wire                 dbg_rd_en,
    input  wire [AW_A-1:0]      dbg_rd_addr,
    output wire [N*16-1:0]      dbg_rd_data
);
    localparam K_GEMM=4'd0, K_LN=4'd1, K_SMAX=4'd2, K_RES=4'd3,
               K_MEAN=4'd4, K_ARGMAX=4'd5, K_POS=4'd6;
    localparam C_INT8=2'd0, C_Q411=2'd1, C_BF16=2'd2, C_Q69=2'd3;

    // 인스턴스보다 먼저 선언해야 하는 신호들
    // (사용이 선언보다 앞서면 Verilog 가 1비트 net 으로 암묵 선언합니다)
    reg                 col_v_d,  col_v_d2;
    reg [N*PSUM_W-1:0]  col_d_d,  col_d_d2;
    reg [DIM_W-1:0]     col_n_d,  col_mt_d;
    reg [DIM_W-1:0]     col_n_d2, col_mt_d2;
    reg                 sm_iv;
    reg [N*16-1:0]      sm_id;

    // =========================================================================
    // FSM
    // =========================================================================
    localparam S_IDLE=4'd0, S_FETCH=4'd1, S_DEC=4'd2, S_GCONST=4'd3,
               S_RUN=4'd4, S_WAIT=4'd5, S_NEXT=4'd6, S_TSTEP=4'd7, S_DONE=4'd8,
               S_TLOAD=4'd9;
    reg [3:0]        st;
    reg [AW_S-1:0]   sp;              // step 포인터
    reg [5:0]        ti;              // 타임스텝 인덱스
    reg              in_tail;
    reg [DIM_W-1:0]  n_tok;           // 이번 타임스텝의 토큰 수

    assign busy      = (st != S_IDLE);
    assign dbg_state = st;
    assign dbg_step  = sp;
    assign tok_rd_idx = ti;

    // =========================================================================
    // Step_Mem — step 하나 = 256비트 워드 하나
    // =========================================================================
    wire [N*8-1:0] s_rd;
    reg  [AW_S-1:0] s_addr;
    Bram_Sdp #(.DW(N*8), .AW(AW_S)) u_smem (
        .clk(clk), .we_en(ld_we && ld_sel == 3'd4), .we_addr(ld_addr[AW_S-1:0]),
        .we_be({N{1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(1'b1), .rd_addr(s_addr), .rd_data(s_rd));

    // 디코드 (S_DEC 에서 래치)
    reg [3:0]        q_kind;
    reg [1:0]        q_cons, q_act;
    reg [3:0]        q_var, q_flag;
    (* max_fanout = 16 *)
    reg [5:0]        q_sh, q_gsh;   // requant 시프트 — 위와 같은 이유로 복제
    reg [DIM_W-1:0]  q_M, q_K, q_NOUT;
    reg [AW_A-1:0]   q_AIN, q_AOUT;
    reg [AW_W-1:0]   q_BIN;
    reg [15:0]       q_PB, q_OSTR;
    reg [3:0]        q_flag2;        // 워드0 [31:28]
    // LayerNorm 은 **한 번에 32행(레인)** 만 처리합니다 (특징 축 리덕션이라
    // 레인이 곧 행). M 이 32를 넘으면 행타일 수만큼 다시 겁니다. step 을 타일마다
    // 쪼개면 프로그램이 n_tok 에 의존하게 되므로(최악 4벌) 엔진이 셉니다.
    // 하위 코어들의 `done` 은 **다음 start 까지 계속 1** 입니다. start 를 준
    // 사이클에 done 을 그대로 보면 **직전 완료를 이번 완료로 착각**합니다
    // (LayerNorm 행타일 반복에서 두 번째 타일이 통째로 건너뛰어져 드러났습니다).
    // done 이 한 번 0 으로 내려간 것을 본 뒤부터 인정합니다.
    reg              wait_ack;
    reg [5:0]        rs_mt;          // 현재 행타일 (RES)
    // RES 는 A_Mem 이 `base + mt*K + k` 라 행타일마다 다시 돕니다.
    // (레인이 행이므로 한 바퀴가 32행. M=52 면 두 바퀴, 96 이면 세 바퀴)
    wire [5:0]       rs_mt_last = (q_M > 0) ? ((q_M - 1'b1) >> 5) : 6'd0;
    wire [5:0]       ln_mt_last = rs_mt_last;      // MEAN 이 같이 씁니다
    reg [1:0]        gc;

    wire [DIM_W-1:0] Lk = n_tok + 1'b1;

    // =========================================================================
    // 메모리
    // =========================================================================
    wire            w_rd_en;
    wire [AW_W-1:0] w_rd_addr;
    wire [N*8-1:0]  w_rd_data;
    Bram_Sdp #(.DW(N*8), .AW(AW_W)) u_wmem (
        .clk(clk), .we_en(ld_we && ld_sel == 3'd0), .we_addr(ld_addr),
        .we_be({N{1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(w_rd_en), .rd_addr(w_rd_addr), .rd_data(w_rd_data));

    // A_Mem : 읽기 2포트가 필요합니다 (GEMM 이 A 와 B 를 동시에 읽는 경우).
    // 같은 내용을 두 벌에 미러링합니다 — BRAM 이 남고 제어가 가장 단순합니다.
    reg              a_we_en;
    reg  [AW_A-1:0]  a_we_addr;
    reg  [N*16-1:0]  a_we_data;
    reg              ar_en, br_en;
    reg  [AW_A-1:0]  ar_addr, br_addr;
    wire [N*16-1:0]  ar_data, br_data;

    wire a_we_ld = ld_we && ld_sel == 3'd1;
    wire            aw_en   = a_we_ld ? 1'b1              : a_we_en;
    wire [AW_A-1:0] aw_addr = a_we_ld ? ld_addr[AW_A-1:0] : a_we_addr;
    wire [N*16-1:0] aw_data = a_we_ld ? ld_data           : a_we_data;

    Bram_Sdp #(.DW(N*16), .AW(AW_A)) u_amem0 (
        .clk(clk), .we_en(aw_en), .we_addr(aw_addr), .we_be({(N*2){1'b1}}),
        .we_data(aw_data), .rd_en(ar_en), .rd_addr(ar_addr), .rd_data(ar_data));
    Bram_Sdp #(.DW(N*16), .AW(AW_A)) u_amem1 (
        .clk(clk), .we_en(aw_en), .we_addr(aw_addr), .we_be({(N*2){1'b1}}),
        .we_data(aw_data), .rd_en(br_en), .rd_addr(br_addr), .rd_data(br_data));

    assign dbg_rd_data = ar_data;

    // PB_Mem : 채널 c 의 {mult|step, bias}
    wire [AW_PB-1:0] pb_word;
    wire [N*8-1:0]   pb_rd;
    reg  [AW_PB+1:0] pb_idx;
    assign pb_word = pb_idx[AW_PB+1:2];
    Bram_Sdp #(.DW(N*8), .AW(AW_PB)) u_pbmem (
        .clk(clk), .we_en(ld_we && ld_sel == 3'd2), .we_addr(ld_addr[AW_PB-1:0]),
        .we_be({N{1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(1'b1), .rd_addr(pb_word), .rd_data(pb_rd));
    reg [1:0] pb_sub_d;
    always @(posedge clk) pb_sub_d <= pb_idx[1:0];
    wire signed [31:0] pb_mult = pb_rd[{pb_sub_d, 6'd0}      +: 32];
    wire signed [31:0] pb_bias = pb_rd[{pb_sub_d, 6'd0} + 32 +: 32];

    // ---- 한 단 더 잡아 둡니다 (타이밍) ----
    // 임플 결과 크리티컬 패스가 **PB_Mem BRAM → 4:1 서브워드 먹스 → 누산기
    // 덧셈 → Requant_Bf16 → 레지스터** 한 덩어리로 WNS +0.001 ns 였습니다.
    // 상수를 레지스터로 받아 BRAM+먹스와 재양자화를 갈라 놓습니다.
    // 컬럼도 한 단 더 늦춰 정렬을 맞추므로 사이클 비용은 타일당 1 입니다.
    // **팬아웃을 잘라 둡니다.** 이 상수 하나가 Col_Post 32레인 x 2개 = DSP 64개,
    // LayerNorm_Ev 의 requant, 그리고 ARGMAX 의 64비트 곱을 동시에 먹습니다.
    // 복제하지 않으면 칩 전역으로 뻗은 배선이 u_cp 경로를 -0.811 ns 로 끌어내립니다
    // (ZU9EG -2 실측). 상수라 복제 비용은 FF 몇 개뿐입니다.
    (* max_fanout = 16 *)
    reg signed [31:0] pb_mult_q, pb_bias_q;
    always @(posedge clk) begin
        pb_mult_q <= pb_mult;
        pb_bias_q <= pb_bias;
    end

    // PG_Mem : 특징 k 의 {gamma, beta}
    wire [AW_PG-1:0] pg_word;
    wire [N*8-1:0]   pg_rd;
    wire [AW_PG+2:0] pg_idx;
    assign pg_word = pg_idx[AW_PG+2:3];
    Bram_Sdp #(.DW(N*8), .AW(AW_PG)) u_pgmem (
        .clk(clk), .we_en(ld_we && ld_sel == 3'd3), .we_addr(ld_addr[AW_PG-1:0]),
        .we_be({N{1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(1'b1), .rd_addr(pg_word), .rd_data(pg_rd));
    reg [2:0] pg_sub_d;
    always @(posedge clk) pg_sub_d <= pg_idx[2:0];
    wire signed [15:0] pg_gamma = pg_rd[{pg_sub_d, 5'd0}      +: 16];
    wire signed [15:0] pg_beta  = pg_rd[{pg_sub_d, 5'd0} + 16 +: 16];

    // =========================================================================
    // GEMM
    // =========================================================================
    reg                  gm_start;
    wire                 gm_done, col_v, col_first, col_last;
    wire [N*PSUM_W-1:0]  col_d;
    wire [DIM_W-1:0]     col_n, col_mt;
    wire [N-1:0]         col_re;
    wire                 ga_rd_en, gb_rd_en;
    wire [AW_A-1:0]      ga_rd_addr;
    wire [AW_W-1:0]      gb_rd_addr;

    // B 피연산자 출처 : Linear 은 W_Mem, attention 은 A_Mem
    // FLAG[2] = B 를 A_Mem 에서 읽음. `q_flag` 는 4비트이므로 절대 비트번호
    // (s_rd[14])가 아니라 **상대 비트 [2]** 로 씁니다.
    wire b_from_a = (q_kind == K_GEMM) && q_flag[2];
    wire [N*8-1:0] gb_data_w = w_rd_data;
    wire [N*8-1:0] gb_data_a;
    genvar bn;
    generate
        for (bn = 0; bn < N; bn = bn + 1) begin : B_NARROW
            assign gb_data_a[bn*8 +: 8] = br_data[bn*16 +: 8];
        end
    endgenerate
    // ---- bias_k / bias_v 토큰 (FLAG2[2]) ----
    // attention 은 키가 `Lk = n_tok + 1` 개이고 마지막 하나가 학습된 상수
    // 토큰입니다. 위치(키 인덱스 n_tok)가 타임스텝마다 달라 K/V 에 써 넣으려면
    // 읽고-고쳐-쓰기가 필요합니다. 대신 **읽는 쪽에서 끼워 넣습니다** — 상수는
    // 각 head 영역 끝의 예약 칸에 호스트가 한 번 넣어 둡니다.
    //
    //   QK : B 워드의 레인 = 키. 키 n_tok 인 **레인 하나**만 바꿉니다.
    //        bias_k 워드는 레인 = head_dim 이라 step 시작에 한 번 읽어 둡니다.
    //        주소는 `AOUT` — QK 는 메모리에 안 써서 그 칸이 놉니다.
    //   AV : B 워드 자체가 키 하나(레인 = head_dim). 키 n_tok 이면 **주소만**
    //        `OSTR` 이 가리키는 칸으로 돌립니다 (AV 는 행타일이 하나라 놉니다).
    //
    // K/V 영역은 블록 3개가 돌려 쓰므로 예약 칸을 그 안에 둘 수 없습니다 —
    // 블록마다 bias 값이 다릅니다. 그래서 별도 영역(BKV)에 두고 주소를 step 이
    // 실어 옵니다.

    reg  [N*16-1:0] bk_word;                 // bias_k (레인 = head_dim)
    wire            has_bkv  = q_flag2[2];
    wire            is_qk    = has_bkv && (q_cons == C_Q69);
    wire            is_av    = has_bkv && (q_cons != C_Q69);
    wire [AW_A-1:0] b_off    = gb_rd_addr[AW_A-1:0] - q_BIN[AW_A-1:0];
    // bias 토큰은 **마지막 키** 입니다. cross 는 Lk = n_tok+1, latent 은 96+1 로
    // 고정이라 `n_tok` 이 아니라 **그 step 의 Lk** 로 잡아야 합니다.
    //   QK : 출력 열이 키라 Lk = q_NOUT      AV : reduce 가 키라 Lk = q_K
    wire [DIM_W-1:0] qk_key = q_NOUT - 1'b1;
    wire [DIM_W-1:0] av_key = q_K    - 1'b1;
    wire            qk_hit   = is_qk && (b_off[AW_A-1:5] == qk_key[AW_A-1:5]);
    wire [4:0]      qk_lane  = qk_key[4:0];
    wire [4:0]      qk_k     = b_off[4:0];   // reduce 인덱스 = head_dim
    wire            av_hit   = is_av && (b_off == av_key[AW_A-1:0]);

    // QK 의 B 워드는 **레인 = 키** 입니다. 마지막 키 타일에서 `Lk` 를 넘는 레인은
    // `in_proj.K` 가 쓴 적이 없습니다 — latent 블록은 키 96개가 타일 3개를 딱
    // 채워서 bias 키(96) 가 **아무도 안 쓴 4번째 타일**을 엽니다. 그대로 두면 그
    // 레인들이 X 로 올라오고, softmax 가 전 키를 합산하므로 **출력 전체가 X** 가
    // 됩니다 (cross 는 n_tok=52 라 우연히 안 걸렸습니다).
    wire [DIM_W-1:0] b_tile = {{(DIM_W-(AW_A-5)){1'b0}}, b_off[AW_A-1:5]};
    wire [DIM_W-1:0] b_lim  = q_NOUT - (b_tile << 5);

    // **A_Mem 은 주소를 준 다음 사이클에 답합니다.** 그래서 데이터에 거는 조작
    // (bias 레인 치환, 남는 레인 0)은 주소 조건을 **한 단 늦춰** 써야 짝이 맞습니다.
    // 안 늦추면 한 칸 앞 워드의 조건으로 바꿔치기합니다.
    reg qk_hit_d, is_qk_d;
    reg [4:0] qk_lane_d, qk_k_d;
    reg [DIM_W-1:0] b_lim_d;
    always @(posedge clk) begin
        qk_hit_d  <= qk_hit;  is_qk_d <= is_qk;
        qk_lane_d <= qk_lane; qk_k_d  <= qk_k;
        b_lim_d   <= b_lim;
    end

    wire [N*8-1:0] gb_data_bk;
    genvar bk;
    generate
        for (bk = 0; bk < N; bk = bk + 1) begin : B_BIAS
            assign gb_data_bk[bk*8 +: 8] =
                   (qk_hit_d && bk[4:0] == qk_lane_d) ? bk_word[{qk_k_d, 4'd0} +: 8]
                 : (is_qk_d && bk >= b_lim_d)         ? 8'd0
                                                      : gb_data_a[bk*8 +: 8];
        end
    endgenerate
    wire [N*8-1:0] gb_data = b_from_a ? gb_data_bk : gb_data_w;

    // 코어는 B 주소를 하나만 냅니다. W_Mem 이 답할 때 그 주소를 그대로 씁니다.
    // (선언만 하고 연결을 빠뜨리면 Verilog 는 조용히 X 를 읽습니다 — 컴파일도
    //  통과하고 시뮬도 안 죽습니다. 통합 TB 에서 `b_rd=.../xx` 로 드러났습니다.)
    assign w_rd_en   = gb_rd_en && !b_from_a;
    assign w_rd_addr = gb_rd_addr;

    Gemm_Core_Ev #(.N(N), .ACT_W(ACT_W), .PSUM_W(PSUM_W), .DIM_W(DIM_W),
                   .AW_A(AW_A), .AW_B(AW_W)) u_gemm (
        .clk(clk), .rst(rst), .start(gm_start), .all_done(gm_done),
        .M(q_M), .K(q_K), .Nout(q_NOUT), .a_base(q_AIN), .b_base(q_BIN),
        .a_rd_en(ga_rd_en), .a_rd_addr(ga_rd_addr), .a_rd_data(ar_data),
        .b_rd_en(gb_rd_en), .b_rd_addr(gb_rd_addr), .b_rd_data(gb_data),
        .col_valid(col_v), .col_data(col_d), .col_n(col_n), .col_mt(col_mt),
        .col_first(col_first), .col_last(col_last), .col_row_en(col_re));

    // GELU 뒤 int8 재양자화 곱수 — step 시작 시 한 번 읽어 둡니다
    (* max_fanout = 16 *)
    reg signed [31:0] g_mult_q;

    wire                cp_v;
    wire [N*16-1:0]     cp_d, cp_q69;
    Col_Post_Ev #(.N(N), .ACT_W(ACT_W), .PSUM_W(PSUM_W),
                  .GELU_LUT_FILE(GELU_LUT_FILE)) u_cp (
        .clk(clk), .rst(rst), .consumer(q_cons),
        .bias(pb_bias_q), .mult(pb_mult_q), .shift(q_sh),
        .g_mult(g_mult_q), .g_shift(q_gsh),
        .act_sel(q_act), .act_parm(8'd0), .raw16(q_flag[3]), .req2(q_flag2[3]),
        .in_valid(col_v_d2), .acc(col_d_d2),
        .out_valid(cp_v), .out_data(cp_d), .out_q69(cp_q69));

    // PB 경로 지연(BRAM 1 + 레지스터 1)만큼 컬럼을 늦춰 상수와 정렬
    always @(posedge clk) begin
        col_v_d  <= col_v;   col_v_d2  <= col_v_d;
        col_d_d  <= col_d;   col_d_d2  <= col_d_d;
        col_n_d  <= col_n;   col_n_d2  <= col_n_d;
        col_mt_d <= col_mt;  col_mt_d2 <= col_mt_d;
    end

    // Col_Post 지연만큼 쓰기 주소를 늦춥니다. **소비자마다 다릅니다** —
    //   Requant_Bf16 1단 / Requant_Int **3단** / Q4.11 은 **9단**
    //   (Q4.11 = requant 3 + gelu_pwl 3 + requant 3)
    //   Requant_Int 는 원래 2단이었지만 100 MHz 를 맞추려 프리애더를 끊어
    //   3단이 됐습니다 (`Requant_Int.v` 머리말 참조). 여기 숫자를 같이 안 고치면
    //   쓰기 주소가 데이터보다 빨라 **한 컬럼씩 밀려 저장**됩니다.
    // (`fpga_nl` 에서 여기를 뭉뚱그려 첫 워드가 안 써지는 버그가 있었습니다)
    localparam CP_LAT_B = 1, CP_LAT_S = 3, CP_LAT_G = 9, CP_LAT_R = 6;
    localparam CP_LAT_MAX = CP_LAT_G;
    reg [DIM_W-1:0] nq [0:CP_LAT_MAX-1];
    reg [DIM_W-1:0] mq [0:CP_LAT_MAX-1];
    integer zn;
    always @(posedge clk) begin
        nq[0] <= col_n_d2;  mq[0] <= col_mt_d2;
        for (zn = 1; zn < CP_LAT_MAX; zn = zn + 1) begin
            nq[zn] <= nq[zn-1];  mq[zn] <= mq[zn-1];
        end
    end
    wire [DIM_W-1:0] cp_n = (q_cons == C_Q411) ? nq[CP_LAT_G-1]
                          : (q_cons == C_BF16) ? nq[CP_LAT_B-1]
                          : q_flag2[3]         ? nq[CP_LAT_R-1]
                          :                      nq[CP_LAT_S-1];
    wire [DIM_W-1:0] cp_mt = (q_cons == C_Q411) ? mq[CP_LAT_G-1]
                           : (q_cons == C_BF16) ? mq[CP_LAT_B-1]
                           : q_flag2[3]         ? mq[CP_LAT_R-1]
                           :                      mq[CP_LAT_S-1];

    // Col_Post 파이프라인이 **비었는지** 봅니다. `gm_done` 은 시스톨릭 배열이
    // 다 쏟은 시점이라, 뒤에 붙은 requant/GELU 단(최대 9)이 아직 값을 들고 있을
    // 수 있습니다. 그 상태로 step 을 넘기면 마지막 컬럼들이 **다음 step 의
    // AOUT/OSTR** 로 나갑니다 — 전치 드레인에서 이미 한 번 겪은 실패입니다.
    reg [CP_LAT_MAX-1:0] cp_pipe;
    always @(posedge clk) begin
        if (rst) cp_pipe <= {CP_LAT_MAX{1'b0}};
        else     cp_pipe <= {cp_pipe[CP_LAT_MAX-2:0], col_v_d2};
    end
    wire cp_busy = col_v_d2 | (|cp_pipe);

    // =========================================================================
    // V 전치 : in_proj 의 V 컬럼을 모아 축을 돌립니다
    // =========================================================================
    wire [N*8-1:0] tr_in, tr_out;
    // `tr_in` 이 조합(cp_d) 이므로 쓰기 인에이블/주소도 **조합**이어야 합니다.
    // 레지스터로 한 단 늦추면 데이터만 한 칸 밀려 전치가 통째로 어긋납니다.
    wire [4:0]     tr_widx;
    wire           tr_we;
    wire [4:0]     tr_ridx;
    genvar tn;
    generate
        for (tn = 0; tn < N; tn = tn + 1) begin : TR_NARROW
            assign tr_in[tn*8 +: 8] = cp_d[tn*16 +: 8];
        end
    endgenerate
    Transpose32 #(.N(N), .W(8)) u_tr (
        .clk(clk), .rst(rst), .we(tr_we), .w_idx(tr_widx), .w_data(tr_in),
        .r_idx(tr_ridx), .r_data(tr_out));

    // ---- 채우고 → 쏟기 ----
    // `attn·V` 의 reduce 축은 **키**입니다. 그런데 in_proj 이 내는 컬럼은
    // (레인 = 토큰, 워드 = head_dim) 이라 축이 반대입니다. head 하나의 32컬럼
    // (d = 0..31)을 다 받으면 32x32 블록이 차고, 그걸 행(키)별로 32번 읽어
    // `V[h*OSTR + mt*32 + key]` 에 씁니다 — 레인이 head_dim 이 됩니다.
    //
    // 코어가 타일 하나(32컬럼)를 내고 다음 타일을 K 사이클 계산하는 동안이
    // **쏟을 틈**입니다 (K >= 32 이라 항상 충분). 그래서 멈춤이 없습니다.
    reg        tr_run, tr_arm, tr_go;
    reg [5:0]  tr_r;
    reg [1:0]  tr_h;
    reg [DIM_W-1:0] tr_mt;
    wire       tr_fill  = (q_kind == K_GEMM) && q_flag[0] && cp_v;
    wire       tr_last   = tr_fill && (cp_n[4:0] == 5'd31);
    wire [1:0] cp_n_h    = cp_n[6:5];        // head = 컬럼 / 32
    assign     tr_ridx   = tr_r[4:0];
    assign     tr_we     = tr_fill;
    assign     tr_widx   = cp_n[4:0];
    always @(posedge clk) begin
        if (rst) begin
            tr_run <= 1'b0; tr_arm <= 1'b0; tr_go <= 1'b0; tr_r <= 0;
        end else begin
            // 마지막 컬럼(d=31)의 쓰기가 **반영된 뒤** 읽어야 하므로 두 단 늦춥니다
            tr_arm  <= tr_last;
            tr_go   <= tr_arm;
            // head/타일은 **그 컬럼이 나온 순간** 잡아야 합니다 (다음 사이클엔
            // cp_n 이 이미 다음 타일 값입니다)
            if (tr_last) begin tr_h <= cp_n_h; tr_mt <= cp_mt; end
            if (tr_go) begin tr_run <= 1'b1; tr_r <= 0; end
            else if (tr_run) begin
                if (tr_r == 6'd31) tr_run <= 1'b0;
                tr_r <= tr_r + 1'b1;
            end
        end
    end

    // =========================================================================
    // positional encoding — 온칩 표에서 모아 PIN 뒤쪽에 씁니다
    //
    // 호스트가 미리 펴서 보내던 96.7 MB 를 없앱니다. 표(27.6 KB)는 여기 BRAM 에
    // 있고 타임스텝마다 오는 것은 `pos_idx`(최대 246 B) 뿐입니다. 자세한 것은
    // `rtl/Pos_Gather.v` 머리말.
    // =========================================================================
    reg              pos_start;
    wire             pos_done, pos_rd_en, pos_we_en;
    wire [AW_A-1:0]  pos_rd_addr, pos_we_addr;
    wire [N*16-1:0]  pos_we_data;

    Pos_Gather #(.N(N), .FEAT(64), .AW_A(AW_A), .AW_T(9), .DIM_W(DIM_W)) u_pos (
        .clk(clk), .rst(rst), .start(pos_start), .done(pos_done),
        .n_tok(q_M), .a_base(q_AOUT), .ostr(q_OSTR), .idx_base(q_AIN),
        .ld_we(ld_we && ld_sel == 3'd5), .ld_addr(ld_addr[8:0]),
        .ld_data(ld_data[64*8-1:0]),
        .rd_en(pos_rd_en), .rd_addr(pos_rd_addr), .rd_data(ar_data),
        .we_en(pos_we_en), .we_addr(pos_we_addr), .we_data(pos_we_data));

    // =========================================================================
    // LayerNorm
    // =========================================================================
    reg              ln_start;
    wire             ln_done, ln_ov;
    wire [DIM_W-1:0] ln_k, ln_paddr;
    wire [5:0]       ln_mt_o;
    wire [N*8-1:0]   ln_out;
    wire             la_rd_en;
    wire [AW_A-1:0]  la_rd_addr;

    // ---- 입력 포맷 : bf16 이거나, Q4.11 **정수 코드** ----
    // `layer_norm_2` 만 앞이 `linear1` 의 raw16 출력이라 정수입니다. bf16 은
    // 지수만 빼면 되므로(정규화된 값의 가수는 그대로) 곱셈이 없습니다 —
    // 2^-11 을 곱한 것과 **비트 단위로 같습니다.** 코드가 0 이면 지수도 0 이라
    // 건드리면 안 됩니다.
    wire [N*16-1:0] ln_rd_data;
    genvar lc;
    generate
        for (lc = 0; lc < N; lc = lc + 1) begin : LNCONV
            wire [15:0] c_bf;
            Int32_To_Bf16 #(.IN_W(16)) u_lc (
                .din(ar_data[lc*16 +: 16]), .bf16(c_bf));
            wire [15:0] c_q411 = (c_bf[14:7] == 8'd0) ? 16'd0
                               : {c_bf[15], c_bf[14:7] - 8'd11, c_bf[6:0]};
            assign ln_rd_data[lc*16 +: 16] =
                   q_flag2[0] ? c_q411 : ar_data[lc*16 +: 16];
        end
    endgenerate

    // 행타일 반복은 **래퍼가** 합니다 — 코어의 3단 Tile 파이프라인이 겹치려면
    // 타일을 끊지 않고 연달아 밀어야 하기 때문입니다. 그래서 엔진의 `ln_mt`
    // 루프가 없어졌습니다 (예전 코어는 타일마다 start 를 다시 걸었습니다).
    LayerNorm_Ev #(.N(N), .E(E), .DIM_W(DIM_W), .AW(AW_A), .XSW(6)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .done(ln_done),
        .M(q_M), .a_base(q_AIN), .in_shift($signed(q_gsh)),
        .rd_en(la_rd_en), .rd_addr(la_rd_addr), .rd_data(ln_rd_data),
        .p_addr(ln_paddr), .p_gamma(pg_gamma), .p_beta(pg_beta),
        .mult(pb_mult_q), .shift(q_sh),
        .out_valid(ln_ov), .out_mt(ln_mt_o), .out_k(ln_k), .out_data(ln_out));

    // PG(gamma/beta) 와 PB(재양자화 스칼라) 는 **다른 메모리**라 베이스도 둘입니다.
    assign pg_idx = q_OSTR[AW_PG+2:0] + ln_paddr[AW_PG+2:0];

    // =========================================================================
    // Softmax (attention)
    // =========================================================================
    reg             sm_start;
    wire            sm_done, sm_ov;
    wire [7:0]      sm_c;
    wire [N*8-1:0]  sm_out;

    Softmax_Attn #(.N(N), .CMAX(TOKMAX+1),
                   .EXP_FILE(EXP_LUT_FILE), .RCP_FILE(RCP_LUT_FILE)) u_sm (
        .clk(clk), .rst(rst), .start(sm_start), .C(q_NOUT[7:0]), .done(sm_done),
        .in_valid(sm_iv), .in_data(sm_id),
        .out_valid(sm_ov), .out_c(sm_c), .out_data(sm_out));

    // QK GEMM 의 Q6.9 컬럼을 메모리를 안 거치고 직결
    // **`sm_start` 로 한 번 더 막습니다.** `q_cons` 는 S_DEC 에서 바뀌는데 직전
    // GEMM 의 Col_Post 파이프는 아직 비워지는 중이라, 그 잔여 컬럼이 새 step 의
    // softmax 로 새어 들어갑니다. 그러면 길이가 확정되기 전에 첫 열이 들어가
    // **1열짜리 타일**이 만들어지고, 코어의 뱅크 포인터가 한 칸 어긋나
    // 이후 타일이 이전 길이로 나옵니다 (97 을 넣었는데 53 이 나왔습니다).
    always @(posedge clk) begin
        sm_iv <= cp_v && (q_cons == C_Q69) && sm_start;
        sm_id <= cp_q69;
    end

    // =========================================================================
    // RES : bf16 가산 (A_Mem 두 번 읽어 더함) — `fpga_nl` 과 동일
    // =========================================================================
    reg              rs_run;
    reg [DIM_W-1:0]  rs_k;
    reg [2:0]        rs_ph;
    reg [N*16-1:0]   rs_a;
    wire [N*16-1:0]  rs_sum;
    // 두 피연산자가 **정수 코드**이고 스케일이 서로 다른 자리가 하나 있습니다 —
    // `proc_events` 의 `x = seq_init(x) + x_input`. seq_init 출력은 ReLU 앞 int8
    // 격자(0.019991), x_input 은 preproc 출력의 int8 격자(0.038420) 입니다.
    //
    // ## 왜 bf16 곱셈이 아니라 정수인가
    //
    // 두 step 을 bf16 상수로 곱하면 상수 자체가 가수 8비트로 반올림돼 **비율이
    // 0.4 % 흔들립니다.** 결과는 bf16 한 칸(0.4 %)과 같은 크기라 절반이 어긋납니다.
    // 대신 step 을 `round(step * 2^RSH)` 정수로 두고 정수로 더한 뒤 **한 번만**
    // bf16 으로 내리면, 반올림이 골든과 같은 자리에서 한 번만 일어납니다.
    // 지수에서 RSH 를 빼는 것은 2의 거듭제곱이라 가수를 건드리지 않습니다.
    //
    // 스케일을 1 근처로 되돌리는 이유는 LayerNorm 의 `eps` 때문입니다 — 임의
    // 스케일로 두면 분산에 더하는 eps 의 상대 크기가 달라집니다.
    localparam RSH = 20;                       // step 상수의 소수 비트
    wire res_q = q_flag2[1];
    genvar rg;
    generate
        for (rg = 0; rg < N; rg = rg + 1) begin : RESLANE
            wire signed [15:0] ra_c = rs_a[rg*16 +: 16];
            wire signed [15:0] rb_c = ar_data[rg*16 +: 16];
            wire signed [31:0] rq_acc = $signed(ra_c) * $signed({1'b0, q_PB})
                                      + $signed(rb_c) * $signed({1'b0, q_OSTR});
            wire [15:0] rq_bf_raw;
            Int32_To_Bf16 #(.IN_W(32)) u_rqb (.din(rq_acc), .bf16(rq_bf_raw));
            wire [15:0] rq_bf = (rq_bf_raw[14:7] == 8'd0) ? 16'd0
                              : {rq_bf_raw[15], rq_bf_raw[14:7] - RSH[7:0],
                                 rq_bf_raw[6:0]};

            wire [31:0] sf;
            Fp32_Add u_ra (.clk(clk), .rst(rst),
                           .in_valid(rs_run && rs_ph == 3'd3 && !res_q),
                           .a({rs_a[rg*16 +: 16], 16'd0}),
                           .b({ar_data[rg*16 +: 16], 16'd0}),
                           .out_valid(), .y(sf));
            wire [15:0] sb;
            Fp32_To_Bf16 u_rc (.f(sf), .log2e(6'd0), .y(sb));
            // 정수 경로는 조합이라 ph3 에 확정됩니다. bf16 경로(Fp32_Add 2단)와
            // 쓰기 시점(ph5)을 맞추기 위해 두 단 늦춥니다.
            reg [15:0] rq_d1, rq_d2;
            always @(posedge clk) begin rq_d1 <= rq_bf; rq_d2 <= rq_d1; end
            assign rs_sum[rg*16 +: 16] = res_q ? rq_d2 : sb;
        end
    endgenerate

    // =========================================================================
    // MEAN : latent 96개 평균 → 1행 (`proc_embs_block` 의 gap)
    //
    // 레인이 행이라 **레인 축 리덕션**입니다 — GEMM/LayerNorm 과 반대 방향이라
    // 전용 덧셈 트리를 둡니다 (특징 하나당 한 번뿐이라 32입력 트리 하나면 됨).
    // 결과는 다음 GEMM 이 M=1 로 읽으므로 **레인 0** 에만 씁니다.
    //
    // 나누기 96 은 재양자화 곱수에 접혀 있습니다 (`pack_evt.py` 의 MEAN_NEXT).
    // =========================================================================
    reg  [DIM_W-1:0] mn_k;
    reg  [5:0]       mn_mt;
    reg  [1:0]       mn_ph;
    reg              mn_run;
    reg signed [15:0] mn_acc;
    integer mz;
    reg signed [15:0] mn_lane_sum;
    always @* begin                        // 32레인 int8 합 (유효 행만)
        mn_lane_sum = 16'sd0;
        for (mz = 0; mz < N; mz = mz + 1)
            if (mn_mt * N + mz < q_M)
                mn_lane_sum = mn_lane_sum + $signed(ar_data[mz*16 +: 16]);
    end
    wire signed [47:0] mn_prod = $signed(mn_acc) * $signed(g_mult_q);
    wire signed [47:0] mn_rnd  = mn_prod + (48'sd1 <<< (q_gsh - 1));
    wire signed [47:0] mn_sh   = mn_rnd >>> q_gsh;
    wire signed [7:0]  mn_out  = (mn_sh >  127) ?  8'sd127
                               : (mn_sh < -128) ? -8'sd128 : mn_sh[7:0];

    // =========================================================================
    // ARGMAX : 마지막 GEMM 의 컬럼을 메모리에 안 쓰고 최대값만 고릅니다
    //   골든 note : argmax(acc[c]*M[c]) — shift 는 순서를 안 바꿔 생략 가능
    // =========================================================================
    wire signed [PSUM_W-1:0] am_acc = $signed(col_d_d2[PSUM_W-1:0])
                                    + $signed(pb_bias_q);
    wire signed [63:0]       am_val = am_acc * $signed(pb_mult_q);
    reg  signed [63:0]       am_best;
    reg                      am_any;

    // =========================================================================
    // A_Mem 포트 중재 + 쓰기
    // =========================================================================
    integer za;
    always @* begin
        ar_en = 1'b0; ar_addr = {AW_A{1'b0}};
        br_en = 1'b0; br_addr = {AW_A{1'b0}};
        a_we_en = 1'b0; a_we_addr = {AW_A{1'b0}}; a_we_data = {N*16{1'b0}};

        if (st == S_IDLE) begin
            ar_en = dbg_rd_en; ar_addr = dbg_rd_addr;
        end else if (st == S_GCONST) begin
            // QK step 시작에 bias_k 워드를 한 번 읽어 둡니다 (GELU 곱수와 같이)
            ar_en = 1'b1; ar_addr = q_AOUT[AW_A-1:0];
        end else begin
            case (q_kind)
                K_ARGMAX,
                K_GEMM: begin
                    ar_en = ga_rd_en; ar_addr = ga_rd_addr;
                    if (b_from_a) begin
                        br_en   = gb_rd_en;
                        br_addr = av_hit ? q_OSTR[AW_A-1:0]
                                         : gb_rd_addr[AW_A-1:0];
                    end
                    if (tr_run) begin
                        // 전치 드레인 : 워드 = head*OSTR + mt*32 + 키
                        a_we_en   = 1'b1;
                        // `tr_r` 은 6비트입니다 — `tr_r[AW_A-1:0]` 처럼 범위를
                        // 넘겨 잘라 쓰면 Verilog 는 **조용히 X** 를 줍니다
                        // (주소 전체가 X 가 돼 V 영역이 통째로 안 써졌습니다).
                        a_we_addr = q_AOUT + tr_h * q_OSTR[AW_A-1:0]
                                  + (tr_mt << 5) + {{(AW_A-6){1'b0}}, tr_r};
                        for (za = 0; za < N; za = za + 1)
                            a_we_data[za*16 +: 16] =
                                {{8{tr_out[za*8+7]}}, tr_out[za*8 +: 8]};
                    end else if (sm_ov && q_cons == C_Q69) begin
                        // softmax 출력 : 워드 = 키, 레인 = latent 행
                        a_we_en   = 1'b1;
                        a_we_addr = q_OSTR[AW_A-1:0] + {{(AW_A-8){1'b0}}, sm_c};
                        for (za = 0; za < N; za = za + 1)
                            a_we_data[za*16 +: 16] = {8'd0, sm_out[za*8 +: 8]};
                    end else if (cp_v && q_cons != C_Q69 && q_kind == K_GEMM
                                 && !q_flag[0]) begin
                        a_we_en   = 1'b1;
                        // FLAG[13] head-major : AOUT + (n/32)*hstride + mt*32 + n%32
                        // 그 외        : AOUT + mt*NOUT + n
                        a_we_addr = q_flag[1]
                                  ? (q_AOUT + (cp_n[DIM_W-1:5] * q_OSTR[AW_A-1:0])
                                            + (cp_mt << 5) + cp_n[4:0])
                                  : (q_AOUT + cp_mt * q_OSTR + cp_n[AW_A-1:0]);
                        a_we_data = cp_d;
                    end
                end
                K_LN: begin
                    ar_en = la_rd_en; ar_addr = la_rd_addr;
                    if (ln_ov) begin
                        a_we_en   = 1'b1;
                        a_we_addr = q_AOUT + ln_mt_o * q_NOUT[AW_A-1:0]
                                  + ln_k[AW_A-1:0];
                        for (za = 0; za < N; za = za + 1)
                            a_we_data[za*16 +: 16] =
                                {{8{ln_out[za*8+7]}}, ln_out[za*8 +: 8]};
                    end
                end
                K_SMAX: begin
                    if (sm_ov) begin
                        a_we_en   = 1'b1;
                        a_we_addr = q_AOUT + {{(AW_A-8){1'b0}}, sm_c};
                        for (za = 0; za < N; za = za + 1)
                            a_we_data[za*16 +: 16] = {8'd0, sm_out[za*8 +: 8]};
                    end
                end
                K_RES: begin
                    ar_en   = rs_run;
                    ar_addr = (rs_ph <= 3'd1)
                            ? (q_AIN  + rs_mt * q_K[AW_A-1:0] + rs_k[AW_A-1:0])
                            : (q_AOUT + rs_mt * q_K[AW_A-1:0] + rs_k[AW_A-1:0]);
                    if (rs_run && rs_ph == 3'd5) begin
                        a_we_en   = 1'b1;
                        a_we_addr = q_AOUT + rs_mt * q_K[AW_A-1:0]
                                  + rs_k[AW_A-1:0];
                        a_we_data = rs_sum;
                    end
                end
                K_POS: begin
                    ar_en   = pos_rd_en; ar_addr = pos_rd_addr;
                    a_we_en = pos_we_en; a_we_addr = pos_we_addr;
                    a_we_data = pos_we_data;
                end
                K_MEAN: begin
                    ar_en   = mn_run;
                    ar_addr = q_AIN + mn_mt * q_K[AW_A-1:0] + mn_k[AW_A-1:0];
                    if (mn_run && mn_ph == 2'd2) begin
                        a_we_en   = 1'b1;
                        a_we_addr = q_AOUT + mn_k[AW_A-1:0];
                        a_we_data = {{(N-1)*16{1'b0}}, {{8{mn_out[7]}}, mn_out}};
                    end
                end
                default: ;
            endcase
        end
    end

    // PB 인덱스
    always @* begin
        // GELU 뒤 재양자화 곱수는 그 레이어 채널 뒤(PB + NOUT)에 있습니다
        // attention 의 QK/AV 는 채널별이 아니라 **블록당 스칼라 하나**입니다
        // (FLAG2[2] 가 서 있는 step 이 정확히 그 둘입니다).
        //   ARGMAX 도 **채널별**입니다 — 골든이 `argmax(acc[c]*M[c])` 이고
        //   acc 에 채널 바이어스가 들어갑니다. `K_GEMM` 만 걸어 두면 10클래스가
        //   전부 채널 0 의 곱수를 써서 사실상 `argmax(acc[c])` 가 됩니다.
        if      (st == S_GCONST)   pb_idx = q_PB[AW_PB+1:0] + q_NOUT[AW_PB+1:0];
        else if ((q_kind == K_GEMM || q_kind == K_ARGMAX) && !q_flag2[2])
                                   pb_idx = q_PB[AW_PB+1:0] + col_n[AW_PB+1:0];
        else                       pb_idx = q_PB[AW_PB+1:0];
    end

    // =========================================================================
    // 컨트롤
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; sp <= 0; ti <= 0; done <= 1'b0; in_tail <= 1'b0;
            gm_start <= 1'b0; ln_start <= 1'b0; sm_start <= 1'b0;
            rs_run <= 1'b0; rs_k <= 0; rs_ph <= 0; gc <= 0; g_mult_q <= 0;
            q_flag2 <= 4'd0; rs_mt <= 6'd0;
            wait_ack <= 1'b0; bk_word <= 0;
            s_addr <= 0; n_tok <= 0;
            res_class <= 4'd0; res_logits <= 0; tok_req <= 1'b0; pos_start <= 1'b0;
            mn_run <= 1'b0; mn_k <= 0; mn_mt <= 0; mn_ph <= 0; mn_acc <= 0;
            am_best <= 0; am_any <= 1'b0;
        end else begin
            // 코어 파이프라인이 마지막 타일을 비우며 컬럼을 더 낼 수 있어
            // **유효 범위 밖은 세지 않습니다** (안 막으면 클래스 10 이 나옵니다)
            if (q_kind == K_ARGMAX && col_v_d2 && col_n_d2 < q_NOUT) begin
                res_logits[col_n_d2[4:0]*PSUM_W +: PSUM_W] <= am_acc;
                if (!am_any || am_val > am_best) begin
                    am_best   <= am_val;
                    am_any    <= 1'b1;
                    res_class <= col_n_d2[3:0];
                end
            end
            case (st)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        ti <= 0; sp <= 0; in_tail <= 1'b0;
                        s_addr <= 0; st <= S_TLOAD;
                    end
                end
                // 이 타임스텝의 X/PIN 이 채워지기를 기다립니다
                S_TLOAD: begin
                    tok_req <= 1'b1;
                    if (tok_ack) begin
                        tok_req <= 1'b0;
                        n_tok   <= tok_rd_n;
                        st      <= S_FETCH;
                    end
                end
                // Step_Mem 읽기 1사이클
                S_FETCH: begin
                    n_tok <= tok_rd_n;
                    st <= S_DEC;
                end
                S_DEC: begin
                    q_kind <= s_rd[3:0];   q_cons <= s_rd[5:4];
                    q_act  <= s_rd[7:6];   q_var  <= s_rd[11:8];
                    q_flag <= s_rd[15:12];
                    q_sh   <= s_rd[21:16]; q_gsh  <= s_rd[27:22];
                    q_flag2 <= s_rd[31:28];
                    // VAR 비트로 n_tok 의존 필드를 채웁니다 (부분선택을 두 번
                    // 이어 쓰면 Verilog 문법 오류라 개별 비트로 씁니다)
                    q_M    <= s_rd[8]  ? tok_rd_n          : s_rd[32 +: DIM_W];
                    q_K    <= s_rd[10] ? (tok_rd_n + 1'b1) : s_rd[64 +: DIM_W];
                    q_NOUT <= (s_rd[9] | s_rd[11]) ? (tok_rd_n + 1'b1)
                                                   : s_rd[96 +: DIM_W];
                    q_AIN  <= s_rd[128 +: AW_A];
                    q_BIN  <= s_rd[160 +: AW_W];
                    q_AOUT <= s_rd[192 +: AW_A];
                    q_PB   <= s_rd[224 +: 16];
                    q_OSTR <= s_rd[240 +: 16];
                    gc <= 0;
                    st <= S_GCONST;
                end
                // PB 읽기 2사이클 → GELU 뒤 재양자화 곱수 확정
                S_GCONST: begin
                    gc <= gc + 1'b1;
                    if (gc == 2'd1) bk_word <= ar_data;
                    if (gc == 2'd2) begin g_mult_q <= pb_mult_q; st <= S_RUN; end
                end
                S_RUN: begin
                    case (q_kind)
                        K_GEMM: begin
                                    gm_start <= 1'b1;
                                    if (q_cons == C_Q69) sm_start <= 1'b1;
                                end
                        K_LN:   ln_start <= 1'b1;
                        K_SMAX: sm_start <= 1'b1;
                        K_RES:  begin rs_run <= 1'b1; rs_k <= 0; rs_ph <= 0;
                                      rs_mt <= 6'd0; end
                        K_MEAN: begin mn_run <= 1'b1; mn_k <= 0; mn_mt <= 0;
                                      mn_ph <= 0; mn_acc <= 0; end
                        K_POS:  pos_start <= 1'b1;
                        K_ARGMAX: begin gm_start <= 1'b1; am_any <= 1'b0;
                                        am_best <= 0; res_class <= 4'd0; end
                        default: ;
                    endcase
                    wait_ack <= 1'b0;
                    st <= S_WAIT;
                end
                S_WAIT: begin
                    if (!gm_done && !ln_done && !sm_done && !pos_done)
                        wait_ack <= 1'b1;
                    if (q_kind == K_RES) begin
                        rs_ph <= rs_ph + 1'b1;
                        if (rs_ph == 3'd1) rs_a <= ar_data;
                        if (rs_ph == 3'd5) begin
                            rs_ph <= 0;
                            if (rs_k == q_K - 1) begin
                                rs_k <= 0;
                                if (rs_mt == rs_mt_last) begin
                                    rs_run <= 1'b0; st <= S_NEXT;
                                end else rs_mt <= rs_mt + 1'b1;
                            end else rs_k <= rs_k + 1'b1;
                        end
                    // 전치 드레인이 **다 쏟기 전에** step 을 넘기면 마지막
                    // 워드들이 다음 step 의 AOUT/OSTR 로 나갑니다 (실측 256개 중
                    // 23개가 다음 step 으로 넘어갔습니다).
                    end else if (q_kind == K_GEMM && gm_done && wait_ack
                                 && !tr_run && !tr_arm && !tr_go && !cp_busy
                                 && (q_cons != C_Q69 || sm_done)) begin
                        // Q6.9 소비자는 softmax 까지 **같은 step** 입니다
                        gm_start <= 1'b0; sm_start <= 1'b0; st <= S_NEXT;
                    end else if (q_kind == K_LN && ln_done && wait_ack) begin
                        ln_start <= 1'b0; st <= S_NEXT;
                    end else if (q_kind == K_POS && pos_done && wait_ack) begin
                        pos_start <= 1'b0; st <= S_NEXT;
                    end else if (q_kind == K_SMAX && sm_done && wait_ack) begin
                        sm_start <= 1'b0; st <= S_NEXT;
                    end else if (q_kind == K_ARGMAX && gm_done && wait_ack) begin
                        gm_start <= 1'b0; st <= S_NEXT;
                    end else if (q_kind == K_MEAN) begin
                        // 특징 하나당 : 타일마다 (주소 → 합), 그 뒤 재양자화 · 쓰기
                        if (mn_ph == 2'd0) mn_ph <= 2'd1;
                        else if (mn_ph == 2'd1) begin
                            mn_acc <= mn_acc + mn_lane_sum;
                            if (mn_mt == ln_mt_last) mn_ph <= 2'd2;
                            else begin mn_mt <= mn_mt + 1'b1; mn_ph <= 2'd0; end
                        end else begin
                            mn_ph <= 2'd0; mn_mt <= 0; mn_acc <= 0;
                            if (mn_k == q_K - 1) begin
                                mn_run <= 1'b0; st <= S_NEXT;
                            end else mn_k <= mn_k + 1'b1;
                        end
                    end
                end
                S_NEXT: begin
                    if (!gm_done && !ln_done) begin
                        if (!in_tail && sp == n_body - 1) begin
                            // 타임스텝 하나 끝 — 다음 타임스텝 또는 tail 로
                            st <= S_TSTEP;
                        end else if (in_tail && sp == n_body + n_tail - 1) begin
                            st <= S_DONE;
                        end else begin
                            sp <= sp + 1'b1; s_addr <= sp + 1'b1; st <= S_FETCH;
                        end
                    end
                end
                S_TSTEP: begin
                    if (ti == n_time - 1) begin
                        in_tail <= 1'b1; sp <= n_body; s_addr <= n_body;
                        st <= S_FETCH;
                    end else begin
                        ti <= ti + 1'b1; sp <= 0; s_addr <= 0;
                        st <= S_TLOAD;          // 다음 타임스텝 입력 대기
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                    if (!start) st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
