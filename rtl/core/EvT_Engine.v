// -----------------------------------------------------------------------------
// EvT_Engine : EvT(DVS128_10) 실행기 — 명령어 프로그램을 순서대로 돌립니다
//
// Inst_Mem 에서 256비트 명령어를 하나씩 읽어 KIND 에 맞는 유닛을 기동하고, 끝날
// 때까지 기다렸다가 다음 명령어로 넘어갑니다 (ST_FETCH → ST_DECODE → ST_CONST →
// ST_RUN → ST_WAIT → ST_NEXT). 타임스텝 T(<=20) 를 돌며 latent 를 누적하고,
// 마지막에 tail 명령어 5개로 분류합니다.
//
// 신호 이름 규칙과 용어(`inst`/`tstep`/`scale`)는 `rtl/NAMING.md` 를 보십시오.
//
// ## 프로그램은 정적입니다
//
// `n_tok` 은 타임스텝마다 다릅니다(실측 16~123). 영역 크기를 `n_tok` 에 맞추면
// 베이스가 매번 바뀌어 프로그램을 5,182벌 만들어야 합니다. 그래서 **모든 영역
// stride 를 최악치(토큰 128)로 고정**했습니다(`sw/schedule_evt.py`). 베이스가
// 전부 상수가 되고, `n_tok` 에 따라 바뀌는 것은 VAR 이 표시한 네 필드뿐입니다:
//
//   VAR[0] M <- n_tok      VAR[1] NOUT <- Lk
//   VAR[2] K <- Lk         VAR[3] C    <- Lk        (Lk = n_tok + 1)
//
// ST_DECODE 가 `inst_word` 의 VAR 비트를 보고 그 자리에 `tok_n` 을 넣습니다
// (그래서 `op_var` 라는 레지스터가 없습니다). latent 쪽 명령어는 Lk = 96+1 이
// 상수라 VAR 을 안 씁니다.
//
// ## 명령어 워드 (256비트 = Inst_Mem 한 워드)
//
//   [ 31: 0] KIND[3:0] FMT[5:4] ACT[7:6] VAR[11:8] FLAG[15:12]
//            SHIFT[21:16] SHIFT2[27:22] FLAG2[31:28]
//   [ 63:32] M      [ 95:64] K      [127:96] NOUT
//   [159:128] AIN   [191:160] BIN   [223:192] AOUT
//   [255:224] RQ_BASE[15:0] | OSTR[31:16]
//
//   KIND  0 GEMM  1 LN  2 SMAX  3 RES  4 MEAN  5 ARGMAX  6 POS
//   FMT   0 int8(+ACT)  1 Q4.11→GELU→int8  2 bf16  3 Q6.9(softmax 직결)
//   FLAG  [0] 출력을 Transpose32 경유로   [1] 출력을 head-major 주소로
//         [2] B 피연산자를 A_Mem 에서     [3] Q4.11 을 16b 그대로 (raw16)
//   FLAG2 [0] LN 입력이 Q4.11 코드        [1] RES 두 피연산자가 정수 코드
//         [2] bias_k/bias_v 토큰 끼워넣기 [3] 활성함수 뒤 2차 재양자화
//
// `OSTR` · `RQ_BASE` · `SHIFT2` 는 KIND 마다 뜻이 다릅니다. 아래 선언부에
// **쓰는 자리마다 이름 붙인 별칭 wire** 를 두었습니다 (`ostr_as_*`). 표는
// `rtl/NAMING.md` §4.3 — 새 용도를 추가하면 별칭도 같이 추가하십시오.
//
// ## A_Mem 은 읽기 2포트입니다
//
// 읽기를 GEMM(A) · GEMM(B) · LN · RES · MEAN · POS 가 나눠 쓰는데, GEMM 과 RES 는
// **두 워드를 동시에** 읽습니다. Bram_Sdp 두 벌에 같은 내용을 미러링해 읽기
// 포트를 둘로 만들었습니다 (`a_ra_*` / `a_rb_*`). BRAM 이 남으므로 이게 가장
// 단순합니다.
//
// ## 바꾸기 전에
//
// `tb_evt` 가 골든과 tap 대조를 하고(140,170 점 중 알려진 양자화 오차 2건),
// ZCU102 에서 187.5 MHz 로 합성·구현·보드 검증까지 끝난 코드입니다. 고친 뒤에는
// `tb/run_vcs.sh` 로그를 **변경 전과 diff** 하십시오 (`rtl/NAMING.md` §8).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module EvT_Engine #(
    parameter N      = 32,
    parameter ACT_W  = 8,
    // 가중치는 int4 고정입니다 (A8W4). W_Mem 레인 하나가 **니블 두 개**를 담고
    // DSP48E2 의 pre-adder 가 둘을 한 곱셈에 실어 배열 처리량을 두 배로 냅니다
    // (`PE_OS.v` 의 "DSP 패킹"). 활성값은 8비트 그대로입니다.
    parameter PSUM_W = 32,
    parameter DIM_W  = 16,
    parameter E      = 128,
    parameter LAT    = 96,
    parameter HEADS  = 4,
    parameter HD     = 32,
    parameter TOKMAX = 128,
    parameter N_CLASS = 10,      // 분류 클래스 수 (DVS128_10)
    parameter AW_W   = 14,       // 명령어의 B 베이스 폭 (W_Mem 실사용 7,000 워드)
    // A_Mem 실사용 7,649 워드 (`schedule.json` 의 a_words) → 2^13 = 8,192 로 충분.
    // 여유가 543 워드뿐이므로 영역을 늘릴 때 `schedule_evt.py` 상한을 보십시오.
    // [타이밍] 14 로 두면 BRAM 깊이 캐스케이드가 4단이 돼 읽기가 1.69 ns 입니다.
    parameter AW_A   = 13,       // A_Mem  (7,649 워드 사용)
    parameter AW_RQ  = 10,       // Requant_Mem (860 워드,  구 PB_Mem)
    parameter AW_AF  = 8,        // Affine_Mem  (208 워드,  구 PG_Mem)
    parameter AW_INST = 8       // Inst_Mem   (155 워드,  구 Step_Mem)
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 start,
    output reg                  done,
    output wire                 busy,
    output wire [3:0]           dbg_state,
    output wire [AW_INST-1:0]      dbg_inst,

    // ---- 실행 파라미터 ----
    input  wire [AW_INST-1:0]      n_body,        // 전처리 + Attention 명령어 수
    input  wire [AW_INST-1:0]      n_tail,        // Classifier 명령어 수
    input  wire [5:0]           n_tstep,        // 타임스텝 T (Time-window)
    input  wire [31:0]          eps,           // LayerNorm eps (fp32 비트)

    // ---- 타임스텝 입력 요청 ----
    // X/PIN 영역은 **한 타임스텝분**만 들어갑니다 (20 타임스텝을 다 담으면
    // 24k 워드로 A_Mem 을 넘습니다). 그래서 타임스텝마다 호스트가 새로 채워야
    // 하고, 엔진은 채워질 때까지 기다립니다.
    //
    //   tok_req = 1 로 멈춤  →  호스트가 tstep_idx 번째 타임스텝의
    //   X / PIN(pos enc) 를 적재하고 tok_ack 를 한 번 올림  →  진행
    //
    // `tok_n` 은 그 타임스텝의 토큰 수입니다 (ack 시점에 래치).
    output wire [5:0]           tstep_idx,
    input  wire [DIM_W-1:0]     tok_n,
    output reg                  tok_req,
    input  wire                 tok_ack,

    // ---- 메모리 적재 ----
    input  wire                 ld_we,
    input  wire [2:0]           ld_sel,        // LD_* (아래 localparam)
    input  wire [AW_W-1:0]      ld_addr,
    input  wire [N*16-1:0]      ld_data,

    // ---- 결과 ----
    output reg  [3:0]           res_class,     // argmax
    output reg  [N_CLASS*PSUM_W-1:0] res_logits,  // 클래스별 acc (디버그)

    // ---- A_Mem 리드백 (IDLE 일 때만) ----
    input  wire                 dbg_rd_en,
    input  wire [AW_A-1:0]      dbg_rd_addr,
    output wire [N*16-1:0]      dbg_rd_data
);
    // 로더 목적지 (`ld_sel`) — Top.v 의 LOAD_SEL 레지스터와 같은 인코딩
    localparam [2:0] LD_W = 3'd0, LD_A = 3'd1, LD_RQ = 3'd2,
                     LD_AF = 3'd3, LD_INST = 3'd4, LD_POS = 3'd5;

    localparam OP_GEMM=4'd0, OP_LN=4'd1, OP_SMAX=4'd2, OP_RES=4'd3,
               OP_MEAN=4'd4, OP_ARGMAX=4'd5, OP_POS=4'd6;
    localparam FMT_INT8=2'd0, FMT_Q411=2'd1, FMT_BF16=2'd2, FMT_Q69=2'd3;

    genvar lane;                 // 모든 generate 루프가 함께 씁니다

    // 인스턴스보다 먼저 선언해야 하는 신호들
    // (사용이 선언보다 앞서면 Verilog 가 1비트 net 으로 암묵 선언합니다)
    reg                 col_valid_d1, col_valid_d2;
    reg [N*PSUM_W-1:0]  col_data_d1,  col_data_d2;
    reg [DIM_W-1:0]     col_n_d1,     col_n_d2;
    reg [DIM_W-1:0]     col_mt_d1,    col_mt_d2;
    reg                 smax_in_valid;
    reg [N*16-1:0]      smax_in_data;

    // =========================================================================
    // FSM
    // =========================================================================
    localparam ST_IDLE=4'd0, ST_FETCH=4'd1, ST_DECODE=4'd2, ST_CONST=4'd3,
               ST_RUN=4'd4, ST_WAIT=4'd5, ST_NEXT=4'd6, ST_TSTEP=4'd7, ST_DONE=4'd8,
               ST_TLOAD=4'd9;
    reg [3:0]         state;
    reg [AW_INST-1:0] inst_ptr;    // 명령어 포인터 (프로그램 카운터)
    reg [5:0]         tstep;       // 타임스텝 인덱스
    reg               in_tail;     // body 를 다 돌고 tail 을 도는 중
    reg [DIM_W-1:0]   n_tok;       // 이번 타임스텝의 토큰 수

    assign busy       = (state != ST_IDLE);
    assign dbg_state  = state;
    assign dbg_inst   = inst_ptr;
    assign tstep_idx  = tstep;

    // =========================================================================
    // Inst_Mem — 명령어 하나 = 256비트 워드 하나
    // =========================================================================
    wire [N*8-1:0]    inst_word;
    reg  [AW_INST-1:0] inst_addr;
    Bram_Sdp #(.DW(N*8), .AW(AW_INST)) u_inst_mem (
        .clk(clk), .we_en(ld_we && ld_sel == LD_INST), .we_addr(ld_addr[AW_INST-1:0]),
        .we_be({N{1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(1'b1), .rd_addr(inst_addr), .rd_data(inst_word));

    // =========================================================================
    // 디코드된 명령어 필드 (ST_DECODE 에서 래치, 명령어 내내 고정)
    // =========================================================================
    reg [3:0]        op_kind;                 // OP_*
    reg [1:0]        op_fmt;                  // FMT_*  (출력 포맷)
    reg [1:0]        op_act;                  // Activation.v 인코딩
    reg [3:0]        op_flag, op_flag2;
    // 시프트는 32레인 x 2벌 = DSP 64개를 먹습니다. 복제해 배선을 끊습니다.
    (* max_fanout = 16 *)
    reg [5:0]        op_shift, op_shift2;
    reg [DIM_W-1:0]  op_m, op_k, op_nout;     // GEMM 형상
    reg [AW_A-1:0]   op_ain, op_aout;         // A_Mem 피연산자 / 결과 베이스
    reg [AW_W-1:0]   op_bin;                  // B 베이스 (W_Mem 또는 A_Mem)
    reg [15:0]       op_rq_base, op_ostr;     // 아래 별칭 참조

    // `op_ostr` / `op_rq_base` 는 KIND 마다 뜻이 다릅니다 (머리말 참조).
    // 쓰는 자리마다 **이름 붙인 별칭**을 두어 읽는 사람이 헷갈리지 않게 합니다.
    // 새 용도를 추가하면 별칭도 같이 추가하십시오.
    wire [AW_A-1:0]  ostr_as_bkv_addr   = op_ostr[AW_A-1:0];   // AV  : bias_v 워드
    wire [AW_A-1:0]  ostr_as_smax_base  = op_ostr[AW_A-1:0];   // QK  : softmax 출력
    wire [AW_AF+2:0] ostr_as_af_base    = op_ostr[AW_AF+2:0];  // LN  : Affine_Mem
    wire [15:0]      ostr_as_scale_b    = op_ostr;             // RES : B 의 정수 scale
    wire [15:0]      rq_base_as_scale_a = op_rq_base;          // RES : A 의 정수 scale

    // 행타일 : 레인이 곧 행이므로 한 바퀴가 32행 (M=52 면 두 바퀴, 96 이면 세 바퀴).
    // RES 는 A_Mem 이 `base + mt*K + k` 라, LayerNorm 과 MEAN 은 특징 축 리덕션이라
    // 행타일마다 다시 돕니다. 명령어를 타일마다 쪼개면 프로그램이 n_tok 에
    // 의존하게 되므로(최악 4벌) 엔진이 셉니다.
    wire [5:0]       row_tile_last = (op_m > 0) ? ((op_m - 1'b1) >> 5) : 6'd0;
    reg [5:0]        rs_mt;                   // 현재 행타일 (RES)
    reg [1:0]        const_ph;                // ST_CONST 의 3사이클 위상

    // [함정] 하위 코어의 `done` 은 **다음 start 까지 계속 1** 입니다. start 를 준
    // 사이클에 그대로 보면 직전 완료를 이번 완료로 착각해 타일 하나를 통째로
    // 건너뜁니다. done 이 한 번 0 으로 내려간 것을 본 뒤부터 인정합니다.
    reg              wait_ack;


    // =========================================================================
    // 메모리
    // =========================================================================
    // 한 워드가 출력채널 **64개**를 담습니다 — 레인 j 에 {w[n+32], w[n]} 두 니블.
    // 32x8 = 256b 로 A8W8 과 폭이 같지만 담는 채널 수는 두 배라 워드 수가 절반
    // (14,000 -> 7,000) 이고, 그래서 깊이를 2^13 으로 줄여 BRAM 을 A8W4 수준
    // (379 타일) 으로 유지합니다.
    localparam AW_WM = 13;
    wire            w_rd_en;
    wire [AW_W-1:0] w_rd_addr;
    wire [N*8-1:0]  w_rd_data;
    Bram_Sdp #(.DW(N*8), .AW(AW_WM)) u_w_mem (
        .clk(clk), .we_en(ld_we && ld_sel == LD_W), .we_addr(ld_addr[AW_WM-1:0]),
        .we_be({(N){1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(w_rd_en), .rd_addr(w_rd_addr[AW_WM-1:0]), .rd_data(w_rd_data));

    // A_Mem : 읽기 2포트가 필요합니다 (GEMM 이 A 와 B 를 동시에 읽는 경우).
    // 같은 내용을 두 벌에 미러링합니다 — BRAM 이 남고 제어가 가장 단순합니다.
    reg              a_we_en;
    reg  [AW_A-1:0]  a_we_addr;
    reg  [N*16-1:0]  a_we_data;
    reg              a_ra_en, a_rb_en;
    reg  [AW_A-1:0]  a_ra_addr, a_rb_addr;
    wire [N*16-1:0]  a_ra_data, a_rb_data;

    wire a_wr_from_ld = ld_we && ld_sel == LD_A;
    wire            a_wr_en   = a_wr_from_ld ? 1'b1              : a_we_en;
    wire [AW_A-1:0] a_wr_addr = a_wr_from_ld ? ld_addr[AW_A-1:0] : a_we_addr;
    wire [N*16-1:0] a_wr_data = a_wr_from_ld ? ld_data           : a_we_data;

    Bram_Sdp #(.DW(N*16), .AW(AW_A)) u_a_mem0 (
        .clk(clk), .we_en(a_wr_en), .we_addr(a_wr_addr), .we_be({(N*2){1'b1}}),
        .we_data(a_wr_data), .rd_en(a_ra_en), .rd_addr(a_ra_addr), .rd_data(a_ra_data));
    Bram_Sdp #(.DW(N*16), .AW(AW_A)) u_a_mem1 (
        .clk(clk), .we_en(a_wr_en), .we_addr(a_wr_addr), .we_be({(N*2){1'b1}}),
        .we_data(a_wr_data), .rd_en(a_rb_en), .rd_addr(a_rb_addr), .rd_data(a_rb_data));

    assign dbg_rd_data = a_ra_data;

    // Requant_Mem : 채널 c 의 {scale, bias} — 워드 하나에 4채널
    //   scale = round(s_x * s_w[c] / lsb_out * 2^shift)  — 역양자화와 재양자화를
    //           곱수 하나로 접은 값입니다 (bf16 소비자는 bf16 상수 16비트).
    wire [AW_RQ-1:0] rq_word;
    wire [N*8-1:0]   rq_rd;
    reg  [AW_RQ+1:0] rq_idx;
    assign rq_word = rq_idx[AW_RQ+1:2];
    Bram_Sdp #(.DW(N*8), .AW(AW_RQ)) u_rq_mem (
        .clk(clk), .we_en(ld_we && ld_sel == LD_RQ), .we_addr(ld_addr[AW_RQ-1:0]),
        .we_be({N{1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(1'b1), .rd_addr(rq_word), .rd_data(rq_rd));
    reg [1:0] rq_sub_d1;
    always @(posedge clk) rq_sub_d1 <= rq_idx[1:0];
    wire signed [31:0] rq_scale = rq_rd[{rq_sub_d1, 6'd0}      +: 32];
    wire signed [31:0] rq_bias = rq_rd[{rq_sub_d1, 6'd0} + 32 +: 32];

    // 1차 재양자화 상수 — **컬럼마다 바뀝니다** (채널별).
    // [타이밍] BRAM → 4:1 서브워드 먹스 → 누산기 덧셈 → Requant_Bf16 이 한
    // 사이클에 붙어 있었습니다. 상수를 레지스터로 받아 갈라 놓습니다 (컬럼도
    // 같이 늦추므로 사이클 비용은 타일당 1).
    // 남은 소비자는 LayerNorm_Top 의 requant 와 ARGMAX 뿐입니다 — Format_Cast_Act
    // 는 8레인마다 직접 뜨므로(`bias_l[]`) 여기를 거치지 않습니다.
    // `max_fanout` 은 **복제 지시**입니다. 상수라 복제 비용은 FF 몇 개뿐입니다.
    (* max_fanout = 16 *)
    reg signed [31:0] rq_scale_q, rq_bias_q;
    always @(posedge clk) begin
        rq_scale_q <= rq_scale;
        rq_bias_q <= rq_bias;
    end

    // Affine_Mem : 특징 k 의 {gamma, beta}  — 워드 하나에 8특징
    wire [AW_AF-1:0] af_word;
    wire [N*8-1:0]   af_rd;
    wire [AW_AF+2:0] af_idx;
    assign af_word = af_idx[AW_AF+2:3];
    Bram_Sdp #(.DW(N*8), .AW(AW_AF)) u_af_mem (
        .clk(clk), .we_en(ld_we && ld_sel == LD_AF), .we_addr(ld_addr[AW_AF-1:0]),
        .we_be({N{1'b1}}), .we_data(ld_data[N*8-1:0]),
        .rd_en(1'b1), .rd_addr(af_word), .rd_data(af_rd));
    reg [2:0] af_sub_d1;
    always @(posedge clk) af_sub_d1 <= af_idx[2:0];
    wire signed [15:0] af_gamma = af_rd[{af_sub_d1, 5'd0}      +: 16];
    wire signed [15:0] af_beta  = af_rd[{af_sub_d1, 5'd0} + 16 +: 16];

    // =========================================================================
    // GEMM
    // =========================================================================
    reg                 gemm_start;
    wire                gemm_done;
    wire                col_valid;
    wire [N*PSUM_W-1:0] col_data;        // 출력채널 하나에 대한 32행
    wire [DIM_W-1:0]    col_n, col_mt;   // 전역 출력채널 n / 행타일 번호
    wire                gemm_a_rd_en,   gemm_b_rd_en;
    wire [AW_A-1:0]     gemm_a_rd_addr;
    wire [AW_W-1:0]     gemm_b_rd_addr;

    // B 피연산자 출처 : Linear 은 W_Mem, attention(Q·Kᵀ, attn·V)은 A_Mem
    wire b_src_amem = (op_kind == OP_GEMM) && op_flag[2];
    // W_Mem 레인은 이미 {w1, w0} 두 니블이라 **그대로** 배열에 넣습니다. 부호확장도
    // 자리 맞추기도 DSP48E2 의 pre-adder 가 합니다 (`PE_OS.v` 참고). 니블 순서는
    // `pack_evt.py` 가 만드는 레이아웃과 한 벌입니다 — 하위 = 출력채널 nt*64+j,
    // 상위 = nt*64+32+j.
    wire [N*8-1:0] gemm_b_from_w = w_rd_data;
    wire [N*8-1:0] gemm_b_from_a;
    generate
        for (lane = 0; lane < N; lane = lane + 1) begin : g_b_narrow
            assign gemm_b_from_a[lane*8 +: 8] = a_rb_data[lane*16 +: 8];
        end
    endgenerate
    // ---- bias_k / bias_v 토큰 (FLAG2[2]) ----
    // attention 은 키가 `Lk = n_tok + 1` 개이고 마지막 하나가 학습된 상수
    // 토큰입니다. 위치(키 인덱스 n_tok)가 타임스텝마다 달라 K/V 에 써 넣으려면
    // 읽고-고쳐-쓰기가 필요합니다. 대신 **읽는 쪽에서 끼워 넣습니다** — 상수는
    // 각 head 영역 끝의 예약 칸에 호스트가 한 번 넣어 둡니다.
    //
    //   QK : B 워드의 레인 = 키. 키 n_tok 인 **레인 하나**만 바꿉니다.
    //        bias_k 워드는 레인 = head_dim 이라 명령어 시작에 한 번 읽어 둡니다.
    //        주소는 `AOUT` — QK 는 메모리에 안 써서 그 칸이 놉니다.
    //   AV : B 워드 자체가 키 하나(레인 = head_dim). 키 n_tok 이면 **주소만**
    //        `OSTR` 이 가리키는 칸으로 돌립니다 (AV 는 행타일이 하나라 놉니다).
    //
    // K/V 영역은 블록 3개가 돌려 쓰므로 예약 칸을 그 안에 둘 수 없습니다 —
    // 블록마다 bias 값이 다릅니다. 그래서 별도 영역(BKV)에 두고 주소를 명령어가
    // 실어 옵니다.

    reg  [N*16-1:0]  bias_k_word;      // bias_k 워드 (레인 = head_dim)
    wire             use_bkv     = op_flag2[2]; // 현재 B에 bias token을 끼워넣으라는 flag
    wire             bkv_is_qk   = use_bkv && (op_fmt == FMT_Q69);
    wire             bkv_is_av   = use_bkv && (op_fmt != FMT_Q69);
    wire [AW_A-1:0]  b_word_off  = gemm_b_rd_addr[AW_A-1:0] - op_bin[AW_A-1:0];

    // bias 토큰은 **마지막 키** 입니다. cross 는 Lk = n_tok+1, latent 은 96+1 로
    // 고정이라 `n_tok` 이 아니라 **그 명령어의 Lk** 로 잡아야 합니다.
    //   QK : 출력 열이 키라 Lk = op_nout      AV : reduce 가 키라 Lk = op_k
    wire [DIM_W-1:0] qk_key      = op_nout - 1'b1; // bias key의 Tile idx
    wire [DIM_W-1:0] av_key      = op_k    - 1'b1;
    wire [4:0]       qk_lane     = qk_key[4:0]; // bias key의 lane idx
    wire [4:0]       qk_dim      = b_word_off[4:0];   // reduce 인덱스 = head_dim
    wire             qk_hit      = bkv_is_qk && (b_word_off[AW_A-1:5] == qk_key[AW_A-1:5]); // 현재 Tile과 bias key의 Tile 일치 여부
    wire             av_hit      = bkv_is_av && (b_word_off == av_key[AW_A-1:0]);

    // [함정] QK 의 B 워드는 레인 = 키입니다. 마지막 타일에서 `Lk` 를 넘는 레인은
    // `in_proj.K` 가 쓴 적이 없어 X 로 올라오고, softmax 가 전 키를 합산하므로
    // **출력 전체가 X** 가 됩니다. 그 레인들을 0 으로 막습니다.
    wire [DIM_W-1:0] b_key_tile = {{(DIM_W-(AW_A-5)){1'b0}}, b_word_off[AW_A-1:5]};
    wire [DIM_W-1:0] b_key_lim  = op_nout - (b_key_tile << 5);

    // A_Mem 은 주소를 준 **다음** 사이클에 답합니다. 데이터에 거는 조작(bias 레인
    // 치환, 남는 레인 0)은 주소 조건을 한 단 늦춰야 짝이 맞습니다.
    reg              qk_hit_d1, bkv_is_qk_d1;
    reg [4:0]        qk_lane_d1, qk_dim_d1;
    reg [DIM_W-1:0]  b_key_lim_d1;
    always @(posedge clk) begin
        qk_hit_d1    <= qk_hit;    bkv_is_qk_d1 <= bkv_is_qk;
        qk_lane_d1   <= qk_lane;   qk_dim_d1    <= qk_dim;
        b_key_lim_d1 <= b_key_lim;
    end

    wire [N*8-1:0] gemm_b_patched;
    generate
        for (lane = 0; lane < N; lane = lane + 1) begin : g_b_bias
            assign gemm_b_patched[lane*8 +: 8] =
                   (qk_hit_d1 && lane[4:0] == qk_lane_d1)   // bias_k 레인 치환
                       ? bias_k_word[{qk_dim_d1, 4'd0} +: 8]
                 : (bkv_is_qk_d1 && lane >= b_key_lim_d1)   // Lk 를 넘는 레인은 0
                       ? 8'd0
                       : gemm_b_from_a[lane*8 +: 8];
        end
    endgenerate
    wire [N*8-1:0] gemm_b_data = b_src_amem ? gemm_b_patched : gemm_b_from_w;

    // 코어는 B 주소를 하나만 냅니다. W_Mem 이 답할 때 그 주소를 그대로 씁니다.
    assign w_rd_en   = gemm_b_rd_en && !b_src_amem;
    assign w_rd_addr = gemm_b_rd_addr;

    Gemm_Core #(.N(N), .ACT_W(ACT_W), .PSUM_W(PSUM_W), .DIM_W(DIM_W),
                   .AW_A(AW_A), .AW_B(AW_W)) u_gemm (
        .clk(clk), .rst(rst), .start(gemm_start), .all_done(gemm_done),
        // W_Mem 에서 읽을 때만 DSP 패킹입니다. 단 **Transpose32 경유(FLAG[0])는
        // 제외**합니다 — 전치 버퍼는 32컬럼을 받은 뒤 "다음 타일을 계산하는 동안"
        // 32사이클에 걸쳐 쏟는 구조인데, 패킹하면 64컬럼이 연속으로 나와 쏟는
        // 도중에 덮어씁니다. 해당하는 것은 in_proj.V 뿐입니다.
        .pack(!b_src_amem && !op_flag[0]),
        .M(op_m), .K(op_k), .Nout(op_nout), .a_base(op_ain), .b_base(op_bin),
        .a_rd_en(gemm_a_rd_en), .a_rd_addr(gemm_a_rd_addr), .a_rd_data(a_ra_data),
        .b_rd_en(gemm_b_rd_en), .b_rd_addr(gemm_b_rd_addr), .b_rd_data(gemm_b_data),
        .col_valid(col_valid), .col_data(col_data),
        .col_n(col_n), .col_mt(col_mt));

    // 2차 재양자화 상수 — **명령어당 스칼라 하나**입니다 (활성함수 뒤 int8 격자,
    // MEAN 의 곱수). `RQ_BASE + NOUT` 칸을 ST_CONST 에서 한 번 읽어 떠 둡니다.
    (* max_fanout = 16 *)
    reg signed [31:0] rq_scale2_q;

    wire                fca_valid;
    wire [N*16-1:0]     fca_data, fca_q69;
    Format_Cast_Act #(.N(N), .ACT_W(ACT_W), .PSUM_W(PSUM_W)) u_fca (
        .clk(clk), .rst(rst), .fmt(op_fmt),
        // [타이밍] FCA 에는 **레지스터 전** 값을 넘깁니다. 8레인마다 자기 DSP
        // 근처에서 다시 뜨므로(`Format_Cast_Act.bias_l[]`) 지연은 동일하고,
        // `rq_*_q` 의 팬아웃에서 32레인 x2 계통이 통째로 빠집니다.
        .bias(rq_bias), .mult(rq_scale), .shift(op_shift),
        .g_mult(rq_scale2_q), .g_shift(op_shift2),
        .act_sel(op_act), .act_parm(8'd0), .raw16(op_flag[3]), .req2(op_flag2[3]),
        .in_valid(col_valid_d2), .acc(col_data_d2),
        .out_valid(fca_valid), .out_data(fca_data), .out_q69(fca_q69));

    // Requant_Mem 경로 지연(BRAM 1 + 레지스터 1)만큼 컬럼을 늦춰 상수와 정렬
    always @(posedge clk) begin
        col_valid_d1 <= col_valid;   col_valid_d2 <= col_valid_d1;
        col_data_d1  <= col_data;    col_data_d2  <= col_data_d1;
        col_n_d1     <= col_n;       col_n_d2     <= col_n_d1;
        col_mt_d1    <= col_mt;      col_mt_d2    <= col_mt_d1;
    end

    // Format_Cast_Act 지연만큼 쓰기 주소를 늦춥니다. **출력 포맷마다 다릅니다.**
    localparam FCA_LAT_BF16 = 2,   // bf16
               FCA_LAT_INT8 = 3,   // int8 (+ 활성함수)
               FCA_LAT_REQ2 = 6,   // int8 + 활성함수 뒤 2차 재양자화
               FCA_LAT_GELU = 9;   // Q4.11 → GELU → int8
    localparam FCA_LAT_MAX  = FCA_LAT_GELU;

    reg [DIM_W-1:0] fca_n_pipe  [0:FCA_LAT_MAX-1];
    reg [DIM_W-1:0] fca_mt_pipe [0:FCA_LAT_MAX-1];
    integer stage;
    always @(posedge clk) begin
        fca_n_pipe[0] <= col_n_d2;   fca_mt_pipe[0] <= col_mt_d2;
        for (stage = 1; stage < FCA_LAT_MAX; stage = stage + 1) begin
            fca_n_pipe [stage] <= fca_n_pipe [stage-1];
            fca_mt_pipe[stage] <= fca_mt_pipe[stage-1];
        end
    end

    // 네 후보 중 하나를 고르는 규칙이 아래 탭 넷에 공통이라 함수로 한 번만 씁니다.
    // **인덱스를 계산해서 넣지 않습니다** — 그러면 9:1 가변 선택이 되고, 이 먹스는
    // 아래 주소 계산의 최악 경로 위에 있습니다. 상수 인덱스 넷을 그대로 먹싱합니다.
    function [DIM_W-1:0] fca_tap;
        input [DIM_W-1:0] t_q411, t_bf16, t_req2, t_int8;
        begin
            fca_tap = (op_fmt == FMT_Q411) ? t_q411
                    : (op_fmt == FMT_BF16) ? t_bf16
                    : op_flag2[3]          ? t_req2
                    :                        t_int8;
        end
    endfunction

    // 지금 나온 컬럼의 (채널, 행타일) — 전치 드레인이 이 시점 값을 잡습니다
    wire [DIM_W-1:0] fca_n  = fca_tap(fca_n_pipe [FCA_LAT_GELU-1], fca_n_pipe [FCA_LAT_BF16-1],
                                      fca_n_pipe [FCA_LAT_REQ2-1], fca_n_pipe [FCA_LAT_INT8-1]);
    wire [DIM_W-1:0] fca_mt = fca_tap(fca_mt_pipe[FCA_LAT_GELU-1], fca_mt_pipe[FCA_LAT_BF16-1],
                                      fca_mt_pipe[FCA_LAT_REQ2-1], fca_mt_pipe[FCA_LAT_INT8-1]);

    // [타이밍] 위 4:1 먹스 + 주소 곱셈(DSP 2개 캐스케이드) + A_Mem 주소 디코드가
    // 한 사이클에 붙어 187.5 MHz 의 최악 경로였습니다 (5.578 ns). 레지스터 한 단을
    // 넣되 탭을 한 칸 당겨(`-2`) 상쇄하므로 **주소가 유효한 사이클은 그대로**이고
    // 총 사이클도 안 변합니다. 네 경로 모두 FCA_LAT >= 2 라 당길 여유가 있습니다.
    wire [DIM_W-1:0] fca_n_early  = fca_tap(fca_n_pipe [FCA_LAT_GELU-2], fca_n_pipe [FCA_LAT_BF16-2],
                                            fca_n_pipe [FCA_LAT_REQ2-2], fca_n_pipe [FCA_LAT_INT8-2]);
    wire [DIM_W-1:0] fca_mt_early = fca_tap(fca_mt_pipe[FCA_LAT_GELU-2], fca_mt_pipe[FCA_LAT_BF16-2],
                                            fca_mt_pipe[FCA_LAT_REQ2-2], fca_mt_pipe[FCA_LAT_INT8-2]);

    //   FLAG[1] head-major : AOUT + (n/32)*OSTR + mt*32 + n%32
    //   그 외              : AOUT + mt*OSTR + n
    reg [AW_A-1:0] gemm_we_addr_q;
    always @(posedge clk)
        gemm_we_addr_q <= op_flag[1]
            ? (op_aout + (fca_n_early[DIM_W-1:5] * op_ostr[AW_A-1:0])
                       + (fca_mt_early << 5) + fca_n_early[4:0])
            : (op_aout +  fca_mt_early * op_ostr + fca_n_early[AW_A-1:0]);

    // [함정] `gemm_done` 은 시스톨릭 배열이 다 쏟은 시점이라, 뒤에 붙은
    // requant/GELU 단(최대 9)이 아직 값을 들고 있을 수 있습니다. 그 상태로
    // 명령어를 넘기면 마지막 컬럼들이 **다음 명령어의 AOUT/OSTR** 로 나갑니다.
    reg [FCA_LAT_MAX-1:0] fca_pipe;
    always @(posedge clk) begin
        if (rst) fca_pipe <= {FCA_LAT_MAX{1'b0}};
        else     fca_pipe <= {fca_pipe[FCA_LAT_MAX-2:0], col_valid_d2};
    end
    wire fca_busy = col_valid_d2 | (|fca_pipe);

    // =========================================================================
    // V 전치 : in_proj 의 V 컬럼을 모아 축을 돌립니다
    // =========================================================================
    wire [N*8-1:0] tr_in, tr_out;
    // `tr_in` 이 조합(fca_data) 이므로 쓰기 인에이블/주소도 **조합**이어야 합니다.
    // 레지스터로 한 단 늦추면 데이터만 한 칸 밀려 전치가 통째로 어긋납니다.
    wire [4:0]     tr_widx;
    wire           tr_we;
    wire [4:0]     tr_ridx;
    generate
        for (lane = 0; lane < N; lane = lane + 1) begin : g_tr_narrow
            assign tr_in[lane*8 +: 8] = fca_data[lane*16 +: 8];
        end
    endgenerate
    Transpose32 #(.N(N), .W(8)) u_tr (
        .clk(clk), .rst(rst), .we(tr_we), .w_idx(tr_widx), .w_data(tr_in),
        .r_idx(tr_ridx), .r_data(tr_out));

    // ---- 채우기(write) → 쏟기(drain) ----
    // `attn·V` 의 reduce 축은 **키**입니다. 그런데 in_proj 이 내는 컬럼은
    // (레인 = 토큰, 워드 = head_dim) 이라 축이 반대입니다. head 하나의 32컬럼
    // (d = 0..31)을 다 받으면 32x32 블록이 차고, 그걸 행(키)별로 32번 읽어
    // `V[h*OSTR + mt*32 + key]` 에 씁니다 — 레인이 head_dim 이 됩니다.
    //
    // 코어가 타일 하나(32컬럼)를 내고 다음 타일을 K 사이클 계산하는 동안이
    // **쏟을 틈**입니다 (K >= 32 이라 항상 충분). 그래서 멈춤이 없습니다.
    //
    // 제어는 상태기계가 아니라 **한 줄짜리 지연 파이프**입니다 :
    //
    //   tr_we ─(32컬럼째)─→ tr_last ─d1─→ tr_last_d1 ─d1─→ tr_last_d2
    //                                                          └→ tr_drain (32사이클)
    //
    // 마지막 컬럼(d=31)의 쓰기가 **반영된 뒤** 읽어야 하므로 두 단 늦춥니다.
    wire             tr_last = tr_we && (fca_n[4:0] == 5'd31);  // 버퍼가 다 참
    reg              tr_last_d1, tr_last_d2;                    // 지연 2단
    reg              tr_drain;                                  // 쏟는 중 (32사이클)
    reg [5:0]        tr_row;                                    // 쏟는 중인 행(키)
    reg [1:0]        tr_head;                                   // 어느 head 로
    reg [DIM_W-1:0]  tr_mt;                                     // 어느 행타일로
    wire [1:0]       fca_head = fca_n[6:5];                     // head = 컬럼 / 32

    assign tr_we   = (op_kind == OP_GEMM) && op_flag[0] && fca_valid;
    assign tr_widx = fca_n[4:0];        // 채우기 : 워드 = head_dim
    assign tr_ridx = tr_row[4:0];       // 쏟기   : 워드 = 키

    always @(posedge clk) begin
        if (rst) begin
            tr_last_d1 <= 1'b0;  tr_last_d2 <= 1'b0;
            tr_drain   <= 1'b0;  tr_row     <= 0;
        end else begin
            tr_last_d1 <= tr_last;
            tr_last_d2 <= tr_last_d1;
            // head/타일은 **그 컬럼이 나온 순간** 잡아야 합니다 (다음 사이클엔
            // fca_n 이 이미 다음 타일 값입니다)
            if (tr_last) begin tr_head <= fca_head;  tr_mt <= fca_mt; end

            if (tr_last_d2) begin                    // 쏟기 시작
                tr_drain <= 1'b1;  tr_row <= 0;
            end else if (tr_drain) begin             // 행 32개를 하나씩
                if (tr_row == 6'd31) tr_drain <= 1'b0;
                tr_row <= tr_row + 1'b1;
            end
        end
    end

    // =========================================================================
    // positional encoding — 온칩 표에서 모아 PIN 뒤쪽에 씁니다
    //
    // 표(27.6 KB)는 PL BRAM 에 있고 타임스텝마다 오는 것은 `pos_idx`(최대 246 B)
    // 뿐입니다. 자세한 것은 `rtl/core/Pos_Gather.v` 머리말.
    // =========================================================================
    reg              pos_start;
    wire             pos_done, pos_rd_en, pos_we_en;
    wire [AW_A-1:0]  pos_rd_addr, pos_we_addr;
    wire [N*16-1:0]  pos_we_data;

    Pos_Gather #(.N(N), .FEAT(64), .AW_A(AW_A), .AW_T(9), .DIM_W(DIM_W)) u_pos (
        .clk(clk), .rst(rst), .start(pos_start), .done(pos_done),
        .n_tok(op_m), .a_base(op_aout), .ostr(op_ostr), .idx_base(op_ain),
        .ld_we(ld_we && ld_sel == LD_POS), .ld_addr(ld_addr[8:0]),
        .ld_data(ld_data[64*8-1:0]),
        .rd_en(pos_rd_en), .rd_addr(pos_rd_addr), .rd_data(a_ra_data),
        .we_en(pos_we_en), .we_addr(pos_we_addr), .we_data(pos_we_data));

    // =========================================================================
    // LayerNorm
    // =========================================================================
    reg              ln_start;
    wire             ln_done, ln_valid;
    wire [DIM_W-1:0] ln_k, ln_af_addr;
    wire [5:0]       ln_mt;
    wire [N*8-1:0]   ln_out;
    wire             ln_rd_en;
    wire [AW_A-1:0]  ln_rd_addr;

    // 입력은 bf16 이거나 Q4.11 정수 코드입니다 (`layer_norm_2` 만 앞이 `linear1` 의
    // raw16 출력이라 정수). 엔진은 A_Mem 워드를 **그대로** 넘기고 포맷 비트만
    // 알려 줍니다 — 코어가 `Bf16_To_Fix` / `Q411_To_Fix` 를 병렬로 두고 고릅니다.
    // 행타일 반복은 **래퍼가** 합니다 (코어의 3단 Tile 파이프라인이 겹치려면
    // 타일을 끊지 않고 연달아 밀어야 하므로 엔진에 `ln_mt` 루프가 없습니다).
    LayerNorm_Top #(.N(N), .E(E), .DIM_W(DIM_W), .AW(AW_A), .XSW(6)) u_ln (
        .clk(clk), .rst(rst), .start(ln_start), .done(ln_done),
        .M(op_m), .a_base(op_ain), .in_shift($signed(op_shift2)),
        .rd_en(ln_rd_en), .rd_addr(ln_rd_addr), .rd_data(a_ra_data),
        .in_q411(op_flag2[0]),
        .af_addr(ln_af_addr), .af_gamma(af_gamma), .af_beta(af_beta),
        .mult(rq_scale_q), .shift(op_shift),
        .out_valid(ln_valid), .out_mt(ln_mt), .out_k(ln_k), .out_data(ln_out));

    // gamma/beta(Affine_Mem)와 재양자화 스칼라(Requant_Mem)는 **다른 메모리**라
    // 베이스도 둘입니다 — 전자는 `op_ostr`, 후자는 `op_rq_base` 가 실어 옵니다.
    assign af_idx = ostr_as_af_base + ln_af_addr[AW_AF+2:0];

    // =========================================================================
    // Softmax (attention)
    // =========================================================================
    reg             smax_start;
    wire            smax_done, smax_valid, smax_last;
    wire [7:0]      smax_col;
    wire [N*8-1:0]  smax_out;

    // QK 는 행타일 `⌈M/32⌉` 개를 한 명령어로 돕니다. 나눗셈이 아니라 비트 슬라이스
    // 입니다 (M 은 32 의 배수). OP_SMAX 단독 경로는 예전처럼 타일 1개입니다.
    wire [5:0] smax_ntile = (op_fmt == FMT_Q69)
                          ? (op_m[10:5] + {5'd0, |op_m[4:0]}) : 6'd1;

    Softmax_Top #(.N(N), .CMAX(TOKMAX+1)) u_smax (
        .clk(clk), .rst(rst), .start(smax_start), .n_col(op_nout[7:0]),
        .n_tile(smax_ntile), .out_last(smax_last), .done(smax_done),
        .in_valid(smax_in_valid), .in_data(smax_in_data),
        .out_valid(smax_valid), .out_n(smax_col), .out_data(smax_out));

    // =========================================================================
    // softmax 출력의 행타일 베이스 — **곱셈 없이 누적**합니다
    //
    // `OSTR + smax_mt*SMSTR + col` 을 그대로 쓰면 곱셈기가 A_Mem 주소 경로에 붙고,
    // CLB 가 95 % 인 상황에서 그 LUT/캐리 사슬을 놓을 자리가 없습니다. 행타일은
    // 순서대로만 진행하고 증분이 정확히 SMSTR 이라 **덧셈 하나**로 끝납니다
    // (`Gemm_Core` 의 `a_tile_base`, RES 의 `rs_base_*` 와 같은 처방).
    // =========================================================================
    localparam [AW_A-1:0] SMSTR = TOKMAX + 1;      // SM 의 행타일 간 간격
    reg [AW_A-1:0] smax_wr_base;
    always @(posedge clk) begin
        if (!smax_start)                   smax_wr_base <= ostr_as_smax_base;
        else if (smax_valid && smax_last)  smax_wr_base <= smax_wr_base + SMSTR;
    end


    // QK GEMM 의 Q6.9 컬럼을 메모리를 안 거치고 softmax 로 직결합니다.
    // [함정] `smax_start` 로 한 번 더 막습니다 — `op_fmt` 는 ST_DECODE 에서 바뀌는데
    // 직전 GEMM 의 Format_Cast_Act 파이프가 아직 비워지는 중이라, 그 잔여 컬럼이
    // 새 명령어의 softmax 로 새어 들어가 **1열짜리 타일**을 만듭니다. 그러면 코어의
    // 뱅크 포인터가 어긋나 이후 타일이 이전 길이로 나옵니다.
    always @(posedge clk) begin
        smax_in_valid <= fca_valid && (op_fmt == FMT_Q69) && smax_start;
        smax_in_data <= fca_q69;
    end

    // =========================================================================
    // RES : 잔차 덧셈 — `AOUT[x] = AIN[x] + AOUT[x]`
    //
    // A_Mem 두 워드를 동시에 읽어(미러 2벌) 더하고 제자리에 씁니다.
    //
    // ## 5단 파이프라인 — 워드당 1사이클
    //
    //   S0  주소 발행                              rs_addr_a / rs_addr_b
    //   S1  A_Mem 두 워드 도착 → 래치              rs_a / rs_b
    //   S2  덧셈 투입      bf16 : Fp32_Add 1단  |  정수 : 곱·덧셈 → res_int_acc_q
    //   S3                 bf16 : Fp32_Add 2단  |  정수 : Int32_To_Bf16 → res_int_bf_q
    //   S4  결과 확정 → A_Mem 쓰기                 rs_sum
    //
    // 지연은 5사이클이지만 **매 사이클 새 워드를 발행**하므로 처리율은 1/사이클
    // 입니다. 두 경로 모두 단수가 같도록 맞춰 두어 제어가 하나면 됩니다.
    //
    // 워드끼리는 서로 독립이라(x 마다 읽기 1회·쓰기 1회) 겹쳐 흘려도 안전합니다.
    // 읽는 주소는 k, 쓰는 주소는 k-4 라 같은 사이클에 같은 칸을 건드리지 않습니다.
    // =========================================================================
    reg              rs_run;                 // 발행 중 (마지막 워드를 내면 내려감)
    reg [DIM_W-1:0]  rs_k;                   // 발행 중인 특징
    reg [4:1]        rs_vld;                 // 단별 유효 — [1] 수신 … [4] 쓰기
    reg [N*16-1:0]   rs_a, rs_b;             // S1 래치 (포트 A / 포트 B)
    wire [N*16-1:0]  rs_sum;

    // [타이밍] 행타일 베이스를 **레지스터로 미리** 들고 있습니다. 매 사이클 경로에
    // 남는 것은 덧셈 하나뿐입니다.
    //
    // `base + rs_mt*K + rs_k` 를 그대로 쓰면 `rs_mt` 곱셈(DSP 5단) → 덧셈 →
    // A_Mem 깊이 캐스케이드의 ENBWREN 디코드가 한 사이클에 들어갑니다. 워드당
    // 5사이클일 때는 `rs_k` 가 5사이클에 한 번만 바뀌어 여유가 있었지만, 매
    // 사이클 발행하면 이 경로가 최악이 됩니다 (실측 WNS -0.171 ns).
    //
    // 베이스는 행타일이 바뀔 때만 갱신되고 증분이 정확히 K 라, 곱셈 없이
    // **누적**으로 끝납니다 (`Gemm_Core` 의 `a_tile_base` 와 같은 처방).
    reg  [AW_A-1:0]  rs_base_a, rs_base_b;
    wire [AW_A-1:0]  rs_addr_a = rs_base_a + rs_k[AW_A-1:0];
    wire [AW_A-1:0]  rs_addr_b = rs_base_b + rs_k[AW_A-1:0];
    reg  [AW_A-1:0]  rs_wr_pipe [0:3];       // [0] 방금 발행 … [3] 4사이클 전
    integer rs_stage;
    always @(posedge clk) begin
        rs_wr_pipe[0] <= rs_addr_b;
        for (rs_stage = 1; rs_stage < 4; rs_stage = rs_stage + 1)
            rs_wr_pipe[rs_stage] <= rs_wr_pipe[rs_stage-1];
    end
    // 보통은 bf16 두 값을 더하지만(`res_is_int`=0), `proc_events` 의
    // `x = seq_init(x) + x_input` 한 자리만 **두 피연산자가 정수 코드이고 scale 이
    // 서로 다릅니다** (0.019991 vs 0.038420).
    //
    // 그 자리를 bf16 상수 곱으로 처리하면 상수가 가수 8비트로 반올림돼 비율이
    // 0.4 % 흔들리는데, 이는 bf16 한 칸과 같은 크기라 결과의 절반이 어긋납니다.
    // 대신 scale 을 `round(scale * 2^RSH)` 정수로 두고 정수로 더한 뒤 **한 번만**
    // bf16 으로 내립니다 — 반올림이 골든과 같은 자리에서 한 번만 일어납니다.
    // (지수에서 RSH 를 빼는 것은 2의 거듭제곱이라 가수를 안 건드립니다.
    //  scale 을 1 근처로 되돌리는 것은 LayerNorm 의 `eps` 때문입니다 — 임의
    //  scale 이면 분산에 더하는 eps 의 상대 크기가 달라집니다.)
    localparam RSH = 20;                       // 정수 scale 상수의 소수 비트
    wire res_is_int = op_flag2[1];
    generate
        for (lane = 0; lane < N; lane = lane + 1) begin : g_res_lane
            wire signed [15:0] res_a_lane = rs_a[lane*16 +: 16];   // 포트 A (ph1 래치)
            wire signed [15:0] res_b_lane = rs_b[lane*16 +: 16];   // 포트 B (ph1 래치)

            // ---- 정수 경로 (res_is_int=1) ----
            // 두 피연산자의 정수 scale 은 RQ_BASE / OSTR 칸에 실려 옵니다
            wire signed [31:0] res_int_acc =
                     $signed(res_a_lane) * $signed({1'b0, rq_base_as_scale_a})
                   + $signed(res_b_lane) * $signed({1'b0, ostr_as_scale_b});
            // [타이밍] 곱·덧셈 **뒤에서** 끊습니다. 뒤쪽에서 늦추면
            // `곱셈2+덧셈 → 32b LZC·정규화 → 지수` 가 한 사이클에 들어갑니다.
            // 단수(2단)와 쓰기 시점은 그대로라 `Fp32_Add`(2단)와의 정렬이 안 깨집니다.
            reg signed [31:0] res_int_acc_q;
            always @(posedge clk) res_int_acc_q <= res_int_acc;

            // 정수 합을 bf16 으로 내리면서 지수에서 RSH 를 뺍니다 (2의 거듭제곱
            // 이라 가수는 안 건드립니다). 지수가 0 이면 그대로 0.
            wire [15:0] res_int_bf_raw;
            Int32_To_Bf16 #(.IN_W(32)) u_res_int_bf (
                .din(res_int_acc_q), .bf16(res_int_bf_raw));
            wire [15:0] res_int_bf =
                   (res_int_bf_raw[14:7] == 8'd0) ? 16'd0
                 : {res_int_bf_raw[15], res_int_bf_raw[14:7] - RSH[7:0],
                    res_int_bf_raw[6:0]};

            // ---- bf16 경로 (res_is_int=0) ----
            // [타이밍] 두 피연산자 모두 ph1 에 래치한 레지스터를 씁니다. BRAM 출력을
            // 직결하면 `BRAM → FP 정렬 배럴시프터 → sum1` 이 한 사이클입니다.
            wire [31:0] res_fp_sum;
            wire [15:0] res_fp_bf;
            Fp32_Add u_res_fp_add (
                .clk(clk), .rst(rst),
                .in_valid(rs_vld[2] && !res_is_int),
                .a({res_a_lane, 16'd0}), .b({res_b_lane, 16'd0}),
                .out_valid(), .y(res_fp_sum));
            Fp32_To_Bf16 u_res_fp_bf (.f(res_fp_sum), .log2e(6'd0), .y(res_fp_bf));

            // 정수 경로도 2단(S2 곱·덧셈 / S3 정규화)이라 bf16 경로(Fp32_Add 2단)와
            // S4 에서 나란히 도착합니다 — 그래서 제어 루프가 하나면 됩니다.
            reg [15:0] res_int_bf_q;
            always @(posedge clk) res_int_bf_q <= res_int_bf;

            assign rs_sum[lane*16 +: 16] = res_is_int ? res_int_bf_q : res_fp_bf;
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
    reg              mean_run;
    reg  [DIM_W-1:0] mean_k;      // 특징
    reg  [5:0]       mean_mt;     // 행타일
    reg  [2:0]       mean_ph;     // 위상 0~4
    reg signed [15:0] mean_acc;
    // [타이밍] BRAM 출력을 레지스터로 받은 뒤 더합니다 — 직결하면
    // `BRAM → 32입력 덧셈트리(CARRY8 5개) → mean_acc` 가 한 사이클입니다.
    // MEAN 은 샘플당 1회라 위상이 하나 늘어도 +384 사이클(0.024 %) 뿐입니다.
    reg [N*16-1:0]    mean_rd_q;
    always @(posedge clk) mean_rd_q <= a_ra_data;
    integer sum_lane;
    reg signed [15:0] mean_lane_sum;
    always @* begin                                   // 32레인 합 (유효 행만)
        mean_lane_sum = 16'sd0;
        for (sum_lane = 0; sum_lane < N; sum_lane = sum_lane + 1)
            if (mean_mt * N + sum_lane < op_m)
                mean_lane_sum = mean_lane_sum + $signed(mean_rd_q[sum_lane*16 +: 16]);
    end
    wire signed [47:0] mean_prod = $signed(mean_acc) * $signed(rq_scale2_q);
    // [타이밍] 곱 뒤에서 끊습니다 (DSP48E2 의 P 레지스터로 흡수). 직결하면
    // `곱 → 48b 반올림 가산 → 가변 시프트 → 포화 → A_Mem 쓰기` 가 한 사이클입니다.
    // 위상이 하나 늘지만 전체 165만 사이클에서 128 사이클(0.008 %) 뿐입니다.
    reg  signed [47:0] mean_prod_q;
    always @(posedge clk) mean_prod_q <= mean_prod;
    wire signed [47:0] mean_rnd     = mean_prod_q + (48'sd1 <<< (op_shift2 - 1));
    wire signed [47:0] mean_shifted = mean_rnd >>> op_shift2;
    wire signed [ 7:0] mean_out     = (mean_shifted >  127) ?  8'sd127
                                    : (mean_shifted < -128) ? -8'sd128
                                    :                          mean_shifted[7:0];

    // =========================================================================
    // ARGMAX : 마지막 GEMM 의 컬럼을 메모리에 안 쓰고 최대값만 고릅니다
    //   골든 note : argmax(acc[c]*M[c]) — shift 는 순서를 안 바꿔 생략 가능
    // =========================================================================
    wire signed [PSUM_W-1:0] argmax_acc = $signed(col_data_d2[PSUM_W-1:0])
                                        + $signed(rq_bias_q);
    wire signed [63:0]       argmax_val = argmax_acc * $signed(rq_scale_q);
    // [타이밍] 곱셈 뒤에서 끊습니다 — 직결하면 `bias 덧셈 → DSP 2개 캐스케이드 →
    // 64비트 비교 → argmax_best 의 CE` 가 한 사이클입니다. ARGMAX 는 샘플당 한
    // 번뿐이라 한 사이클 늘어도 전체에 영향이 없습니다.
    reg  signed [63:0]       argmax_val_q;
    reg  signed [PSUM_W-1:0] argmax_acc_q;
    reg  [DIM_W-1:0]         argmax_n_q;
    reg                      argmax_valid_q;
    always @(posedge clk) begin
        if (rst) argmax_valid_q <= 1'b0;
        else begin
            argmax_val_q   <= argmax_val;
            argmax_acc_q   <= argmax_acc;
            argmax_n_q     <= col_n_d2;
            // 코어가 마지막 타일을 비우며 유효 범위 밖 컬럼을 더 낼 수 있어
            // `col_n_d2 < op_nout` 로 막습니다 (안 막으면 클래스 10 이 나옵니다).
            argmax_valid_q <= (op_kind == OP_ARGMAX) && col_valid_d2
                              && (col_n_d2 < op_nout);
        end
    end
    reg  signed [63:0]       argmax_best;
    reg                      argmax_any;

    // =========================================================================
    // A_Mem 포트 중재 + 쓰기
    // =========================================================================
    // int8 결과(레인당 8비트)를 A_Mem 워드(레인당 16비트)로 부호확장해 담습니다.
    integer wr_lane;
    always @* begin
        // ---- 기본값 : 아무도 안 쓰면 전부 잠급니다 ----
        a_ra_en   = 1'b0;  a_ra_addr = {AW_A{1'b0}};
        a_rb_en   = 1'b0;  a_rb_addr = {AW_A{1'b0}};
        a_we_en   = 1'b0;  a_we_addr = {AW_A{1'b0}};  a_we_data = {N*16{1'b0}};

        if (state == ST_IDLE) begin
            // ---- 쉬는 동안에만 호스트 리드백을 붙입니다 ----
            a_ra_en   = dbg_rd_en;
            a_ra_addr = dbg_rd_addr;

        end else if (state == ST_CONST) begin
            // ---- QK 명령어 시작에 bias_k 워드를 한 번 읽어 둡니다 ----
            a_ra_en   = 1'b1;
            a_ra_addr = op_aout[AW_A-1:0];

        end else case (op_kind)

            // ---------------- GEMM / ARGMAX ----------------
            OP_ARGMAX,
            OP_GEMM: begin
                a_ra_en   = gemm_a_rd_en;
                a_ra_addr = gemm_a_rd_addr;
                if (b_src_amem) begin                     // B 도 A_Mem 에서
                    a_rb_en   = gemm_b_rd_en;
                    a_rb_addr = av_hit ? ostr_as_bkv_addr // bias_v 칸으로 돌림
                                       : gemm_b_rd_addr[AW_A-1:0];
                end

                if (tr_drain) begin
                    // 전치 드레인 : 워드 = head*OSTR + mt*32 + 키
                    // [함정] `tr_row` 은 6비트입니다. `tr_row[AW_A-1:0]` 처럼
                    // 폭을 넘겨 잘라 쓰면 Verilog 가 **조용히 X** 를 줍니다.
                    a_we_en   = 1'b1;
                    a_we_addr = op_aout + tr_head * op_ostr[AW_A-1:0]
                              + (tr_mt << 5) + {{(AW_A-6){1'b0}}, tr_row};
                    for (wr_lane = 0; wr_lane < N; wr_lane = wr_lane + 1)
                        a_we_data[wr_lane*16 +: 16] =
                            {{8{tr_out[wr_lane*8+7]}}, tr_out[wr_lane*8 +: 8]};

                end else if (smax_valid && op_fmt == FMT_Q69) begin
                    // QK 직결 softmax 의 출력 : 워드 = 키, 레인 = latent 행
                    // 행타일마다 SMSTR 만큼 떨어진 자리에 씁니다 (베이스는 누적).
                    a_we_en   = 1'b1;
                    a_we_addr = smax_wr_base + {{(AW_A-8){1'b0}}, smax_col};
                    for (wr_lane = 0; wr_lane < N; wr_lane = wr_lane + 1)
                        a_we_data[wr_lane*16 +: 16] = {8'd0, smax_out[wr_lane*8 +: 8]};

                end else if (fca_valid && op_fmt != FMT_Q69
                             && op_kind == OP_GEMM && !op_flag[0]) begin
                    // 보통의 컬럼 되쓰기 (주소는 한 사이클 앞서 계산해 둔 것)
                    a_we_en   = 1'b1;
                    a_we_addr = gemm_we_addr_q;
                    a_we_data = fca_data;
                end
            end

            // ---------------- LayerNorm ----------------
            OP_LN: begin
                a_ra_en   = ln_rd_en;
                a_ra_addr = ln_rd_addr;
                if (ln_valid) begin
                    a_we_en   = 1'b1;
                    a_we_addr = op_aout + ln_mt * op_nout[AW_A-1:0] + ln_k[AW_A-1:0];
                    for (wr_lane = 0; wr_lane < N; wr_lane = wr_lane + 1)
                        a_we_data[wr_lane*16 +: 16] =
                            {{8{ln_out[wr_lane*8+7]}}, ln_out[wr_lane*8 +: 8]};
                end
            end

            // ---------------- Softmax (단독) ----------------
            OP_SMAX: begin
                if (smax_valid) begin
                    a_we_en   = 1'b1;
                    a_we_addr = op_aout + {{(AW_A-8){1'b0}}, smax_col};
                    for (wr_lane = 0; wr_lane < N; wr_lane = wr_lane + 1)
                        a_we_data[wr_lane*16 +: 16] = {8'd0, smax_out[wr_lane*8 +: 8]};
                end
            end

            // ---------------- RES (잔차 덧셈) ----------------
            // 매 사이클 두 워드를 읽으면서(S0) 4사이클 전 워드를 씁니다(S4).
            // 읽기는 미러 2벌이라 두 포트가 독립이고, 쓰기는 두 벌에 같이 갑니다.
            OP_RES: begin
                a_ra_en   = rs_run;   a_ra_addr = rs_addr_a;
                a_rb_en   = rs_run;   a_rb_addr = rs_addr_b;
                if (rs_vld[4]) begin
                    a_we_en   = 1'b1;
                    a_we_addr = rs_wr_pipe[3];     // 발행한 주소를 그대로 따라옴
                    a_we_data = rs_sum;
                end
            end

            // ---------------- positional encoding ----------------
            OP_POS: begin
                a_ra_en   = pos_rd_en;   a_ra_addr = pos_rd_addr;
                a_we_en   = pos_we_en;   a_we_addr = pos_we_addr;
                a_we_data = pos_we_data;
            end

            // ---------------- MEAN (레인 축 리덕션) ----------------
            OP_MEAN: begin
                a_ra_en   = mean_run;
                a_ra_addr = op_ain + mean_mt * op_k[AW_A-1:0] + mean_k[AW_A-1:0];
                if (mean_run && mean_ph == 3'd4) begin
                    a_we_en   = 1'b1;
                    a_we_addr = op_aout + mean_k[AW_A-1:0];   // 레인 0 에만
                    a_we_data = {{(N-1)*16{1'b0}}, {{8{mean_out[7]}}, mean_out}};
                end
            end

            default: ;
        endcase
    end

    // -------------------------------------------------------------------------
    // Requant_Mem 인덱스 — 세 가지뿐입니다
    //
    //   ST_CONST      RQ_BASE + NOUT   GELU 뒤 2차 곱수 (채널 테이블 바로 뒤 칸)
    //   채널별        RQ_BASE + n      GEMM · ARGMAX 의 보통 경우
    //   블록 스칼라   RQ_BASE          attention 의 QK/AV (FLAG2[2] 가 선 명령어)
    //
    // ARGMAX 도 **채널별**입니다 — 골든이 `argmax(acc[c]*M[c])` 이라 클래스마다
    // 곱수가 다릅니다. `OP_GEMM` 만 걸어 두면 10클래스가 전부 채널 0 의 곱수를
    // 써서 사실상 `argmax(acc[c])` 가 됩니다.
    // -------------------------------------------------------------------------
    wire rq_per_channel = (op_kind == OP_GEMM || op_kind == OP_ARGMAX) && !op_flag2[2];
    always @* begin
        if      (state == ST_CONST) rq_idx = op_rq_base[AW_RQ+1:0] + op_nout[AW_RQ+1:0];
        else if (rq_per_channel)    rq_idx = op_rq_base[AW_RQ+1:0] + col_n[AW_RQ+1:0];
        else                        rq_idx = op_rq_base[AW_RQ+1:0];
    end

    // =========================================================================
    // 컨트롤
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            // 시퀀서
            state      <= ST_IDLE;  inst_ptr <= 0;      inst_addr <= 0;
            tstep      <= 0;        in_tail  <= 1'b0;   n_tok     <= 0;
            done       <= 1'b0;     tok_req  <= 1'b0;   wait_ack  <= 1'b0;
            const_ph   <= 0;        op_flag2 <= 4'd0;
            // 유닛 기동
            gemm_start <= 1'b0;  ln_start <= 1'b0;  smax_start <= 1'b0;  pos_start <= 1'b0;
            // RES / MEAN 루프
            rs_run     <= 1'b0;  rs_k     <= 0;  rs_vld  <= 4'd0;  rs_mt  <= 6'd0;
            rs_base_a  <= {AW_A{1'b0}};  rs_base_b <= {AW_A{1'b0}};
            mean_run   <= 1'b0;  mean_k   <= 0;  mean_mt <= 0;  mean_ph  <= 0;  mean_acc <= 0;
            // 상수 / 결과
            rq_scale2_q  <= 0;    bias_k_word <= 0;
            argmax_best <= 0;    argmax_any  <= 1'b0;
            res_class   <= 4'd0; res_logits  <= 0;
        end else begin
            if (argmax_valid_q) begin
                res_logits[argmax_n_q[3:0]*PSUM_W +: PSUM_W] <= argmax_acc_q;
                if (!argmax_any || argmax_val_q > argmax_best) begin
                    argmax_best <= argmax_val_q;
                    argmax_any  <= 1'b1;
                    res_class   <= argmax_n_q[3:0];
                end
            end
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        tstep <= 0; inst_ptr <= 0; in_tail <= 1'b0;
                        inst_addr <= 0; state <= ST_TLOAD;
                    end
                end
                // 이 타임스텝의 X/PIN 이 채워지기를 기다립니다
                ST_TLOAD: begin
                    tok_req <= 1'b1;
                    if (tok_ack) begin              // 호스트 적재 완료
                        tok_req <= 1'b0;
                        n_tok   <= tok_n;
                        state   <= ST_FETCH;
                    end
                end
                // Inst_Mem 읽기 1사이클
                ST_FETCH: begin
                    n_tok <= tok_n;
                    state <= ST_DECODE;
                end
                ST_DECODE: begin
                    op_kind     <= inst_word[  3: 0];
                    op_fmt      <= inst_word[  5: 4];
                    op_act      <= inst_word[  7: 6];
                    op_flag     <= inst_word[ 15:12];
                    op_shift    <= inst_word[ 21:16];
                    op_shift2   <= inst_word[ 27:22];
                    op_flag2    <= inst_word[ 31:28];
                    op_ain      <= inst_word[128 +: AW_A];
                    op_bin      <= inst_word[160 +: AW_W];
                    op_aout     <= inst_word[192 +: AW_A];
                    op_rq_base  <= inst_word[224 +: 16];
                    op_ostr     <= inst_word[240 +: 16];
                    // VAR 비트가 선 필드는 상수 대신 이번 타임스텝의 토큰 수로
                    // 채웁니다 (부분선택을 두 번 이어 쓰면 문법 오류라 개별 비트).
                    op_m        <= inst_word[8]  ? tok_n          : inst_word[32 +: DIM_W];
                    op_k        <= inst_word[10] ? (tok_n + 1'b1) : inst_word[64 +: DIM_W];
                    op_nout     <= (inst_word[9] | inst_word[11]) ? (tok_n + 1'b1)
                                                                 : inst_word[96 +: DIM_W];
                    const_ph <= 0;
                    state    <= ST_CONST;
                end
                // Requant_Mem 읽기 2사이클 → GELU 뒤 재양자화 곱수 확정
                ST_CONST: begin
                    const_ph <= const_ph + 1'b1;
                    if (const_ph == 2'd1) bias_k_word <= a_ra_data;   // QK 용
                    if (const_ph == 2'd2) begin
                        rq_scale2_q <= rq_scale_q;                      // 2차 곱수
                        state      <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    case (op_kind)
                        OP_GEMM: begin
                            gemm_start <= 1'b1;
                            // Q6.9 는 컬럼이 softmax 로 직결이라 같이 기동합니다
                            if (op_fmt == FMT_Q69) smax_start <= 1'b1;
                        end
                        OP_LN:     ln_start   <= 1'b1;
                        OP_SMAX:   smax_start <= 1'b1;
                        OP_POS:    pos_start  <= 1'b1;
                        OP_RES: begin
                            rs_run   <= 1'b1;  rs_k   <= 0;  rs_vld <= 4'd0;  rs_mt <= 6'd0;
                            rs_base_a <= op_ain;   rs_base_b <= op_aout;   // 행타일 0
                        end
                        OP_MEAN: begin
                            mean_run <= 1'b1;  mean_k <= 0;  mean_mt <= 0;
                            mean_ph  <= 0;     mean_acc <= 0;
                        end
                        OP_ARGMAX: begin
                            gemm_start  <= 1'b1;
                            argmax_any  <= 1'b0;  argmax_best <= 0;  res_class <= 4'd0;
                        end
                        default: ;
                    endcase
                    wait_ack <= 1'b0;
                    state    <= ST_WAIT;
                end
                // ---------------------------------------------------------
                // 유닛이 끝나기를 기다립니다. RES / MEAN 은 하위 코어가 없어
                // 여기서 직접 위상을 돌립니다.
                // ---------------------------------------------------------
                ST_WAIT: begin
                    if (!gemm_done && !ln_done && !smax_done && !pos_done)
                        wait_ack <= 1'b1;

                    if (op_kind == OP_RES) begin
                        // 파이프 유효를 한 칸씩 밀고, S1 에서 두 워드를 래치합니다
                        rs_vld <= {rs_vld[3:1], rs_run};
                        if (rs_vld[1]) begin
                            rs_a <= a_ra_data;              // 포트 A
                            rs_b <= a_rb_data;              // 포트 B — 같은 사이클
                        end

                        // **매 사이클** 다음 워드로. 마지막을 발행하면 발행만 멈추고,
                        // 파이프에 남은 4개가 다 써질 때까지 기다립니다.
                        if (rs_run) begin
                            if (rs_k != op_k - 1)                rs_k  <= rs_k + 1'b1;
                            else begin
                                rs_k <= 0;
                                if (rs_mt != row_tile_last) begin
                                    rs_mt     <= rs_mt + 1'b1;
                                    rs_base_a <= rs_base_a + op_k[AW_A-1:0];  // 곱셈 대신 누적
                                    rs_base_b <= rs_base_b + op_k[AW_A-1:0];
                                end else                         rs_run <= 1'b0;
                            end
                        end else if (!(|rs_vld)) state <= ST_NEXT;

                    end else if (op_kind == OP_MEAN) begin
                        // 특징 하나당 : 타일마다 (주소 → 합), 그 뒤 재양자화 · 쓰기
                        case (mean_ph)
                            3'd0: mean_ph <= 3'd1;          // 주소 발행
                            3'd1: mean_ph <= 3'd2;          // BRAM 출력 수신
                            3'd2: begin                     // 레인 합 누산
                                mean_acc <= mean_acc + mean_lane_sum;
                                if (mean_mt == row_tile_last) mean_ph <= 3'd3;
                                else begin mean_mt <= mean_mt + 1'b1; mean_ph <= 3'd0; end
                            end
                            3'd3: mean_ph <= 3'd4;          // 곱이 mean_prod_q 에 잡힘
                            default: begin                  // ph4 : 재양자화 · 쓰기
                                mean_ph <= 3'd0;  mean_mt <= 0;  mean_acc <= 0;
                                if (mean_k != op_k - 1)     mean_k   <= mean_k + 1'b1;
                                else begin mean_run <= 1'b0; state   <= ST_NEXT; end
                            end
                        endcase

                    // [함정] 전치 드레인이 **다 쏟기 전에** 명령어를 넘기면 마지막
                    // 워드들이 다음 명령어의 AOUT/OSTR 로 나갑니다. Q6.9 는 softmax
                    // 까지가 같은 명령어라 `smax_done` 도 같이 봅니다.
                    end else if (op_kind == OP_GEMM && gemm_done && wait_ack
                                 && !tr_drain && !tr_last_d1 && !tr_last_d2   // 다 쏟았나
                                 && !fca_busy
                                 && (op_fmt != FMT_Q69 || smax_done)) begin
                        gemm_start <= 1'b0;  smax_start <= 1'b0;  state <= ST_NEXT;

                    // [함정] `argmax_valid_q` 가 아직 서 있으면 마지막 컬럼이
                    // 아직 반영 전입니다 (곱셈 뒤 레지스터로 한 단 늘었습니다).
                    end else if (op_kind == OP_ARGMAX && gemm_done && wait_ack
                                 && !argmax_valid_q) begin
                        gemm_start <= 1'b0;                       state <= ST_NEXT;

                    end else if (op_kind == OP_LN   && ln_done   && wait_ack) begin
                        ln_start   <= 1'b0;                       state <= ST_NEXT;
                    end else if (op_kind == OP_SMAX && smax_done && wait_ack) begin
                        smax_start <= 1'b0;                       state <= ST_NEXT;
                    end else if (op_kind == OP_POS  && pos_done  && wait_ack) begin
                        pos_start  <= 1'b0;                       state <= ST_NEXT;
                    end
                end
                // 다음 명령어로. body 끝이면 타임스텝을 넘기고, tail 끝이면 완료.
                ST_NEXT: begin
                    if (!gemm_done && !ln_done) begin
                        if      (!in_tail && inst_ptr == n_body - 1)
                            state <= ST_TSTEP;
                        else if ( in_tail && inst_ptr == n_body + n_tail - 1)
                            state <= ST_DONE;
                        else begin
                            inst_ptr  <= inst_ptr + 1'b1;
                            inst_addr <= inst_ptr + 1'b1;
                            state     <= ST_FETCH;
                        end
                    end
                end

                // 마지막 타임스텝이면 tail 로, 아니면 다음 타임스텝 입력을 기다립니다.
                ST_TSTEP: begin
                    if (tstep == n_tstep - 1) begin
                        in_tail   <= 1'b1;
                        inst_ptr  <= n_body;  inst_addr <= n_body;
                        state     <= ST_FETCH;
                    end else begin
                        tstep     <= tstep + 1'b1;
                        inst_ptr  <= 0;       inst_addr <= 0;
                        state     <= ST_TLOAD;
                    end
                end
                ST_DONE: begin
                    done <= 1'b1;
                    if (!start) state <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
