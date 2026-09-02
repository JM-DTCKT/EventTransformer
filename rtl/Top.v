// -----------------------------------------------------------------------------
// Top : 보드 최상위 — AXI4-Lite(제어) + AXI-Stream(데이터) 래퍼
//
//   PS ─HPM0─ AXI-Lite ──→ 제어 / 상태 / 타임스텝 핸드셰이크 / 결과
//   PS ─HP0── AXI DMA ─MM2S(128b)→ S_AXIS → Axis_Loader → W·A·RQ·AF·INST·POS
//                     ←S2MM(128b)─ M_AXIS ← Axis_Dump  ← A_Mem
//
// Vivado BD 는 이 모듈을 `evt_accel_0` 셀로 인스턴스합니다 (`tcl/build.tcl`).
//
// ## PS 가 해야 하는 일
//
//  · **명령어 프로그램도 DMA 로 넣습니다** (LOAD_SEL 4). 123 명령어 x 8 워드라
//    AXI-Lite 레지스터로 쓰면 976번 써야 합니다.
//  · **타임스텝마다 입력을 새로 받습니다.** X/PIN 은 A_Mem 에 한 타임스텝분만
//    들어갑니다 (20벌이면 24k 워드로 A_Mem 을 넘습니다). 엔진이 `tok_req` 로
//    멈추면 PS 가 그 타임스텝의 X/pos enc 를 DMA 하고 `TOK_ACK` 를 씁니다.
//  · **결과는 클래스 하나**입니다 (argmax). 로짓 10개는 디버그용입니다.
//
// ## 레지스터 맵
//
//  off    R/W  이름          내용
//  0x000   W   CTRL          [0] start(펄스)  [1] dump(펄스)
//  0x004   R   STATUS        [0] done  [1] busy  [5:2] state  [13:6] inst_ptr
//                            [14] tok_req  [15] loader busy  [16] dumping
//                            [22:17] 요청 중인 타임스텝 t
//  0x008  RW   N_BODY        타임스텝당 명령어 수 (118)
//  0x00C  RW   N_TAIL        끝에 한 번 도는 명령어 수 (5)
//  0x010  RW   N_TSTEP       타임스텝 수 T (<= 20)
//  0x014  RW   LOAD_SEL      0=W 1=A 2=RQ 3=AF 4=INST 5=POS
//  0x018   W   LOAD_BASE     쓰면 로더를 arm (시작 워드 주소)
//  0x01C   R   VERSION       0x4556_5401 ("EVT" + v1)
//  0x020   R   CYCLES        마지막 실행 클럭 수
//  0x024  RW   EPS           LayerNorm eps (fp32 비트패턴)
//  0x028  RW   DUMP_BASE     A_Mem 덤프 시작 워드
//  0x02C  RW   DUMP_LEN      덤프 워드 수 (워드당 64바이트)
//  0x030   R   WORDS_LOADED  마지막 로드가 쓴 워드 수 (DMA 검산용)
//  0x034  RW   TOK_N         이번 타임스텝의 토큰 수
//  0x038   W   TOK_ACK       쓰면 ack 펄스 (X/PIN 적재 완료 통보)
//  0x03C   R   RES_CLASS     [3:0] argmax
//  0x400+  R   RES_LOGITS    10워드 (클래스별 acc, 디버그)
//
// ## 사용 순서
//
//   1회      : W / RQ / AF / INST / POS 를 DMA (LOAD_SEL → LOAD_BASE → DMA)
//   샘플마다 : latinit 을 Z 와 LATV 에, bkv 를 BKV 에 DMA
//              N_TSTEP 쓰고 CTRL.start
//              T 번 반복 { STATUS.tok_req 대기 → t 읽기 → X/PIN DMA
//                          → TOK_N 쓰기 → TOK_ACK }
//              STATUS.done 대기 → RES_CLASS 읽기
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Top #(
    parameter N      = 32,
    parameter ACT_W  = 8,
    parameter PSUM_W = 32,
    parameter DIM_W  = 16,
    parameter W_W    = 4,        // 가중치 레인 폭 : 8 = A8W8, 4 = A8W4 (니블 팩)
    parameter AW_W   = 14,
    parameter AW_A   = 13,       // A_Mem 깊이 2^13 = 8,192 (실사용 7,649)
    parameter AW_RQ  = 10,
    parameter AW_AF  = 8,
    parameter AW_INST   = 8,
    parameter N_CLASS = 10,
    parameter SW     = 128,
    parameter C_S_AXI_ADDR_WIDTH = 12,
    parameter C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                            aclk,
    input  wire                            aresetn,

    // ---- AXI4-Lite ----
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  wire [2:0]                      s_axi_awprot,
    input  wire                            s_axi_awvalid,
    output wire                            s_axi_awready,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]   s_axi_wdata,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] s_axi_wstrb,
    input  wire                            s_axi_wvalid,
    output wire                            s_axi_wready,
    output wire [1:0]                      s_axi_bresp,
    output wire                            s_axi_bvalid,
    input  wire                            s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]   s_axi_araddr,
    input  wire [2:0]                      s_axi_arprot,
    input  wire                            s_axi_arvalid,
    output wire                            s_axi_arready,
    output reg  [C_S_AXI_DATA_WIDTH-1:0]   s_axi_rdata,
    output wire [1:0]                      s_axi_rresp,
    output wire                            s_axi_rvalid,
    input  wire                            s_axi_rready,

    // ---- AXI-Stream 입력 (DMA MM2S) ----
    input  wire                            s_axis_tvalid,
    output wire                            s_axis_tready,
    input  wire [SW-1:0]                   s_axis_tdata,
    input  wire [SW/8-1:0]                 s_axis_tkeep,
    input  wire                            s_axis_tlast,

    // ---- AXI-Stream 출력 (DMA S2MM) ----
    output wire                            m_axis_tvalid,
    input  wire                            m_axis_tready,
    output wire [SW-1:0]                   m_axis_tdata,
    output wire [SW/8-1:0]                 m_axis_tkeep,
    output wire                            m_axis_tlast
);
    wire rst = ~aresetn;

    // =========================================================================
    // AXI4-Lite slave
    // =========================================================================
    localparam AB = C_S_AXI_ADDR_WIDTH - 2;         // 워드 주소 비트수 (=10)
    localparam RES_LO = 10'd256;                    // 0x400 (RES_LOGITS)

    reg awready_r, wready_r, bvalid_r, arready_r, rvalid_r;

    assign s_axi_awready = awready_r;
    assign s_axi_wready  = wready_r;
    assign s_axi_bvalid  = bvalid_r;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_arready = arready_r;
    assign s_axi_rvalid  = rvalid_r;
    assign s_axi_rresp   = 2'b00;

    wire wr_fire = s_axi_awvalid && s_axi_wvalid && !awready_r && !bvalid_r;
    wire rd_fire = s_axi_arvalid && !arready_r && !rvalid_r;

    wire [AB-1:0] wa = s_axi_awaddr[C_S_AXI_ADDR_WIDTH-1:2];
    wire [AB-1:0] ra = s_axi_araddr[C_S_AXI_ADDR_WIDTH-1:2];

    reg [AW_INST-1:0]  r_n_body, r_n_tail;
    reg [5:0]       r_n_tstep;
    reg [2:0]       r_load_sel;
    reg [AW_W-1:0]  r_load_base;
    reg [31:0]      r_eps;
    reg [AW_A-1:0]  r_dump_base;
    reg [AW_A:0]    r_dump_len;
    reg [DIM_W-1:0] r_tok_n;
    reg             r_arm, r_start, r_dump, r_tok_ack;

    always @(posedge aclk) begin
        if (rst) begin
            awready_r <= 1'b0; wready_r <= 1'b0; bvalid_r <= 1'b0;
            r_n_body <= 8'd118; r_n_tail <= 8'd5; r_n_tstep <= 6'd20;
            r_load_sel <= 3'd0; r_load_base <= {AW_W{1'b0}};
            r_eps <= 32'h3727c5ac;                  // 1e-5 (골든과 같은 비트)
            r_dump_base <= {AW_A{1'b0}}; r_dump_len <= {(AW_A+1){1'b0}};
            r_tok_n <= {DIM_W{1'b0}};
            r_arm <= 1'b0; r_start <= 1'b0; r_dump <= 1'b0; r_tok_ack <= 1'b0;
        end else begin
            // 전부 1클럭 펄스
            r_arm <= 1'b0; r_start <= 1'b0; r_dump <= 1'b0; r_tok_ack <= 1'b0;

            awready_r <= wr_fire;
            wready_r  <= wr_fire;
            if (wr_fire) begin
                bvalid_r <= 1'b1;
                case (wa)
                    10'h000: begin r_start <= s_axi_wdata[0];
                                   r_dump  <= s_axi_wdata[1]; end
                    10'h002: r_n_body    <= s_axi_wdata[AW_INST-1:0];
                    10'h003: r_n_tail    <= s_axi_wdata[AW_INST-1:0];
                    10'h004: r_n_tstep    <= s_axi_wdata[5:0];
                    10'h005: r_load_sel  <= s_axi_wdata[2:0];
                    10'h006: begin r_load_base <= s_axi_wdata[AW_W-1:0];
                                   r_arm <= 1'b1; end
                    10'h009: r_eps       <= s_axi_wdata;
                    10'h00A: r_dump_base <= s_axi_wdata[AW_A-1:0];
                    10'h00B: r_dump_len  <= s_axi_wdata[AW_A:0];
                    10'h00D: r_tok_n     <= s_axi_wdata[DIM_W-1:0];
                    10'h00E: r_tok_ack   <= 1'b1;
                    default: ;
                endcase
            end else if (bvalid_r && s_axi_bready) begin
                bvalid_r <= 1'b0;
            end
        end
    end

    // =========================================================================
    // 엔진
    // =========================================================================
    wire            eng_done, eng_busy;
    wire [3:0]      eng_state;
    wire [AW_INST-1:0] eng_inst;
    wire [5:0]      tstep_idx;
    wire            tok_req;
    wire [3:0]      res_class;
    wire [N_CLASS*PSUM_W-1:0] res_logits;

    wire                ld_we;
    wire [2:0]          ld_sel;
    wire [AW_W-1:0]     ld_addr;
    wire [N*16-1:0]     ld_data;
    wire [AW_W-1:0]     words_loaded;
    wire                ld_busy;

    wire                dump_rd_en, dump_busy;
    wire [AW_A-1:0]     dump_rd_addr;
    wire [N*16-1:0]     dump_rd_data;

    EvT_Engine #(.N(N), .ACT_W(ACT_W), .W_W(W_W), .PSUM_W(PSUM_W), .DIM_W(DIM_W),
                 .AW_W(AW_W), .AW_A(AW_A), .AW_RQ(AW_RQ), .AW_AF(AW_AF),
                 .AW_INST(AW_INST), .N_CLASS(N_CLASS)) u_engine (
        .clk(aclk), .rst(rst),
        .start(r_start), .done(eng_done), .busy(eng_busy),
        .dbg_state(eng_state), .dbg_inst(eng_inst),
        .n_body(r_n_body), .n_tail(r_n_tail), .n_tstep(r_n_tstep), .eps(r_eps),
        .tstep_idx(tstep_idx), .tok_n(r_tok_n),
        .tok_req(tok_req), .tok_ack(r_tok_ack),
        .ld_we(ld_we), .ld_sel(ld_sel), .ld_addr(ld_addr), .ld_data(ld_data),
        .res_class(res_class), .res_logits(res_logits),
        // 덤프는 엔진이 쉴 때만 (A_Mem 읽기 포트를 나눠 씁니다)
        .dbg_rd_en(dump_rd_en), .dbg_rd_addr(dump_rd_addr),
        .dbg_rd_data(dump_rd_data));

    // =========================================================================
    // 로더 / 덤프
    // =========================================================================
    Axis_Loader #(.SW(SW), .DW(N*16), .AW(AW_W)) u_loader (
        .clk(aclk), .rst(rst),
        .arm(r_arm), .arm_sel(r_load_sel), .arm_base(r_load_base),
        .s_tvalid(s_axis_tvalid), .s_tready(s_axis_tready),
        .s_tdata(s_axis_tdata), .s_tlast(s_axis_tlast),
        .ld_we(ld_we), .ld_sel(ld_sel), .ld_addr(ld_addr), .ld_data(ld_data),
        .words_written(words_loaded), .busy(ld_busy));

    Axis_Dump #(.SW(SW), .DW(N*16), .AW(AW_A)) u_dump (
        .clk(aclk), .rst(rst),
        .start(r_dump), .base(r_dump_base), .len(r_dump_len),
        .rd_en(dump_rd_en), .rd_addr(dump_rd_addr), .rd_data(dump_rd_data),
        .m_tvalid(m_axis_tvalid), .m_tready(m_axis_tready),
        .m_tdata(m_axis_tdata), .m_tkeep(m_axis_tkeep), .m_tlast(m_axis_tlast),
        .busy(dump_busy));

    // =========================================================================
    // 사이클 카운터 + done 래치
    // =========================================================================
    // 엔진의 `done` 은 한 사이클만 뜨고 사라집니다. AXI 폴링이 그 순간을 놓치기
    // 쉬우므로 **붙잡아 둡니다** (다음 `start` 에서 해제).
    reg [31:0] cyc_run, cyc_last;
    reg        done_latch;
    always @(posedge aclk) begin
        if (rst) begin cyc_run <= 32'd0; cyc_last <= 32'd0; done_latch <= 1'b0; end
        else begin
            if (r_start) begin cyc_run <= 32'd0; done_latch <= 1'b0; end
            else if (eng_busy) cyc_run <= cyc_run + 1'b1;
            // [함정] `done` 은 **busy 가 이미 내려간 뒤** 뜹니다 — 엔진이 ST_DONE
            // 에서 `done<=1` 과 `state<=ST_IDLE` 을 같은 엣지에 하기 때문입니다.
            // busy 로 게이팅하면 영영 못 잡습니다.
            if (eng_done && !done_latch) begin cyc_last <= cyc_run; done_latch <= 1'b1; end
        end
    end

    // =========================================================================
    // 읽기
    // =========================================================================
    wire [31:0] status = {9'd0, tstep_idx, dump_busy, ld_busy, tok_req,
                          eng_inst, eng_state, eng_busy, done_latch};

    always @(posedge aclk) begin
        if (rst) begin
            arready_r <= 1'b0; rvalid_r <= 1'b0; s_axi_rdata <= 32'd0;
        end else begin
            arready_r <= rd_fire;
            if (rd_fire) begin
                rvalid_r <= 1'b1;
                case (ra)
                    10'h001: s_axi_rdata <= status;
                    10'h002: s_axi_rdata <= {{(32-AW_INST){1'b0}}, r_n_body};
                    10'h003: s_axi_rdata <= {{(32-AW_INST){1'b0}}, r_n_tail};
                    10'h004: s_axi_rdata <= {26'd0, r_n_tstep};
                    10'h007: s_axi_rdata <= 32'h4556_5401;   // "EVT" v1
                    10'h008: s_axi_rdata <= cyc_last;
                    10'h009: s_axi_rdata <= r_eps;
                    10'h00C: s_axi_rdata <= {{(32-AW_W){1'b0}}, words_loaded};
                    10'h00D: s_axi_rdata <= {{(32-DIM_W){1'b0}}, r_tok_n};
                    10'h00F: s_axi_rdata <= {28'd0, res_class};
                    default: s_axi_rdata <= (ra >= RES_LO && ra < RES_LO + N_CLASS)
                                          ? res_logits[(ra - RES_LO)*PSUM_W +: PSUM_W]
                                          : 32'd0;
                endcase
            end else if (rvalid_r && s_axi_rready) begin
                rvalid_r <= 1'b0;
            end
        end
    end
endmodule
