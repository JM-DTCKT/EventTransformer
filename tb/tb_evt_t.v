// -----------------------------------------------------------------------------
// tb_evt_t : 샘플 하나를 **끝까지** — 20 타임스텝 + 꼬리 → 클래스 하나
//
// `tb_evt` 가 타임스텝 하나를 값 단위로 대조했다면, 여기는 **latent 누적**과
// 타임스텝별 입력 교체를 봅니다. 확인하는 것:
//
//   · `tok_req`/`tok_ack` 핸드셰이크 — X/PIN 은 A_Mem 에 **한 타임스텝분**만
//     들어가므로 타임스텝마다 호스트가 새로 채워야 합니다 (20벌이면 24k 워드로
//     A_Mem 을 넘습니다)
//   · `n_tok` 이 타임스텝마다 바뀌는데 step 프로그램은 **그대로** 라는 것
//     (VAR 비트가 M/K/NOUT/C 를 발행 시점에 채웁니다)
//   · `latent_vectors += z` 누적 (LATV)
//
// 기대 클래스는 골든의 20 타임스텝 예측입니다 (`data/golden/expect_full.json`).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_evt_t;
  localparam N=32, AW_A=13, AW_W=14, AW_S=8, DIM_W=16;
  localparam TTM = 4;                       // 최악치 타일 수 (TOK_MAX=128)
  localparam T = 20, EXP_CLASS = 9;
  localparam R_X=0, R_PIDX=576, R_PIN=580, R_LATV=2756,
             R_Z=3140, R_BKV=6728;
  localparam W_WORDS=14000, PB_WORDS=869, PG_WORDS=208, S_WORDS=123,
             LAT_WORDS=384, BKV_WORDS=24;

  reg clk = 0, rst = 1; always #5 clk = ~clk;

  reg  [AW_S-1:0] n_body = 118, n_tail = 5;
  reg  [5:0]      n_time = T;
  reg  [31:0]     eps = 32'h3727c5ac;
  reg             start = 0;
  wire            done, busy;
  wire [3:0]      dbg_state;
  wire [AW_S-1:0] dbg_step;
  wire [5:0]      tok_rd_idx;
  reg  [DIM_W-1:0] tok_rd_n = 0;
  wire            tok_req;
  reg             tok_ack = 0;
  reg             ld_we = 0;
  reg  [2:0]      ld_sel = 0;
  reg  [AW_W-1:0] ld_addr = 0;
  reg  [N*16-1:0] ld_data = 0;
  wire [3:0]      res_class;
  wire [N*32-1:0] res_logits;

  EvT_Engine #(.N(N), .AW_A(AW_A), .AW_W(AW_W), .AW_S(AW_S)) dut (
    .clk(clk), .rst(rst), .start(start), .done(done), .busy(busy),
    .dbg_state(dbg_state), .dbg_step(dbg_step),
    .n_body(n_body), .n_tail(n_tail), .n_time(n_time), .eps(eps),
    .tok_rd_idx(tok_rd_idx), .tok_rd_n(tok_rd_n),
    .tok_req(tok_req), .tok_ack(tok_ack),
    .ld_we(ld_we), .ld_sel(ld_sel), .ld_addr(ld_addr), .ld_data(ld_data),
    .res_class(res_class), .res_logits(res_logits),
    .dbg_rd_en(1'b0), .dbg_rd_addr({AW_A{1'b0}}), .dbg_rd_data());

  reg [N*8-1:0]  wimg  [0:W_WORDS-1];
  reg [N*8-1:0]  pbimg [0:PB_WORDS-1];
  reg [N*8-1:0]  pgimg [0:PG_WORDS-1];
  reg [N*8-1:0]  simg  [0:S_WORDS-1];
  reg [N*16-1:0] limg  [0:LAT_WORDS-1];
  reg [N*16-1:0] bimg  [0:BKV_WORDS-1];
  reg [N*16-1:0] ximg  [0:T*TTM*144-1];     // 고정 stride
  reg [N*16-1:0] pimg  [0:T*TTM*64-1];      // (참조용, 이제 안 씁니다)
  reg [N*16-1:0] qimg  [0:T*TTM-1];         // pos_idx — 타임스텝당 TTM 워드
  reg [64*8-1:0] posimg [0:440];            // PL 표 441행 x 64B
  reg [15:0]     ntok  [0:T-1];

  integer cyc, w, mt, d, t_done;

  task load(input [2:0] sel, input integer nw, input integer which);
    integer k;
    begin
      for (k = 0; k < nw; k = k + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = sel; ld_addr = k;
        case (which)
          0: ld_data = {{(N*8){1'b0}}, wimg[k]};
          1: ld_data = {{(N*8){1'b0}}, pbimg[k]};
          2: ld_data = {{(N*8){1'b0}}, pgimg[k]};
          3: ld_data = {{(N*8){1'b0}}, simg[k]};
        endcase
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  // PL 의 pos enc 표 (로더 sel 5) — 샘플과 무관하게 한 번만
  task load_postbl;
    integer k;
    begin
      for (k = 0; k < 441; k = k + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd5; ld_addr = k;
        ld_data = posimg[k];
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  task load_amem(input integer base, input integer nw, input integer which);
    integer k;
    begin
      for (k = 0; k < nw; k = k + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd1; ld_addr = base + k;
        ld_data = (which == 0) ? limg[k] : bimg[k];
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  // ---- 타임스텝 t 의 X / pos_idx 를 채우고 ack ----
  // X    : R_X    + mt*144 + k
  // PIDX : R_PIDX + mt              (pos enc 는 PL 표에서 `Pos_Gather` 가 모음)
  task feed(input integer t);
    integer n, tt, k;
    begin
      n  = ntok[t];
      tt = (n + N - 1) / N;
      tok_rd_n = n[DIM_W-1:0];
      for (k = 0; k < tt * 144; k = k + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd1;
        ld_addr = R_X + k;  ld_data = ximg[t*TTM*144 + k];
      end
      for (mt = 0; mt < tt; mt = mt + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd1;
        ld_addr = R_PIDX + mt;
        ld_data = qimg[t*TTM + mt];
      end
      @(negedge clk); ld_we = 0;
      @(negedge clk); tok_ack = 1;
      @(negedge clk); tok_ack = 0;
    end
  endtask

  // 엔진이 요청하면 채워 줍니다 (보드에서는 DMA 가 하는 일)
  integer fed = 0;
  always @(posedge clk) if (!rst && tok_req && fed <= tok_rd_idx) begin
    fed = tok_rd_idx + 1;
    $display("  t=%0d 요청 → 토큰 %0d개 적재 (%0d 사이클째)", tok_rd_idx,
             ntok[tok_rd_idx], cyc);
    feed(tok_rd_idx);
  end

  initial begin
    $display("[tb_evt_t] 샘플0 전체 %0d 타임스텝 → 클래스 (기대 %0d)", T, EXP_CLASS);
    $readmemh("../data/wmem.hex",  wimg);
    $readmemh("../data/pbmem.hex", pbimg);
    $readmemh("../data/pgmem.hex", pgimg);
    $readmemh("../data/stepmem.hex", simg);
    $readmemh("../data/latinit.hex", limg);
    $readmemh("../data/bkv.hex", bimg);
    $readmemh("../data/board/sim_x.hex",    ximg);
    $readmemh("../data/board/sim_pidx.hex", qimg);
    $readmemh("../data/posmem.hex", posimg);
    $readmemh("../data/board/sim_ntok.hex", ntok);

    repeat (8) @(posedge clk); rst = 0; @(posedge clk);
    $display("-- 상수 적재");
    load(3'd0, W_WORDS,  0);  load(3'd2, PB_WORDS, 1);
    load(3'd3, PG_WORDS, 2);  load(3'd4, S_WORDS,  3);
    load_amem(R_Z, LAT_WORDS, 0); load_amem(R_LATV, LAT_WORDS, 0);
    load_amem(R_BKV, BKV_WORDS, 1);
    load_postbl;                        // pos enc 표 (1회, 27.6 KB)

    $display("-- 실행");
    @(negedge clk); start = 1;
    cyc = 0;
    while (!done && cyc < 100000000) begin @(posedge clk); cyc = cyc + 1; end
    @(negedge clk); start = 0;
    if (cyc >= 100000000) begin
      $display("=== TIMEOUT state=%0d step=%0d t=%0d ===",
               dbg_state, dbg_step, tok_rd_idx);
      $finish;
    end
    $display("     %0d 사이클  (100 MHz 에서 %.2f ms)", cyc, cyc/100000.0);
    $display("     res_class = %0d  (기대 %0d)", res_class, EXP_CLASS);
    if (res_class === EXP_CLASS[3:0])
      $display("=== tb_evt_t: TEST PASSED ===");
    else
      $display("=== tb_evt_t: 클래스 불일치, TEST FAILED ===");
    $finish;
  end
endmodule
