// -----------------------------------------------------------------------------
// tb_gemm_ev : Gemm_Core 를 attention 두 용법으로 — **실제 EvT 데이터** 검증
//
// Linear 용법은 `fpga/`·`fpga_nl` 이 이미 보드까지 검증했으므로, 여기서는 이 코어의
// **새 기능**만 봅니다: B 피연산자가 W_Mem 이 아니라 A_Mem 에서 오는 두 경우.
//
//   ① Q·Kᵀ    M=Lq(96)  K=32(head_dim)  Nout=Lk(53)
//              A 워드[d] 레인=query,  B 워드[d] 레인=key
//              → in_proj 이 뱉는 "출력채널 하나 = 32행 컬럼" 레이아웃 그대로
//
//   ② attn·V  M=Lq(96)  K=Lk(53)        Nout=32(head_dim)
//              A 워드[j] 레인=query  ← score GEMM 컬럼 출력 그대로
//              B 워드[j] 레인=d      ← **Transpose32 가 돌려 놓은** V
//
// 벡터는 `real_dvs_script/taps` 의 골든입니다 — cross_attention head 0 의 실제
// Q/K/V/attn 정수 코드와 INT32 누산기. 즉 양자화 모델이 실제로 계산한 값입니다.
//
// M=96 은 32의 3배라 **행 타일링(mt)** 이 실제로 돌고, Nout=53 은 32의 배수가
// 아니라 **edge mask** 도 같이 검증됩니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_gemm_ev;
  localparam N = 32, ACT_W = 8, PSUM_W = 32, DIM_W = 16;
  localparam AW_A = 12, AW_B = 14;
  localparam LQ = 96, LK = 53, HD = 32;
  localparam MT = (LQ + 31) / 32;                 // 행 타일 3개

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg  [DIM_W-1:0] M = 0, K = 0, Nout = 0;
  reg  [AW_A-1:0]  a_base = 0;
  reg  [AW_B-1:0]  b_base = 0;
  reg              start = 0;
  wire             all_done;

  wire             a_rd_en, b_rd_en;
  wire [AW_A-1:0]  a_rd_addr;
  wire [AW_B-1:0]  b_rd_addr;
  reg  [N*16-1:0]  a_rd_data;
  reg  [N*8-1:0]   b_rd_data;

  wire                 col_valid;
  wire [N*PSUM_W-1:0]  col_data;
  wire [DIM_W-1:0]     col_n, col_mt;

  Gemm_Core #(.N(N), .ACT_W(ACT_W), .PSUM_W(PSUM_W), .DIM_W(DIM_W),
                 .AW_A(AW_A), .AW_B(AW_B)) dut (
    .clk(clk), .rst(rst), .start(start), .all_done(all_done),
    .M(M), .K(K), .Nout(Nout), .a_base(a_base), .b_base(b_base),
    .a_rd_en(a_rd_en), .a_rd_addr(a_rd_addr), .a_rd_data(a_rd_data),
    .b_rd_en(b_rd_en), .b_rd_addr(b_rd_addr), .b_rd_data(b_rd_data),
    .col_valid(col_valid), .col_data(col_data), .col_n(col_n), .col_mt(col_mt));

  // ---- 메모리 (읽기 지연 1사이클 — Bram_Sdp 와 동일) ----
  reg [N*16-1:0] amem [0:4095];
  reg [N*8-1:0]  bmem [0:4095];
  always @(posedge clk) begin
    if (a_rd_en) a_rd_data <= amem[a_rd_addr];
    if (b_rd_en) b_rd_data <= bmem[b_rd_addr];
  end

  // ---- 골든 ----
  reg [N*16-1:0] qk_a [0:MT*HD-1];
  reg [N*8-1:0]  qk_b [0:1*64-1];
  reg [N*32-1:0] qk_e [0:MT*LK-1];
  reg [N*16-1:0] av_a [0:MT*LK-1];
  reg [N*8-1:0]  av_b [0:LK-1];
  reg [N*32-1:0] av_e [0:MT*32-1];

  reg [N*32-1:0] got [0:MT*128-1];        // [mt*Nout + n]
  reg [MT*128-1:0] seen;

  integer errors = 0, checks = 0, i, j, e0, cyc;

  // 컬럼은 **항상** 받아 둡니다
  always @(posedge clk) if (!rst && col_valid) begin
    got[col_mt*Nout + col_n] <= col_data;
    seen[col_mt*Nout + col_n] <= 1'b1;
  end

  task run_gemm(input [DIM_W-1:0] m_, input [DIM_W-1:0] k_, input [DIM_W-1:0] n_);
    begin
      M = m_; K = k_; Nout = n_; a_base = 0; b_base = 0;
      seen = 0;
      @(negedge clk); start = 1;
      cyc = 0;
      while (!all_done) begin
        @(posedge clk); cyc = cyc + 1;
        if (cyc > 500000) begin $display("=== TIMEOUT ==="); $finish; end
      end
      @(negedge clk); start = 0;
      repeat (6) @(posedge clk);
    end
  endtask

  integer w;
  initial begin
    $display("[tb_gemm_ev] attention GEMM 2종 — cross_attention head0 실데이터");
    $readmemh("../data/vec/qk_a.hex", qk_a);
    $readmemh("../data/vec/qk_b.hex", qk_b);
    $readmemh("../data/vec/qk_exp.hex", qk_e);
    $readmemh("../data/vec/av_a.hex", av_a);
    $readmemh("../data/vec/av_b.hex", av_b);
    $readmemh("../data/vec/av_exp.hex", av_e);

    // unisim 의 **GSR 은 100 ns 까지 모든 프리미티브를 리셋에 붙들어 둡니다**
    // (`glbl.v`). 그 전에 DSP48E2 가 곱하기 시작하면 MREG/PREG 가 안 걸려
    // **첫 k 가 조용히 사라집니다.** 예전 코어는 순서기의 CLR 상태 한 칸
    // 덕에 우연히 105 ns 부터 곱해서 안 걸렸을 뿐입니다 — 타일 파이프라인은
    // 한 사이클 일찍 시작합니다. 보드에 GSR 은 없으니 TB 쪽을 맞춥니다.
    repeat (16) @(posedge clk); rst = 0; @(posedge clk);

    // ---------------- ① Q·Kᵀ ----------------
    for (w = 0; w < MT*HD; w = w + 1) amem[w] = qk_a[w];
    for (w = 0; w < 64;    w = w + 1) bmem[w] = qk_b[w];
    run_gemm(LQ, HD, LK);
    e0 = errors;
    begin : CHK_QK
      integer mt, n, lane, nseen; reg signed [31:0] g, x;
      nseen = 0;
      for (mt = 0; mt < MT; mt = mt + 1)
        for (n = 0; n < LK; n = n + 1) begin
          if (seen[mt*LK + n]) nseen = nseen + 1;
          for (lane = 0; lane < N; lane = lane + 1) begin
            if (mt*32 + lane >= LQ) continue;
            checks = checks + 1;
            x = got[mt*LK + n][lane*32 +: 32];
            g = qk_e[mt*LK + n][lane*32 +: 32];
            if (x !== g) begin
              errors = errors + 1;
              if (errors - e0 < 5)
                $display("    [FAIL] QK mt=%0d n=%0d lane=%0d  got %0d  exp %0d",
                         mt, n, lane, x, g);
            end
          end
        end
      $display("  ① Q·Kᵀ    M=%0d K=%0d Nout=%0d   %0d/%0d 일치  (컬럼 %0d/%0d, %0d 사이클)",
               LQ, HD, LK, LQ*LK - (errors - e0), LQ*LK, nseen, MT*LK, cyc);
    end

    // ---------------- ② attn·V ----------------
    for (w = 0; w < MT*LK; w = w + 1) amem[w] = av_a[w];
    for (w = 0; w < LK;    w = w + 1) bmem[w] = av_b[w];
    run_gemm(LQ, LK, 32);
    e0 = errors;
    begin : CHK_AV
      integer mt, n, lane, nseen; reg signed [31:0] g, x;
      nseen = 0;
      for (mt = 0; mt < MT; mt = mt + 1)
        for (n = 0; n < 32; n = n + 1) begin
          if (seen[mt*32 + n]) nseen = nseen + 1;
          for (lane = 0; lane < N; lane = lane + 1) begin
            if (mt*32 + lane >= LQ) continue;
            checks = checks + 1;
            x = got[mt*32 + n][lane*32 +: 32];
            g = av_e[mt*32 + n][lane*32 +: 32];
            if (x !== g) begin
              errors = errors + 1;
              if (errors - e0 < 5)
                $display("    [FAIL] AV mt=%0d n=%0d lane=%0d  got %0d  exp %0d",
                         mt, n, lane, x, g);
            end
          end
        end
      $display("  ② attn·V  M=%0d K=%0d Nout=32   %0d/%0d 일치  (컬럼 %0d/%0d, %0d 사이클)",
               LQ, LK, LQ*32 - (errors - e0), LQ*32, nseen, MT*32, cyc);
    end

    if (errors == 0) $display("=== tb_gemm_ev: %0d checks, TEST PASSED ===", checks);
    else             $display("=== tb_gemm_ev: %0d/%0d failed, TEST FAILED ===", errors, checks);
    $finish;
  end
endmodule
