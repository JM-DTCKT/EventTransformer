// tb_evt_compile : EvT_Engine 탄력 검사 — **컴파일과 리셋 탈출만** 봅니다.
// 기능 검증은 tap 대조(tb_evt)에서 하며, 아직 하지 않았습니다.
`timescale 1ns/1ps
module tb_evt_compile;
  localparam N=32, AW_A=14, AW_W=14, AW_S=8, DIM_W=16;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg start=0;
  wire done, busy; wire [3:0] dbg_state; wire [AW_S-1:0] dbg_step;
  reg [AW_S-1:0] n_body=4, n_tail=1; reg [5:0] n_time=2;
  reg [31:0] eps=32'h3727c5ac;
  wire [5:0] tok_rd_idx; reg [DIM_W-1:0] tok_rd_n=32;
  reg ld_we=0; reg [2:0] ld_sel=0; reg [AW_W-1:0] ld_addr=0; reg [N*16-1:0] ld_data=0;
  wire [3:0] res_class; wire [N*32-1:0] res_logits;
  reg dbg_rd_en=0; reg [AW_A-1:0] dbg_rd_addr=0; wire [N*16-1:0] dbg_rd_data;

  EvT_Engine #(.N(N), .AW_A(AW_A), .AW_W(AW_W), .AW_S(AW_S),
               .GELU_LUT_FILE("../../nl_export/lut/gelu.hex"),
               .EXP_LUT_FILE ("../../nl_export/lut/exp.hex"),
               .RCP_LUT_FILE ("../../nl_export/lut/recip.hex"),
               .RSQRT_LUT_FILE("../../nl_export/lut/rsqrt.hex")) dut (
    .clk(clk), .rst(rst), .start(start), .done(done), .busy(busy),
    .dbg_state(dbg_state), .dbg_step(dbg_step),
    .n_body(n_body), .n_tail(n_tail), .n_time(n_time), .eps(eps),
    .tok_rd_idx(tok_rd_idx), .tok_rd_n(tok_rd_n),
    .ld_we(ld_we), .ld_sel(ld_sel), .ld_addr(ld_addr), .ld_data(ld_data),
    .res_class(res_class), .res_logits(res_logits),
    .dbg_rd_en(dbg_rd_en), .dbg_rd_addr(dbg_rd_addr), .dbg_rd_data(dbg_rd_data));

  integer cyc=0;
  initial begin
    $display("[tb_evt_compile] EvT_Engine 컴파일 + 리셋/기동 검사");
    repeat(8) @(posedge clk); rst=0; @(posedge clk);
    if (busy !== 1'b0) $display("  [FAIL] 리셋 후 busy=%b (0 이어야 함)", busy);
    @(negedge clk); start=1;
    while (!done && cyc < 200000) begin @(posedge clk); cyc=cyc+1; end
    @(negedge clk); start=0;
    if (done) $display("  기동 → done  %0d 사이클 (step %0d, state %0d)", cyc, dbg_step, dbg_state);
    else      $display("  [진행 중] %0d 사이클에서 state=%0d step=%0d — 기능 검증은 tb_evt 에서",
                       cyc, dbg_state, dbg_step);
    $display("=== tb_evt_compile: TEST PASSED (컴파일/기동만) ===");
    $finish;
  end
endmodule
