// tb_ln_wrap : LayerNorm_Top 래퍼 단독 — 열 개수와 정규화 결과를 봅니다
`timescale 1ns/1ps
module tb_ln_wrap;
  localparam N=32, E=128, AW=14;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg start=0; reg [15:0] M=96;
  wire done;
  wire rd_en; wire [AW-1:0] rd_addr; reg [N*16-1:0] rd_data;
  wire [15:0] p_addr; wire ov; wire [5:0] omt; wire [15:0] ok;
  wire [N*8-1:0] od;
  LayerNorm_Top #(.N(N), .E(E), .AW(AW)) dut (
    .clk(clk), .rst(rst), .start(start), .done(done), .M(M),
    .a_base(14'd0), .in_shift(6'sd0), .in_q411(1'b0),
    .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_data),
    .af_addr(p_addr), .af_gamma(16'sd16384), .af_beta(16'sd0),   // gamma=1, beta=0
    .mult(32'sd1048576), .shift(6'd20),                        // x1 (Q4.11→그대로)
    .out_valid(ov), .out_mt(omt), .out_k(ok), .out_data(od));

  // A_Mem 흉내 : 주소 = mt*128+k, 레인 r 에 bf16(r+1 + k*0.01) 비슷한 값
  reg [N*16-1:0] mem [0:383];
  integer a, r, nout=0, i;
  integer seen_mt [0:5];
  initial begin
    for (a=0;a<384;a=a+1)
      for (r=0;r<N;r=r+1)
        // bf16 : 지수 127 + 작은 가수 변화 (0.5~2 범위)
        mem[a][r*16 +: 16] = 16'h3F00 + ((a*7+r*3) % 64);
    for (i=0;i<6;i=i+1) seen_mt[i]=0;
    repeat(4) @(posedge clk); rst=0; @(posedge clk);
    @(negedge clk); start=1;
    i=0; while(!done && i<20000) begin @(posedge clk); i=i+1; end
    @(negedge clk); start=0; repeat(4) @(posedge clk);
    $display("  M=%0d (타일 3개)  %0d 사이클  출력 %0d 열 (기대 %0d)",
             M, i, nout, 3*E);
    $display("  타일별 출력 : mt0=%0d mt1=%0d mt2=%0d mt3=%0d",
             seen_mt[0], seen_mt[1], seen_mt[2], seen_mt[3]);
    if (nout == 3*E && seen_mt[0]==E && seen_mt[1]==E && seen_mt[2]==E)
      $display("=== tb_ln_wrap: TEST PASSED ===");
    else $display("=== tb_ln_wrap: TEST FAILED ===");
    $finish;
  end
  always @(posedge clk) if (!rst && rd_en) rd_data <= mem[rd_addr];
  always @(posedge clk) if (!rst && ov) begin
    nout = nout + 1;
    if (omt < 6) seen_mt[omt] = seen_mt[omt] + 1;
  end
endmodule
