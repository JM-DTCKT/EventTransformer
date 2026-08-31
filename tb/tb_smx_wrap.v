// tb_smx_wrap : 새 코어 래퍼가 예전 판과 같은 프로토콜로 도는지 (스모크)
`timescale 1ns/1ps
module tb_smx_wrap;
  localparam N=32, C=53;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg start=0, in_valid=0; reg [7:0] Cw=C;
  reg [N*16-1:0] ind=0;
  wire done, ov; wire [7:0] oc; wire [N*8-1:0] od;
  softmax_top #(.N(N), .CMAX(128)) dut (
    .clk(clk), .rst(rst), .start(start), .C(Cw), .done(done),
    .in_valid(in_valid), .in_data(ind),
    .out_valid(ov), .out_c(oc), .out_data(od));
  integer i, j, nout=0; integer sum0=0;
  integer run_c, bad=0;

  task one(input integer CC);
    begin
      nout = 0; sum0 = 0; Cw = CC;
      @(negedge clk); start=1;
      for (i=0;i<CC;i=i+1) begin
        @(negedge clk); in_valid=1;
        for (j=0;j<N;j=j+1) ind[j*16 +: 16] = $signed((i*7+j*3) % 97) - 48;
        @(posedge clk);
      end
      @(negedge clk); in_valid=0;
      i=0; while(!done && i<20000) begin @(posedge clk); i=i+1; end
      @(negedge clk); start=0; repeat(4) @(posedge clk);
      $display("  C=%0d → 출력 %0d 열, 레인0 합 %0d, %0d 사이클", CC, nout, sum0, i);
      if (nout != CC) bad = bad + 1;
    end
  endtask

  initial begin
    repeat(4) @(posedge clk); rst=0; @(posedge clk);
    // **길이를 바꿔 가며** 연달아 — 엔진이 실제로 그렇게 씁니다 (53 → 97)
    one(53); one(53); one(97); one(97); one(53); one(97);
    if (bad == 0) $display("=== tb_smx_wrap: TEST PASSED ===");
    else $display("=== tb_smx_wrap: 길이 불일치 %0d건, TEST FAILED ===", bad);
    $finish;
  end
  always @(posedge clk) if (!rst && ov) begin
    nout = nout + 1; sum0 = sum0 + od[7:0];
  end
endmodule
