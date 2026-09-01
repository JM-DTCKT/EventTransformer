// tb_format_cast_act : 컴파일 + ReLU 분기 스모크. 정밀 검증은 tb_evt(통합) 에서.
`timescale 1ns/1ps
module tb_format_cast_act;
  localparam N=32, ACT_W=8, PSUM_W=32;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg [1:0] fmt=0, act_sel=0;
  reg [7:0] act_parm=0;
  reg signed [PSUM_W-1:0] bias=0, mult=0, g_mult=0;
  reg [5:0] shift=0, g_shift=0;
  reg in_valid=0, raw16=0, req2=0;
  reg [N*PSUM_W-1:0] acc=0;
  wire out_valid; wire [N*16-1:0] out_data, out_q69;
  Format_Cast_Act #(.N(N), .ACT_W(ACT_W), .PSUM_W(PSUM_W)) dut (
    .clk(clk), .rst(rst), .fmt(fmt), .bias(bias), .mult(mult),
    .shift(shift), .g_mult(g_mult), .g_shift(g_shift),
    .act_sel(act_sel), .act_parm(act_parm), .raw16(raw16), .req2(req2),
    .in_valid(in_valid), .acc(acc),
    .out_valid(out_valid), .out_data(out_data), .out_q69(out_q69));

  integer errors=0, checks=0, i;
  reg signed [15:0] got;
  task drive(input [1:0] f, input [1:0] sel, input integer v);
    begin
      fmt=f; act_sel=sel; mult=32'sd1<<20; shift=6'd20; bias=0;
      @(negedge clk); in_valid=1;
      for (i=0;i<N;i=i+1) acc[i*PSUM_W +: PSUM_W] = (i%2) ? v : -v;
      @(posedge clk); @(negedge clk); in_valid=0;
      // 가장 긴 소비자(Q4.11 = requant 3 + GELU 3 + requant 3 = 9단) 보다
      // 넉넉히 기다립니다. `Requant_Int` 을 3단으로 늘렸을 때 여기 4가
      // 그대로 남아 REQ2(6단) 가 아직 안 나온 값을 읽었습니다.
      repeat (12) @(posedge clk);
    end
  endtask

  initial begin
    $display("[tb_format_cast_act] 소비자 분기 + ReLU");
    repeat(4) @(posedge clk); rst=0; @(posedge clk);

    drive(2'd0, 2'd0, 100);            // INT8, 활성 없음 → 음수 보존
    got = $signed(out_data[15:0]);
    checks=checks+1; if (got !== -16'sd100) begin errors=errors+1;
      $display("    [FAIL] ACT_NONE lane0 got %0d exp -100", got); end
    $display("  ACT_NONE  lane0=%0d lane1=%0d  (기대 -100, 100)",
             $signed(out_data[15:0]), $signed(out_data[31:16]));

    // 활성함수 뒤 2차 재양자화 : g_mult/g_shift 로 한 번 더 (여기선 x1/2)
    req2 = 1'b1; g_mult = 32'sd1 <<< 19; g_shift = 6'd20;
    drive(2'd0, 2'd0, 100);
    got = $signed(out_data[15:0]);
    checks=checks+1; if (got !== -16'sd50) begin errors=errors+1;
      $display("    [FAIL] REQ2 lane0 got %0d exp -50", got); end
    $display("  REQ2      lane0=%0d lane1=%0d  (기대 -50, 50)",
             $signed(out_data[15:0]), $signed(out_data[31:16]));
    req2 = 1'b0;

    drive(2'd0, 2'd1, 100);            // INT8 + ReLU → 음수 0
    got = $signed(out_data[15:0]);
    checks=checks+1; if (got !== 16'sd0) begin errors=errors+1;
      $display("    [FAIL] ReLU lane0 got %0d exp 0", got); end
    checks=checks+1; if ($signed(out_data[31:16]) !== 16'sd100) errors=errors+1;
    $display("  ReLU      lane0=%0d lane1=%0d  (기대 0, 100)",
             $signed(out_data[15:0]), $signed(out_data[31:16]));

    if (errors==0) $display("=== tb_format_cast_act: %0d checks, TEST PASSED ===", checks);
    else           $display("=== tb_format_cast_act: %0d/%0d failed, TEST FAILED ===", errors, checks);
    $finish;
  end
endmodule
