// -----------------------------------------------------------------------------
// tb_transpose32 : 32x32 corner-turn — **전수** 검증
//
// 입력 공간이 32x32 바이트라 무작위 표본을 뽑을 이유가 없습니다. 두 가지를 봅니다:
//
//  ① 인덱스 전수 : X[t][d] = t*32+d 로 채우면 읽은 값이 곧 자기 좌표라
//    **한 칸이라도 어긋나면 즉시 드러납니다.** 축이 뒤집힌 버그(t↔d)는 대칭 패턴을
//    쓰면 통과해 버리므로 비대칭 패턴이어야 합니다.
//  ② 무작위 라운드트립 : 난수 행렬 200벌을 넣고 X[t][d] 를 전부 대조.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_transpose32;
  localparam N = 32, W = 8;

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg          we = 0;
  reg  [4:0]   w_idx = 0, r_idx = 0;
  reg  [N*W-1:0] w_data = 0;
  wire [N*W-1:0] r_data;

  Transpose32 #(.N(N), .W(W)) dut (
    .clk(clk), .rst(rst), .we(we), .w_idx(w_idx), .w_data(w_data),
    .r_idx(r_idx), .r_data(r_data));

  integer errors = 0, checks = 0;
  reg [W-1:0] X [0:N-1][0:N-1];      // X[t][d]
  integer t, d, r, e0;

  task load;                          // X 를 워드 d 단위로 써 넣음
    begin
      for (d = 0; d < N; d = d + 1) begin
        @(negedge clk);
        we = 1; w_idx = d[4:0];
        for (t = 0; t < N; t = t + 1) w_data[t*W +: W] = X[t][d];
      end
      @(negedge clk); we = 0;
    end
  endtask

  task check(input [255:0] nm);
    begin
      e0 = errors;
      for (t = 0; t < N; t = t + 1) begin
        @(negedge clk); r_idx = t[4:0];
        #1;                            // 조합 읽기 — 같은 사이클에 유효
        for (d = 0; d < N; d = d + 1) begin
          checks = checks + 1;
          if (r_data[d*W +: W] !== X[t][d]) begin
            errors = errors + 1;
            if (errors - e0 < 4)
              $display("    [FAIL] %0s t=%0d d=%0d  got %02x  exp %02x",
                       nm, t, d, r_data[d*W +: W], X[t][d]);
          end
        end
      end
    end
  endtask

  initial begin
    $display("[tb_transpose32] 32x32 corner-turn");
    repeat (4) @(posedge clk); rst = 0; @(posedge clk);

    // ---- ① 인덱스 전수 (비대칭이라 t/d 뒤바뀜을 잡습니다) ----
    for (t = 0; t < N; t = t + 1)
      for (d = 0; d < N; d = d + 1) X[t][d] = (t*N + d) & 8'hFF;
    load; check("index");
    $display("  ① 인덱스 전수      %0d/%0d 일치", checks - errors, checks);

    // ---- ② 무작위 라운드트립 ----
    e0 = checks;
    for (r = 0; r < 200; r = r + 1) begin
      for (t = 0; t < N; t = t + 1)
        for (d = 0; d < N; d = d + 1) X[t][d] = $random;
      load; check("rand");
    end
    $display("  ② 무작위 200벌     %0d/%0d 일치",
             (checks - e0) - errors, checks - e0);

    if (errors == 0) $display("=== tb_transpose32: %0d checks, TEST PASSED ===", checks);
    else             $display("=== tb_transpose32: %0d/%0d failed, TEST FAILED ===", errors, checks);
    $finish;
  end
endmodule
