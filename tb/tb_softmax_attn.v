// -----------------------------------------------------------------------------
// tb_softmax_attn : attention softmax — **실제 EvT 데이터**로 검증
//
// 합성 벡터가 아니라 `real_dvs_script/taps` 가 뽑은 골든입니다:
//
//   sm_score.hex   cross_attention head 0, 쿼리 0..31 의 Q6.9 score (컬럼 Lk개)
//   sm_prob.hex    같은 자리의 softmax 출력 uint8 (정수 LUT 경로) ← 기대값
//
// 기대값은 골든의 `attn_QK^T` INT32 누산기를 manifest 의 requant(M=1165286311,
// sh=32)로 내린 것이고, prob 은 골든의 `attn_AV` 첫 피연산자입니다. 즉 **양자화
// 모델이 실제로 계산한 값**이라, 여기를 통과하면 이 유닛은 실데이터에서 맞습니다.
//
// Lk=53 = 유효 토큰 52 + bias_k 1. 이 샘플/스텝은 패딩이 없어 마스크가 걸리지
// 않습니다 — 보드는 애초에 패딩을 안 받으므로 그것이 정상 경로입니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_softmax_attn;
  localparam N = 32, CMAX = 128;
  localparam LK = 53;

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg              start = 0, in_valid = 0;
  reg  [7:0]       C = LK;
  reg  [N*16-1:0]  in_data = 0;
  wire             done, out_valid;
  wire [7:0]       out_c;
  wire [N*8-1:0]   out_data;

  Softmax_Top #(.N(N), .CMAX(CMAX)) dut (
    .clk(clk), .rst(rst), .start(start), .n_col(C), .done(done),
    .in_valid(in_valid), .in_data(in_data),
    .out_valid(out_valid), .out_n(out_c), .out_data(out_data));

  reg [N*16-1:0] score [0:LK-1];
  reg [N*8-1:0]  gold  [0:LK-1];
  reg [N*8-1:0]  got   [0:LK-1];
  integer tol = 0;
  function integer diff(input integer a, input integer b);
    diff = (a > b) ? (a - b) : (b - a);
  endfunction

  integer errors = 0, checks = 0, j, i, cyc;
  integer rowsum [0:N-1];

  // 출력은 항상 받아 둡니다 (폴링으로 켜면 첫 비트를 놓칩니다)
  always @(posedge clk) if (!rst && out_valid) got[out_c] <= out_data;

  initial begin
    $display("[tb_softmax_attn] cross_attention head0, Lk=%0d, 쿼리 32개", LK);
    $readmemh("../data/vec/sm_score.hex", score);
    $readmemh("../data/vec/sm_prob.hex",  gold);

    repeat (6) @(posedge clk); rst = 0; @(posedge clk);

    @(negedge clk); start = 1;
    // score 컬럼을 하나씩 밀어 넣습니다
    for (j = 0; j < LK; j = j + 1) begin
      @(negedge clk); in_valid = 1; in_data = score[j];
      @(posedge clk);
    end
    @(negedge clk); in_valid = 0;

    cyc = 0;
    while (!done) begin
      @(posedge clk); cyc = cyc + 1;
      if (cyc > 200000) begin $display("=== TIMEOUT ==="); $finish; end
    end
    @(negedge clk); start = 0;
    repeat (4) @(posedge clk);
    $display("     %0d 사이클  (4·N·C ≈ %0d 예상)", cyc, 4*N*LK);

    for (j = 0; j < LK; j = j + 1)
      for (i = 0; i < N; i = i + 1) begin
        checks = checks + 1;
        // **1 LSB 허용.** 기대값은 골든의 *전수 LUT* softmax 이고, 지금 코어는
        // `SOFTMAX/` 의 exp2+역수 판이라 반올림 자리가 다릅니다. 1696개 중
        // 1개가 0 <-> 1 로 갈립니다 (확률 0.008 근처의 경계값).
        // 알고리즘이 다른 데서 오는 차이라 0 허용으로 두면 영원히 실패합니다 —
        // `tb_evt` 의 CTX 검사가 이 오차를 포함해 통과하는 것이 실제 근거입니다.
        if (diff(got[j][i*8 +: 8], gold[j][i*8 +: 8]) > 1) begin
          errors = errors + 1;
          if (errors < 6)
            $display("    [FAIL] key=%0d q=%0d  got %0d  exp %0d",
                     j, i, got[j][i*8 +: 8], gold[j][i*8 +: 8]);
        end else if (got[j][i*8 +: 8] !== gold[j][i*8 +: 8])
          tol = tol + 1;
      end
    $display("  softmax 출력  %0d/%0d 일치  (1 LSB 허용 %0d개, 초과 %0d개)",
             checks - errors - tol, checks, tol, errors);

    // 확률 합 — 127 로 정규화했으므로 쿼리마다 127 근처여야 합니다
    for (i = 0; i < N; i = i + 1) rowsum[i] = 0;
    for (j = 0; j < LK; j = j + 1)
      for (i = 0; i < N; i = i + 1) rowsum[i] = rowsum[i] + got[j][i*8 +: 8];
    $display("  확률 합 (쿼리 0,1,31) : %0d %0d %0d   (127 근처)",
             rowsum[0], rowsum[1], rowsum[31]);

    if (errors == 0) $display("=== tb_softmax_attn: %0d checks, TEST PASSED ===", checks);
    else             $display("=== tb_softmax_attn: %0d/%0d failed, TEST FAILED ===", errors, checks);
    $finish;
  end
endmodule
