// ============================================================================
//  tb_top.v  --  functional testbench for top (Q4.11 고정소수점 in/out)
//   * data/vectors.hex (int16 Q4.11 입력) 전수(65536) 스트리밍
//   * DUT(Q4.11 출력) <-> bit-accurate 모델(model_out.hex) 완전일치 확인
//   * latency / throughput 실측
//   * DUT 출력을 build/dut_out.hex 로 덤프 (analyze.py / plot 용)
// ============================================================================
`timescale 1ns/1ps

module tb_top;
    localparam integer W    = 16;
    localparam integer MAXN = 70000;
    localparam integer LATMAX = 8;

    reg                 clk, rst_n, in_valid;
    reg  signed [W-1:0] x;
    wire                out_valid;
    wire signed [W-1:0] y;

    // ---- DUT (top module, Q4.11 in/out) ----
    top #(.W(W), .QF(11)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .x(x),
        .out_valid(out_valid), .y(y)
    );

    // ---- clock : 2ns ----
    initial clk = 1'b0;
    always #1 clk = ~clk;

    // ---- 메모리 ----
    reg [W-1:0] in_mem   [0:MAXN-1];
    reg [W-1:0] model_mem[0:MAXN-1];
    reg [W-1:0] dut_mem  [0:MAXN-1];

    integer nvf, code, i, N, ci, nmiss;

    // ---- 출력 캡처 (in_valid 연속 => i번째 valid == i번째 입력) ----
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            dut_mem[ci] = y;
            ci = ci + 1;
        end
    end

    // ---- latency / throughput 실측 모니터 ----
    integer cyc, t_in0, t_out0, t_outL, vcnt;
    initial begin cyc = 0; t_in0 = -1; t_out0 = -1; t_outL = -1; vcnt = 0; end
    always @(posedge clk) begin
        if (rst_n) begin
            cyc = cyc + 1;
            if (in_valid && t_in0 < 0) t_in0 = cyc;
            if (out_valid) begin
                if (t_out0 < 0) t_out0 = cyc;
                t_outL = cyc;
                vcnt = vcnt + 1;
            end
        end
    end

    integer fout;
    initial begin
        nvf = $fopen("data/nvec.txt", "r");
        if (nvf == 0) begin $display("ERROR: cannot open data/nvec.txt"); $finish; end
        code = $fscanf(nvf, "%d", N); $fclose(nvf);
        $readmemh("data/vectors.hex",   in_mem);
        $readmemh("data/model_out.hex", model_mem);
        $display("[TB] N = %0d Q4.11 vectors", N);

        ci = 0; in_valid = 0; x = 0; rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        for (i = 0; i < N; i = i + 1) begin
            x        <= in_mem[i];
            in_valid <= 1'b1;
            @(posedge clk);
        end
        in_valid <= 1'b0;
        x        <= 0;

        wait (ci == N);
        repeat (LATMAX) @(posedge clk);

        // ---- DUT vs bit-accurate 모델 + 덤프 ----
        nmiss = 0;
        fout = $fopen("build/dut_out.hex", "w");
        for (i = 0; i < N; i = i + 1) begin
            $fwrite(fout, "%04x\n", dut_mem[i]);
            if (dut_mem[i] !== model_mem[i]) begin
                if (nmiss < 10)
                    $display("  MISMATCH[%0d]: in=%04x dut=%04x model=%04x",
                             i, in_mem[i], dut_mem[i], model_mem[i]);
                nmiss = nmiss + 1;
            end
        end
        $fclose(fout);

        $display("=======================================================");
        $display(" GELU Q4.11 (Residual-PWL core) functional check (N=%0d)", N);
        $display("-------------------------------------------------------");
        $display(" RTL(top) vs bit-accurate model : %0d mismatch  (%s)",
                 nmiss, (nmiss==0)?"PASS":"FAIL");
        $display("-------------------------------------------------------");
        $display(" latency    : %0d cycle  (first in_valid -> first out_valid)",
                 t_out0 - t_in0);
        $display(" throughput : out_valid high for %0d cycles over a %0d-cycle span",
                 vcnt, t_outL - t_out0 + 1);
        if (vcnt == N && (t_outL - t_out0 + 1) == N)
            $display("              => 1 output / clock, NO bubbles  (PASS)");
        else
            $display("              => throughput CHECK FAIL (bubbles/stall)");
        $display("=======================================================");
        $display(" (실수영역 max/RMS 오차는 analyze.py 로 산출)");
        if (nmiss != 0) $display(" *** WARNING: RTL 이 SW 모델과 불일치 ***");
        $finish;
    end

    initial begin
        #4000000;
        $display("ERROR: timeout");
        $finish;
    end
endmodule
