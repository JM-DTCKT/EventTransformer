// ============================================================================
//  tb_gelu_fp32.v  --  functional testbench for top (FP32 in / FP32 out)
//   * data/vectors_fp32.hex (FP32 입력) 스트리밍
//   * DUT(FP32 출력) <-> bit-accurate 모델(model_fp32_out.hex) 완전일치 확인
//   * DUT 출력을 build/dut_out_fp32.hex 로 덤프 (analyze.py / plot 용)
//   * 실수영역 max/RMS 오차는 analyze.py 에서 산출 (여기선 비트일치 검증)
// ============================================================================
`timescale 1ns/1ps

module tb_gelu_fp32;
    localparam integer MAXN = 40000;
    localparam integer LAT  = 5;        // gelu_fp32 파이프라인 지연

    reg         clk, rst_n, in_valid;
    reg  [31:0] x_fp32;
    wire        out_valid;
    wire [31:0] y_fp32;

    // ---- DUT (top module) ----
    top #(.W(16)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .x_fp32(x_fp32),
        .out_valid(out_valid), .y_fp32(y_fp32)
    );

    // ---- clock : 2ns ----
    initial clk = 1'b0;
    always #1 clk = ~clk;

    // ---- 메모리 ----
    reg [31:0] in_mem   [0:MAXN-1];
    reg [31:0] model_mem[0:MAXN-1];
    reg [31:0] dut_mem  [0:MAXN-1];

    integer nvf, code, i, N, ci, nmiss;

    // ---- 출력 캡처 (in_valid 연속 => i번째 valid == i번째 입력) ----
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            dut_mem[ci] = y_fp32;
            ci = ci + 1;
        end
    end

    // ---- latency / throughput 실측 모니터 ----
    integer cyc, t_in0, t_out0, t_outL, vcnt;
    initial begin cyc = 0; t_in0 = -1; t_out0 = -1; t_outL = -1; vcnt = 0; end
    always @(posedge clk) begin
        if (rst_n) begin
            cyc = cyc + 1;
            if (in_valid && t_in0 < 0) t_in0 = cyc;         // 첫 입력 사이클
            if (out_valid) begin
                if (t_out0 < 0) t_out0 = cyc;               // 첫 출력 사이클
                t_outL = cyc;                               // 마지막 출력 사이클
                vcnt = vcnt + 1;                            // out_valid high 누적
            end
        end
    end

    integer fout;
    initial begin
        // 벡터 개수 + 입력/모델 로드
        nvf = $fopen("data/nvec.txt", "r");
        if (nvf == 0) begin $display("ERROR: cannot open data/nvec.txt"); $finish; end
        code = $fscanf(nvf, "%d", N); $fclose(nvf);
        $readmemh("data/vectors_fp32.hex", in_mem);
        $readmemh("data/model_fp32_out.hex", model_mem);
        $display("[TB] N = %0d FP32 vectors", N);

        // 초기화 & 리셋
        ci = 0; in_valid = 0; x_fp32 = 0; rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // 스트리밍 구동
        for (i = 0; i < N; i = i + 1) begin
            x_fp32   <= in_mem[i];
            in_valid <= 1'b1;
            @(posedge clk);
        end
        in_valid <= 1'b0;
        x_fp32   <= 0;

        wait (ci == N);
        repeat (LAT + 2) @(posedge clk);

        // ---- DUT vs bit-accurate 모델 (완전일치 기대) + 덤프 ----
        nmiss = 0;
        fout = $fopen("build/dut_out_fp32.hex", "w");
        for (i = 0; i < N; i = i + 1) begin
            $fwrite(fout, "%08x\n", dut_mem[i]);
            if (dut_mem[i] !== model_mem[i]) begin
                if (nmiss < 10)
                    $display("  MISMATCH[%0d]: in=%08x dut=%08x model=%08x",
                             i, in_mem[i], dut_mem[i], model_mem[i]);
                nmiss = nmiss + 1;
            end
        end
        $fclose(fout);

        $display("=======================================================");
        $display(" GELU FP32 (Residual-PWL core) functional check (N=%0d)", N);
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
