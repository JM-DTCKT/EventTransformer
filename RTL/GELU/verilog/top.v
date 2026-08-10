// ============================================================================
//  top.v  --  FP32 인터페이스 GELU 유닛 (top module)
//             내부는 Residual + PWL fixed-point (Q4.11, QF=11) 로 처리
//
//    x_fp32 ─▶[FP32→Q4.11]─▶ gelu_pwl(core, Q4.11, latency 3) ─▶[Q4.11→FP32]─▶ y_fp32
//              └ Stage A ┘                                        └ Stage E ┘
//
//  Stage A(FP32→고정소수점), Stage E(고정소수점→FP32) 는 조합 변환기다.
//  각 단에 파이프라인 레지스터를 넣을지는 REG_IN / REG_OUT 로 선택한다.
//    REG_IN=0, REG_OUT=0 : 조합 (latency = 3)          <- 기본
//    REG_IN=1, REG_OUT=1 : 레지스터 (latency = 5, Fmax↑)
//    혼합 시 latency = 3 + REG_IN + REG_OUT
//  어느 경우든 throughput 은 1 결과/clock 로 동일.
// ============================================================================
`timescale 1ns/1ps

module top #(
    parameter W       = 16,   // 내부 고정소수점 폭
    parameter QF      = 11,   // 입출력 소수 비트 (Q(W-1-QF).QF = Q4.11)
    parameter REG_IN  = 0,    // 1: Stage A 뒤 입력 레지스터 사용 (latency +1)
    parameter REG_OUT = 0     // 1: Stage E 뒤 출력 레지스터 사용 (latency +1)
)(
    input             clk,
    input             rst_n,
    input             in_valid,
    input  [31:0]     x_fp32,     // IEEE-754 single
    output            out_valid,
    output [31:0]     y_fp32       // IEEE-754 single ~= GELU(x)
);
    // ---------------- Stage A : FP32 -> Q(W-1-QF).QF (조합) ----------------
    wire signed [W-1:0] x_fix_c;
    fp32_to_fixed #(.W(W), .QF(QF)) u_in (.f(x_fp32), .q(x_fix_c));

    wire signed [W-1:0] x_fix;
    wire                va;
    generate
        if (REG_IN) begin : g_in                 // 레지스터 삽입
            reg signed [W-1:0] xr; reg vr;
            always @(posedge clk or negedge rst_n)
                if (!rst_n) begin xr <= {W{1'b0}}; vr <= 1'b0; end
                else        begin xr <= x_fix_c;   vr <= in_valid; end
            assign x_fix = xr; assign va = vr;
        end else begin : g_in                    // 조합 통과
            assign x_fix = x_fix_c; assign va = in_valid;
        end
    endgenerate

    // ---------------- Core : Q5.10 Residual-PWL GELU (latency 3) ----------------
    wire                cov;
    wire signed [W-1:0] y_fix;
    gelu_pwl #(.W(W), .QF(QF), .FR(14)) u_core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(va), .x(x_fix),
        .out_valid(cov), .y(y_fix)
    );

    // ---------------- Stage E : Q(W-1-QF).QF -> FP32 (조합) ----------------
    wire [31:0] y_fp32_c;
    fixed_to_fp32 #(.W(W), .QF(QF)) u_out (.q(y_fix), .f(y_fp32_c));

    generate
        if (REG_OUT) begin : g_out               // 레지스터 삽입
            reg [31:0] yr; reg vr;
            always @(posedge clk or negedge rst_n)
                if (!rst_n) begin yr <= 32'd0; vr <= 1'b0; end
                else        begin yr <= y_fp32_c; vr <= cov; end
            assign y_fp32 = yr; assign out_valid = vr;
        end else begin : g_out                   // 조합 통과
            assign y_fp32 = y_fp32_c; assign out_valid = cov;
        end
    endgenerate
endmodule
