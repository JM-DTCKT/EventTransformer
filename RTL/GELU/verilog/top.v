// ============================================================================
//  top.v  --  GELU 유닛 (Q4.11 고정소수점 직접 입출력)
//
//    입력 x, 출력 y : signed Q4.11 (16-bit, value = x / 2^11, 범위 [-16, 16))
//    내부 : Residual + PWL-LUT 코어(gelu_pwl). FP32 변환 없음.
//    latency = 3 clock,  throughput = 1 결과/clock (파이프라인).
// ============================================================================
`timescale 1ns/1ps

module top #(
    parameter W  = 16,   // 데이터 폭
    parameter QF = 11    // 소수 비트 (Q(W-1-QF).QF = Q4.11)
)(
    input                     clk,
    input                     rst_n,
    input                     in_valid,
    input      signed [W-1:0] x,        // Q4.11 입력
    output                    out_valid,
    output     signed [W-1:0] y         // Q4.11 출력  ~= GELU(x)
);
    // Residual + PWL GELU 코어 (Q4.11 in/out, latency 3)
    gelu_pwl #(.W(W), .QF(QF), .FR(14)) u_core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .x(x),
        .out_valid(out_valid), .y(y)
    );
endmodule
