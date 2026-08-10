// ============================================================================
//  fp32_to_fixed.v  --  IEEE-754 single(FP32) -> signed fixed-point 변환 (조합)
//    포맷 Q(W-1-QF).QF : value = q / 2^QF,  범위 [-2^(W-1-QF), +2^(W-1-QF))
//    q = round( value * 2^QF ), 범위 포화, e==0 -> 0, e==255(Inf/NaN) -> 포화
//    (QF=10 -> Q5.10, QF=11 -> Q4.11)
// ============================================================================
`timescale 1ns/1ps

module fp32_to_fixed #(
    parameter W  = 16,
    parameter QF = 11          // 소수 비트 (Q(W-1-QF).QF)
)(
    input      [31:0]        f,
    output reg signed [W-1:0] q
);
    localparam integer INTB = W - 1 - QF;   // 정수 비트: E >= INTB 이면 포화
    localparam integer SHB  = 23 - QF;      // rs = SHB - E
    localparam integer ZTH  = -(QF + 2);    // E <= ZTH 이면 0

    wire        s = f[31];
    wire [7:0]  e = f[30:23];
    wire [22:0] m = f[22:0];
    wire signed [10:0] E   = $signed({3'b0, e}) - 11'sd127;   // -127..128
    wire        [23:0] sig = {1'b1, m};                        // 1.m, 2^-23 단위

    reg        [W-1:0]  mag;
    reg  signed [10:0]  rs_s;
    reg        [5:0]    rs;
    reg        [26:0]   rnd, magw;
    reg                 satf;

    always @* begin
        satf = 1'b0;
        if (e == 8'd0) begin
            mag = {W{1'b0}};                              // zero / subnormal
        end else if (E >= INTB) begin
            mag = {1'b0, {(W-1){1'b1}}}; satf = 1'b1;     // |v|>=2^INTB 포화
        end else if (E <= ZTH) begin
            mag = {W{1'b0}};                              // 너무 작음 -> 0
        end else begin
            rs_s = SHB - E;                               // 양수
            rs   = rs_s[5:0];
            rnd  = {3'b0, sig} + ({26'd0, 1'b1} << (rs - 1'b1));
            magw = rnd >> rs;
            mag  = (magw >= (27'd1 << (W-1))) ? {1'b0, {(W-1){1'b1}}} : magw[W-1:0];
        end
        if (s) q = satf ? {1'b1, {(W-1){1'b0}}} : -$signed({1'b0, mag});
        else   q =                                 $signed({1'b0, mag});
    end
endmodule
