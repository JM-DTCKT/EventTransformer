// ============================================================================
//  fp32_to_q510.v  --  IEEE-754 single(FP32) -> signed Q5.10 (16-bit) 변환 (조합)
//    value = (-1)^s * 1.m * 2^(e-127)
//    q     = round( value * 2^10 ),  포화 [-32768(-32.0), 32767(+31.999)]
//    특수  : e==0(zero/subnormal) -> 0,  e==255(Inf/NaN) -> 포화
//    반올림: round-half-up (더하기 0.5LSB 후 절삭)
// ============================================================================
`timescale 1ns/1ps

module fp32_to_q510 (
    input      [31:0]        f,     // FP32
    output reg signed [15:0] q      // Q5.10
);
    wire        s = f[31];
    wire [7:0]  e = f[30:23];
    wire [22:0] m = f[22:0];

    // 지수(바이어스 제거).  E in [-127, 128]
    wire signed [10:0] E   = $signed({3'b0, e}) - 11'sd127;
    wire        [23:0] sig = {1'b1, m};            // 1.m (implicit 1 포함), 2^-23 단위

    reg        [15:0]  mag;      // |q| (0..32767)
    reg  signed [10:0] rs_s;
    reg        [4:0]   rs;
    reg        [26:0]  rnd, magw;
    reg                satf;     // |value| >= 32 로 포화된 경우

    always @* begin
        satf = 1'b0;
        if (e == 8'd0) begin
            mag = 16'd0;                          // zero / subnormal -> 0
        end else if (E >= 11'sd5) begin
            mag = 16'd32767; satf = 1'b1;         // |v| >= 32 -> 포화 (Inf/NaN 포함)
        end else if (E <= -11'sd12) begin
            mag = 16'd0;                          // |v| < ~2^-11 -> 0
        end else begin
            // mag = round( sig * 2^(E-13) ) = round( sig >> (13-E) )
            rs_s = 11'sd13 - E;                   // 9..24 (E in [-11,4])
            rs   = rs_s[4:0];
            rnd  = {3'b0, sig} + ({26'd0, 1'b1} << (rs - 1'b1));
            magw = rnd >> rs;
            mag  = (magw >= 27'd32768) ? 16'd32767 : magw[15:0];
        end
        // 부호 적용 (음수 포화는 -32768 = -32.0 허용)
        if (s) q = satf ? -16'sd32768 : -$signed({1'b0, mag});
        else   q =                       $signed({1'b0, mag});
    end
endmodule
