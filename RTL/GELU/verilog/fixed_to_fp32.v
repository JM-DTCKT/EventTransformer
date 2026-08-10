// ============================================================================
//  fixed_to_fp32.v  --  signed fixed-point Q(W-1-QF).QF -> IEEE-754 single (조합)
//    value = q * 2^-QF.  유효비트 <= W-1 < 23 이라 무손실(반올림 없음).
//    q==0 -> +0.0.  정규화: exp = msb + (127 - QF)
//    (QF=10 -> Q5.10, QF=11 -> Q4.11)
// ============================================================================
`timescale 1ns/1ps

module fixed_to_fp32 #(
    parameter W  = 16,
    parameter QF = 11
)(
    input  signed [W-1:0] q,
    output reg    [31:0]  f
);
    localparam integer EXPBIAS = 127 - QF;      // exp = msb + EXPBIAS

    wire        s    = q[W-1];
    wire signed [W:0]   qext = {q[W-1], q};
    wire        [W:0]   neg  = (~qext + 1'b1);           // -q
    wire        [W-1:0] mag  = s ? neg[W-1:0] : q[W-1:0];

    integer i;
    reg [5:0]    msb;
    reg [W+23:0] magsh;
    reg [7:0]    exp;

    always @* begin
        if (q == {W{1'b0}}) begin
            f = 32'h0000_0000;                            // +0.0
        end else begin
            msb = 6'd0;
            for (i = 0; i < W; i = i + 1)
                if (mag[i]) msb = i[5:0];                 // 최상위 '1'
            exp   = msb + EXPBIAS;
            magsh = ({{24{1'b0}}, mag} << (6'd23 - msb)); // implicit 1 을 bit23 로
            f = {s, exp, magsh[22:0]};
        end
    end
endmodule
