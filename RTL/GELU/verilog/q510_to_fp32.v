// ============================================================================
//  q510_to_fp32.v  --  signed Q5.10 (16-bit) -> IEEE-754 single(FP32) 변환 (조합)
//    value = q * 2^-10.   |q| <= 32768 이라 유효비트 <= 16 < 23 => 무손실(반올림 없음)
//    q==0 -> +0.0.  정규화: mag = 1.f * 2^msb, exp = (msb-10)+127 = msb+117
// ============================================================================
`timescale 1ns/1ps

module q510_to_fp32 (
    input  signed [15:0] q,      // Q5.10
    output reg    [31:0] f       // FP32
);
    wire        s   = q[15];
    wire signed [16:0] qext = {q[15], q};
    wire        [16:0] neg  = (~qext + 1'b1);        // -q (17-bit, q=-32768 -> 32768)
    wire        [15:0] mag  = s ? neg[15:0] : q[15:0];

    integer i;
    reg [4:0]  msb;
    reg [39:0] magsh;
    reg [7:0]  exp;

    always @* begin
        if (q == 16'sd0) begin
            f = 32'h0000_0000;                       // +0.0
        end else begin
            msb = 5'd0;
            for (i = 0; i < 16; i = i + 1)
                if (mag[i]) msb = i[4:0];            // 최상위 '1' 위치
            exp   = msb + 8'd117;                    // (msb-10)+127
            magsh = ({24'd0, mag} << (6'd23 - msb)); // implicit 1 을 bit23 로 정렬
            f = {s, exp, magsh[22:0]};               // bit23(implicit 1) 제거
        end
    end
endmodule
