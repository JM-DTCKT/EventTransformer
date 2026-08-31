// -----------------------------------------------------------------------------
// Int32_To_Bf16 : 2의 보수 정수 → bfloat16 (조합 논리)
//
//   bf16 = 부호1 + 지수8 + 가수7  (IEEE fp32 의 상위 16비트와 같은 배치)
//   반올림 round-to-nearest-even.
//
// LayerNorm 앞단(입력·mean·centered·var)이 bf16 이라 정수 도메인에서 넘어오는
// 값을 bf16 으로 올릴 때 씁니다. GEMM 누산기 → bf16 경로는 곱과 반올림을 융합한
// `Requant_Bf16` 을 쓰세요 (반올림 1회, 이쪽은 2회가 됨).
//
// IN_W >= 10 이어야 합니다 (가수7 + guard1 + sticky 최소 1비트 + MSB).
// -----------------------------------------------------------------------------
module Int32_To_Bf16 #(
    parameter IN_W = 32
)(
    input  wire signed [IN_W-1:0] din,
    output reg         [15:0]     bf16
);
    reg [IN_W-1:0] mag, norm;
    integer        i, msb;
    reg  [6:0]     mant;
    reg            gbit, sbit, rup;
    reg  [7:0]     mant_r;
    reg  signed [10:0] e;

    always @* begin
        mag = din[IN_W-1] ? (~din + 1'b1) : din;       // |din|

        if (mag == {IN_W{1'b0}}) begin
            bf16 = 16'h0000;
        end else begin
            msb = 0;
            for (i = 0; i < IN_W; i = i + 1)
                if (mag[i]) msb = i;
            norm = mag << (IN_W-1 - msb);              // MSB 를 최상위로

            mant = norm[IN_W-2 -: 7];
            gbit = norm[IN_W-9];
            sbit = |norm[IN_W-10:0];
            rup  = gbit & (sbit | mant[0]);
            mant_r = {1'b0, mant} + rup;

            e = 11'sd127 + msb;
            if (mant_r[7]) begin mant_r = 8'd0; e = e + 11'sd1; end

            if (e >= 11'sd255) bf16 = {din[IN_W-1], 8'hFF, 7'd0};   // ±inf
            else               bf16 = {din[IN_W-1], e[7:0], mant_r[6:0]};
        end
    end
endmodule
