// -----------------------------------------------------------------------------
// Requant_Bf16 : INT32 누산기 → bfloat16
//
//   out = bf16( acc * scale[c] )      scale[c] = bf16(s_x * s_w[c])
//
// LayerNorm 으로 들어가는 6개 GEMM(out_proj ×3, linear3 ×3)이 쓰는 경로입니다.
// LayerNorm 의 입력 레지스터가 bf16 이므로 그 생산자는 Qm.n 격자로 스냅하지
// 않고 bf16 으로 변환합니다 — 정수 시프트도, 포화도 없습니다.
//
// **반올림은 한 번만** 합니다. `Int32_To_Bf16` 로 먼저 줄인 뒤 곱하면 acc 의
// 하위 비트를 곱하기도 전에 버리게 되므로, 여기서는 정수 누산기와 scale 의
// 가수를 **정확히** 곱한 뒤(ACC_W+8 비트) 그 결과를 한 번만 bf16 으로
// 반올림합니다. 반올림 방식은 IEEE 기본인 round-to-nearest-even.
//
//   value = |acc| * (1.ms * 2^(es-127))
//         = (|acc| * {1,ms}) * 2^(es-127-7)
//   prod = |acc| * {1,ms} 의 MSB 위치를 p 라 하면 prod = 1.f * 2^p 이므로
//   bf16 지수 = (p + es - 127 - 7) + 127 = p + es - 7
//
// 조합 논리 + 출력 1단 레지스터 (in_valid → out_valid 1클럭).
// scale 의 지수가 0 이면 flush-to-zero (denormal 미지원 — 상수는 전부 정규수).
// -----------------------------------------------------------------------------
module Requant_Bf16 #(
    parameter ACC_W = 32
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    in_valid,
    input  wire signed [ACC_W-1:0] acc,
    input  wire        [15:0]      scale,      // bf16 상수 s_x*s_w[c]
    output reg                     out_valid,
    output reg         [15:0]      out         // bf16
);
    localparam integer PW = ACC_W + 8;         // 정확한 곱 폭

    wire        sgn_s = scale[15];
    wire [7:0]  exp_s = scale[14:7];
    wire [6:0]  man_s = scale[6:0];

    reg  [ACC_W-1:0] mag;
    reg  [PW-1:0]    prod, norm;
    reg              sgn;
    integer          i, msb;
    reg  signed [11:0] e;
    reg  [6:0]  mant;
    reg         gbit, sbit, rup;
    reg  [7:0]  mant_r;
    reg  [15:0] y;

    always @* begin
        sgn = acc[ACC_W-1] ^ sgn_s;
        mag = acc[ACC_W-1] ? (~acc + 1'b1) : acc;      // |acc| (-2^(W-1) 은 그대로)

        if ((mag == {ACC_W{1'b0}}) || (exp_s == 8'd0)) begin
            y = {sgn, 15'd0};                          // ±0
        end else begin
            prod = mag * {1'b1, man_s};                // 정확한 곱, 반올림 없음
            msb  = 0;
            for (i = 0; i < PW; i = i + 1)
                if (prod[i]) msb = i;
            norm = prod << (PW-1 - msb);               // MSB 를 최상위로 정렬

            mant = norm[PW-2 -: 7];
            gbit = norm[PW-9];
            sbit = |norm[PW-10:0];
            rup  = gbit & (sbit | mant[0]);            // round-to-nearest-even
            mant_r = {1'b0, mant} + rup;

            e = msb + $signed({4'd0, exp_s}) - 12'sd7;
            if (mant_r[7]) begin                       // 가수 올림 넘침 → 지수 +1
                mant_r = 8'd0;
                e = e + 12'sd1;
            end

            if (e <= 12'sd0)        y = {sgn, 15'd0};            // underflow → ±0
            else if (e >= 12'sd255) y = {sgn, 8'hFF, 7'd0};      // overflow  → ±inf
            else                    y = {sgn, e[7:0], mant_r[6:0]};
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; out <= 16'd0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) out <= y;
        end
    end
endmodule
