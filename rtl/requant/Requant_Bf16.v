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
// 파이프라인 **2단** (in_valid → out_valid 2클럭) :
//
//   S0 : 부호 · 절댓값 · 정확 곱          -> prod_q   (DSP48E2 의 P 레지스터로 흡수)
//   S1 : MSB 탐색 · 정규화 · 반올림 · 지수 -> out
//
// 전부 한 사이클에 두면 `|acc| -> 41b 곱(DSP 2개) -> 41b LZC -> 41b 배럴 시프트`
// 가 직렬로 놓여 ZU9EG -2 에서 **8.475 ns** 였습니다 (150 MHz 에서 WNS -1.763 ns,
// `Format_Cast_Act` 의 `a_b` 가산까지 포함한 실측).  곱 뒤에서 끊으면 4.365 / 3.995 로
// 거의 반씩 갈립니다.  자르는 자리가 곱 **뒤**인 이유는 이 모듈의 무게가 뒤쪽
// (LZC + 정규화)에 실려 있기 때문입니다 — 앞을 끊는 `Requant_Int` 의 `PIPE_PRE`
// 와 반대입니다.
//
// 지연이 1 -> 2 로 늘었으므로 `EvT_Engine` 의 **`CP_LAT_B` 도 2** 여야 합니다.
// (쓰기 주소 파이프가 어긋나면 결과가 한 칸 앞 주소에 써집니다.)
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

    reg  [PW-1:0]    norm;
    integer          i, msb;
    reg  signed [11:0] e;
    reg  [6:0]  mant;
    reg         gbit, sbit, rup;
    reg  [7:0]  mant_r;
    reg  [15:0] y;

    // ======================= S0 : 부호 · 절댓값 · 곱 (조합) =======================
    wire             sgn_c  = acc[ACC_W-1] ^ sgn_s;
    wire [ACC_W-1:0] mag_c  = acc[ACC_W-1] ? (~acc + 1'b1) : acc;  // |acc|
    wire [PW-1:0]    prod_c = mag_c * {1'b1, man_s};               // 정확한 곱, 반올림 없음
    wire             zero_c = (mag_c == {ACC_W{1'b0}}) || (exp_s == 8'd0);

    // ---- 파이프 레지스터 (DSP48E2 의 P 레지스터로 흡수) ----
    // `prod_q` 말고 **부호 · 지수 · zero 판정도 같이** 넘겨야 합니다.  S1 이 쓰는데
    // S0 에만 있는 값들이라, 빠뜨리면 파이프가 어긋나 조용히 틀립니다.
    reg [PW-1:0] prod_q;
    reg          sgn_q, zero_q;
    reg [7:0]    exp_q;
    reg          v_q;
    always @(posedge clk) begin
        if (rst) begin
            v_q <= 1'b0;
        end else begin
            v_q <= in_valid;
            if (in_valid) begin
                prod_q <= prod_c;
                sgn_q  <= sgn_c;
                zero_q <= zero_c;
                exp_q  <= exp_s;
            end
        end
    end

    // ============ S1 : MSB 탐색 · 정규화 · 반올림 · 지수 (조합) ============
    always @* begin
        if (zero_q) begin
            y = {sgn_q, 15'd0};                        // ±0
        end else begin
            msb  = 0;
            for (i = 0; i < PW; i = i + 1)
                if (prod_q[i]) msb = i;
            norm = prod_q << (PW-1 - msb);             // MSB 를 최상위로 정렬

            mant = norm[PW-2 -: 7];
            gbit = norm[PW-9];
            sbit = |norm[PW-10:0];
            rup  = gbit & (sbit | mant[0]);            // round-to-nearest-even
            mant_r = {1'b0, mant} + rup;

            e = msb + $signed({4'd0, exp_q}) - 12'sd7;
            if (mant_r[7]) begin                       // 가수 올림 넘침 → 지수 +1
                mant_r = 8'd0;
                e = e + 12'sd1;
            end

            if (e <= 12'sd0)        y = {sgn_q, 15'd0};          // underflow → ±0
            else if (e >= 12'sd255) y = {sgn_q, 8'hFF, 7'd0};    // overflow  → ±inf
            else                    y = {sgn_q, e[7:0], mant_r[6:0]};
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; out <= 16'd0;
        end else begin
            out_valid <= v_q;
            if (v_q) out <= y;
        end
    end
endmodule
