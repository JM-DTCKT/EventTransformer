// -----------------------------------------------------------------------------
// Gelu_Pwl : GELU — 잔차(residual) 형태 + 균일 PWL-LUT
//
//   G(x) = x * Phi(x)          Phi = 표준정규 CDF
//
// GELU 를 그대로 근사하지 않고 **잔차** R(a) 를 근사합니다 (a = |x| >= 0):
//
//   R(a) = a - G(a) = a*(1 - Phi(a)) = (a/2)*erfc(a/sqrt2)
//   x >= 0 : G(x) = x - R(a)
//   x <  0 : G(x) =   - R(a)                 ( G(-a) = -R(a) )
//
// R(a) 는 0 <= R <= 0.17 로 작고 유계이며 a > 4 부근에서 0 으로 사라지는 혹이라,
// **균일 구간 선형보간**으로 싸게 맞출 수 있습니다. 전수 LUT(레인당 8 BRAM36)
// 대신 base/delta 64쌍 = 2 Kb 로 끝나고, 오차는 최대 1 LSB 입니다.
//
// ## 고정소수점
//
//   x, y : signed Q(W-1-QF).QF
//   R    : 내부 Q(FR)          (base14 / delta14 는 `Gelu_Lut.vh`)
//   a in [0,4) → seg = a[11:6] (64 구간, 폭 2^-4),  frac = a[5:0]
//   R14 = base14[seg] + (delta14[seg]*frac >>> 6)          (선형보간)
//   a >= 4     → R = 0        (자연 포화 : x>=0 이면 G=x, x<0 이면 G=0)
//
// 파이프라인 3단 (입력 레지스터 → LUT+보간 → 합성).
// **리셋만 active-low(`rst_n`)** 입니다 — 프로젝트 나머지는 active-high 입니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Gelu_Pwl #(
    parameter W  = 16,   // 데이터 폭
    parameter QF = 11,   // x/y 의 소수부 비트수  (Q(W-1-QF).QF)
    parameter FR = 14    // 내부 R 의 소수부 비트수
)(
    input                     clk,
    input                     rst_n,
    input                     in_valid,
    input      signed [W-1:0] x,        // Q(W-1-QF).QF
    output reg                out_valid,
    output reg signed [W-1:0] y         // Q(W-1-QF).QF  ~= GELU(x)
);
    localparam integer K        = 4;               // 구간 폭 = 2^-K = 0.0625
    localparam integer A_INT    = 4;               // A_MAX = 4.0
    localparam integer NSEG     = A_INT << K;      // = 64  (A_MAX/width), QF 무관
    localparam integer IDXBITS  = $clog2(NSEG);    // = 6   (구간 인덱스 비트수)
    localparam integer FRACBITS = QF - K;          // 구간 안 위치 비트수 (QF-4)
    localparam integer A_MAX_FIX= (A_INT << QF);   // a>=4 -> R(a)=0
    localparam integer RSH      = FR - QF;          // Q(FR) -> Q(QF)

    // ---- LUT (base14_rom[64], delta14_rom[64]) ----
    `include "Gelu_Lut.vh"

    // ======================= Stage 1 : 디코드 + LUT 조회 ====================
    // a = |x|  (|-32768| = 32768 을 담으려면 17비트)
    wire signed [W:0]  x_ext = {x[W-1], x};
    wire        [W:0]  a     = x[W-1] ? (~x_ext + 1'b1) : x_ext;   // |x|, unsigned
    wire               ge4   = (a >= A_MAX_FIX);                   // a >= 4.0

    wire [IDXBITS-1:0]  seg  = a[FRACBITS+IDXBITS-1 : FRACBITS];   // a[11:6]
    wire [FRACBITS-1:0] frac = a[FRACBITS-1 : 0];                  // a[5:0]

    reg                 vld_s1, sign_s1, ge4_s1;
    reg  signed [W-1:0] x_s1;
    reg  signed [W-1:0] base_s1, delta_s1;
    reg  [FRACBITS-1:0] frac_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_s1 <= 1'b0; sign_s1 <= 1'b0; ge4_s1 <= 1'b0;
            x_s1 <= 0; base_s1 <= 0; delta_s1 <= 0; frac_s1 <= 0;
        end else begin
            vld_s1   <= in_valid;
            x_s1     <= x;
            sign_s1  <= x[W-1];
            ge4_s1   <= ge4;
            frac_s1  <= frac;
            base_s1  <= base14_rom [seg];
            delta_s1 <= delta14_rom[seg];
        end
    end

    // ======================= Stage 2 : 선형보간 =============================
    // interp = delta*frac >>> FRACBITS  (frac 은 세그먼트를 2^FRACBITS 등분한 위치)
    // r14 = base + interp   (R in Q(FR)=Q14) ;  r_out = round(r14) -> Q(QF)
    wire signed [W+FRACBITS:0] prod   = delta_s1 * $signed({1'b0, frac_s1});
    wire signed [W:0]          interp = prod >>> FRACBITS;                 // R in Q(FR)
    wire signed [W:0]          r14    = ge4_s1 ? {(W+1){1'b0}} : (base_s1 + interp);
    wire signed [W:0]          r_out  = (r14 + (1 << (RSH-1))) >>> RSH;    // Q(FR)->Q(QF), round

    reg                 vld_s2, sign_s2;
    reg  signed [W-1:0] x_s2;
    reg  signed [W-1:0] r_out_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_s2 <= 1'b0; sign_s2 <= 1'b0; x_s2 <= 0; r_out_s2 <= 0;
        end else begin
            vld_s2   <= vld_s1;
            sign_s2  <= sign_s1;
            x_s2     <= x_s1;
            r_out_s2 <= r_out[W-1:0];
        end
    end

    // ======================= Stage 3 : 합성 =================================
    //   x >= 0 : y = x - R ;   x < 0 : y = -R   (모두 Q(QF))
    wire signed [W-1:0] y_next = sign_s2 ? (-r_out_s2) : (x_s2 - r_out_s2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0; y <= 0;
        end else begin
            out_valid <= vld_s2;
            y         <= y_next;
        end
    end

endmodule
