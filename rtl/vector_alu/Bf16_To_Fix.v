// -----------------------------------------------------------------------------
// Bf16_To_Fix : BFloat16 -> signed 고정소수점 Q(IW-1-IF).IF 변환 (조합)
//
// BF16 = FP32 의 상위 16비트 : [15] sign, [14:7] exp(bias 127), [6:0] mantissa
//
//    value = (-1)^s * 1.mmmmmmm * 2^(E-127)
//          = (-1)^s * M * 2^(E-134)        ,  M = {1'b1, m[6:0]} in [128, 255]
//
// 고정소수점 목표값은 x = value * 2^IF * 2^xsh 이므로
//
//    |x| = M * 2^sh ,   sh = E - 134 + IF + xsh
//
// 즉 **가수 8비트를 sh 만큼 시프트**하는 것이 전부다.  곱셈기가 필요 없고,
// sh >= 0 이면 무손실(좌시프트), sh < 0 이면 반올림(우시프트) 이다.
//
// ## xsh
//
// LayerNorm 출력은 입력 스케일에 불변(LN(a*x) = LN(x)) 이므로 xsh 로 고정소수점
// 창을 통째로 옮겨도 결과가 바뀌지 않는다. 활성값 분포가 기본 Q8.15 창을 벗어날
// 때 호스트가 조절하는 용도다.
//
// ## 경계
//
// IW=24, IF=15 기준
//    sh >= 16   -> |x| >= 2^23  포화 (|value| >= 2^8 = 256),  ovf=1
//    sh <  -8   -> |x| <  0.5 LSB 이므로 0
//    E == 0     -> zero/subnormal -> 0     (BF16 subnormal 은 2^-133 이하)
//    E == 255   -> Inf/NaN        -> 포화, ovf=1
//
// 반올림은 round-half-up (부호는 마지막에 적용하므로 0 기준 대칭이다).
// 포화값은 +-(2^(IW-1)-1) 로 **대칭** 이라 -2^(IW-1) 은 절대 나오지 않는다
// -> 이후 X*X 가 2^(2IW-2) 를 넘지 않음이 보장된다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Bf16_To_Fix #(
    parameter integer IW  = 24,   // 출력 폭 (signed Q8.15)
    parameter integer IF  = 15,   // 출력 소수부 비트수
    parameter integer XSW = 6     // xsh 폭 (signed)
)(
    input         [15:0]      bf,     // BF16 입력
    input  signed [XSW-1:0]   xsh,    // 추가 지수 시프트 (value * 2^xsh)
    output signed [IW-1:0]    x,      // signed Q(IW-1-IF).IF
    output                    ovf     // 포화 발생
);
    localparam integer        SHW    = 12;                  // sh 폭 (signed)
    localparam signed [SHW-1:0] SH_OFS = IF - 134;          // = -119 (IF=15)
    localparam signed [SHW-1:0] SH_HI  = IW - 8;            // = 16, 이상이면 포화
    localparam signed [SHW-1:0] SH_LO  = -8;                // 미만이면 0
    localparam        [IW-2:0]  MAXMAG = {(IW-1){1'b1}};    // 대칭 포화값 2^23-1
    localparam integer        WW     = IW + 9;              // 시프트 중간값 폭

    wire       s = bf[15];
    wire [7:0] e = bf[14:7];
    wire [7:0] M = {1'b1, bf[6:0]};              // 1.m 을 2^7 배한 정수 [128,255]

    wire signed [SHW-1:0] e_s   = $signed({4'b0, e});
    wire signed [SHW-1:0] xsh_s = {{(SHW-XSW){xsh[XSW-1]}}, xsh};
    wire signed [SHW-1:0] sh    = e_s + SH_OFS + xsh_s;

    wire is_zero = (e == 8'd0);                  // zero / subnormal
    wire is_inf  = (e == 8'd255);                // Inf / NaN
    wire hi      = (sh >= SH_HI);                // 포화
    wire lo      = (sh <  SH_LO);                // 언더플로 -> 0
    wire bypass  = hi | lo | is_zero | is_inf;   // 시프터를 쓰지 않는 경우

    // ---- 시프트 : sh in [-8, IW-9] 를 (sh+8) in [0, IW-1] 좌시프트로 통일 ----
    //  wide  = M << (sh+8)          <- M 을 8비트 소수부 위에 올려둔 값
    //  mag_r = (wide + 128) >> 8    <- round-half-up
    wire signed [SHW-1:0] shp  = sh + 12'sd8;
    wire        [4:0]     shl  = bypass ? 5'd0 : shp[4:0];
    wire        [WW-1:0]  wide = ({{(WW-8){1'b0}}, M} << shl);
    wire        [WW-1:0]  magr = (wide + {{(WW-8){1'b0}}, 8'd128}) >> 8;

    wire         sat = hi | is_inf;
    wire [IW-2:0] mag = sat             ? MAXMAG        :
                        (lo | is_zero)  ? {(IW-1){1'b0}}:
                                          magr[IW-2:0];

    assign x   = s ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
    assign ovf = sat;

    // ---- 포맷 정합성 체크 (합성 무관) --------------------------------------
    initial begin
        if (IW < 12 || IW > 31)
            $display("ERROR: Bf16_To_Fix IW(%0d) out of supported range [12,31]", IW);
    end
endmodule
