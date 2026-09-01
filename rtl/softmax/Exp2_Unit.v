// -----------------------------------------------------------------------------
// Exp2_Unit : exp(z) unit,  base-2 변환 + 소수부 PWL-LUT 방식
//
// softmax 에서 exp 의 입력은 항상 z = x - max <= 0 이므로 exp(z) in (0, 1].
//
// ## 핵심 아이디어
//
// 자연지수를 그대로 근사하지 않고 base-2 로 바꾼 뒤 **정수부는 시프트,
// 소수부만 LUT** 으로 처리한다.
//
//    exp(z) = 2^(z * log2(e))            <- base-2 변환 (상수 1개 곱셈)
//           = 2^(-u),        u = -z*log2(e) >= 0
//           = 2^(-n) * 2^(-f),  n = floor(u) 정수부,  f = u - n in [0,1)
//             └ 배럴 시프트 ┘  └ PWL-LUT ┘
//
// LUT 는 f in [0,1) 의 g(f) = 2^-f in (0.5, 1] 하나만 담으면 되고, 동적범위가
// 2배 이내라 128 세그먼트 선형보간으로 최대오차 6.4e-6 (1.7 LSB) 밖에 안 된다.
//
// ## 고정소수점 포맷
//
//    z : signed   Q7.9  (17b) = x - max,  범위 (-128, 0]
//    u : unsigned Q5.9  (14b) = -z 를 16.0 으로 clamp
//                               (z < -16 이면 exp(z) < 2^-23 -> e = 0 이라 무손실)
//    t : unsigned Q5.20 (25b) = u * log2(e)   -> n = t[24:20], f = t[19:0]
//    g : unsigned UQ1.GF      = 2^-f, PWL-LUT + 선형보간, 범위 [2^(GF-1), 2^GF]
//    e : unsigned UQ1.EF      = g >> (n+1) 반올림,  GF = EF+1,  1.0 = 2^EF
//
// ## 파이프라인
//
// 4 stage, throughput 1 결과/clock
//    S1 clamp/negate | S2 base-2 변환 & n/f 분리 | S3 LUT+보간 | S4 2^-n 시프트
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Exp2_Unit #(
    parameter integer ZW = 17,   // z 폭 (signed Q7.9)
    parameter integer ZF = 9,    // z 소수부 비트수
    parameter integer EW = 17,   // e 폭 (unsigned UQ1.16, = EF+1)
    parameter integer EF = 16    // e 소수부 비트수
)(
    input                      clk,
    input                      rst_n,
    input                      in_valid,
    input      signed [ZW-1:0] z,          // Q7.9, z <= 0
    output reg                 out_valid,
    output reg        [EW-1:0] e           // UQ1.EF, ~= exp(z) in (0,1]
);
    // ---- 상수 / 유도 파라미터 ---------------------------------------------
    localparam [20:0]   LOG2E_Q20 = 21'd1512775;  // log2(e) = 1.4427 을 UQ1.20 으로
                                                  //  (= log2(e)*2^20, 상대오차 2.6e-7)
    localparam integer  TF     = 20;              // t 소수부 비트수
    localparam integer  UW     = 5 + ZF;          // = 14, u 폭 (Q5.9, 0..16.0)
    localparam integer  UMAX_I = (16 << ZF);      // = 8192, u clamp 값 (= 16.0)
    localparam [UW-1:0] UMAX   = UMAX_I;
    localparam integer  TW     = 25;              // t 폭 : 16*log2e = 23.08 -> 5+20
    localparam integer  NW     = 5;               // n 폭 (0..23)
    localparam integer  SEGB   = 7;               // LUT 인덱스 비트 (128 세그먼트)
    localparam integer  FRB    = TF - SEGB;       // = 13, 세그먼트 내 위치 비트
    localparam integer  GF     = EF + 1;          // g = 2^-f 소수부 (e 보다 1비트 많음)

    // ---- PWL-LUT : exp_base_rom / exp_delta_rom + 폭 상수 EXP_BW / EXP_DW ----
    //      ROM 폭은 생성기가 실제 값 범위에 맞춰 최소화해 emit 한다.
    `include "Exp2_Lut.vh"

    // ======================= Stage 1 : clamp / negate =======================
    //  u = min(-z, 16.0).  z < -16 은 clamp 해도 결과가 같으므로(e=0) t 폭을
    //  25b 로 고정할 수 있다.  z > 0 (스펙 위반) 은 u=0 으로 처리해 e=1.0.
    wire signed [ZW:0] z_neg = -{z[ZW-1], z};              // 18b, -z (오버플로 없음)
    wire               z_pos = ~z[ZW-1];                   // z >= 0
    wire               z_ovr = (z_neg > UMAX_I);           // -z > 16.0
    wire [UW-1:0]      u_c   = z_pos ? {UW{1'b0}} :
                               z_ovr ? UMAX : z_neg[UW-1:0];

    reg          v1;
    reg [UW-1:0] u1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v1 <= 1'b0;     u1 <= {UW{1'b0}}; end
        else        begin v1 <= in_valid; u1 <= u_c;        end
    end

    // ======================= Stage 2 : base-2 변환 + n/f 분리 ===============
    //  t = u * log2(e).  u 는 UQ5.9, 상수는 UQ1.20 이므로 곱은 UQ6.29 -> >>9 로 UQ5.20.
    wire [UW+20:0] mul_t = u1 * LOG2E_Q20;                       // Q5.29 (35b)
    wire [TW-1:0]  t     = (mul_t + (1 << (ZF-1))) >> ZF;        // Q5.20 (반올림)

    reg            v2;
    reg [NW-1:0]   n2;      // 정수부 n : 2^-n 시프트량
    reg [SEGB-1:0] seg2;    // 소수부 상위 7b : LUT 인덱스
    reg [FRB-1:0]  frac2;   // 소수부 하위 13b : 세그먼트 내 위치
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0; n2 <= {NW{1'b0}}; seg2 <= {SEGB{1'b0}}; frac2 <= {FRB{1'b0}};
        end else begin
            v2    <= v1;
            n2    <= t[TW-1:TF];        // floor(u)
            seg2  <= t[TF-1 -: SEGB];   // f 의 상위 7 비트
            frac2 <= t[FRB-1:0];        // f 의 하위 13 비트
        end
    end

    // ======================= Stage 3 : LUT 조회 + 선형보간 ==================
    //  g = base[seg] + (delta[seg]*frac) / 2^FRB         (delta 는 항상 음수)
    wire signed [EXP_DW+FRB:0] mul_g  = exp_delta_rom[seg2] * $signed({1'b0, frac2});
    wire signed [GF+2:0]       interp = (mul_g + (1 << (FRB-1))) >>> FRB;   // 반올림
    wire signed [GF+2:0]       g_s    = $signed({1'b0, exp_base_rom[seg2]}) + interp;
                                                        // g in [2^(GF-1), 2^GF]

    reg          v3;
    reg [NW-1:0] n3;
    reg [GF:0]   g3;        // 19b : g in (0.5, 1] -> [2^17, 2^18]
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v3 <= 1'b0; n3 <= {NW{1'b0}}; g3 <= {(GF+1){1'b0}}; end
        else        begin v3 <= v2;   n3 <= n2;         g3 <= g_s[GF:0];      end
    end

    // ======================= Stage 4 : 2^-n 배럴 시프트 =====================
    //  e(UQ1.EF) = g(UQ1.GF) * 2^-n  =>  (g + 2^n) >> (n+1)
    //  더해주는 2^n 은 0.5 LSB 반올림.  n >= 19 면 자연히 0 이 된다.
    wire [31:0] g_ext = {{(31-GF){1'b0}}, g3};
    wire [31:0] shf   = (g_ext + (32'd1 << n3)) >> (n3 + 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin out_valid <= 1'b0; e <= {EW{1'b0}}; end
        else        begin out_valid <= v3;   e <= shf[EW-1:0]; end
    end

    // 포맷 정합성 체크 : e 는 1.0(=2^EF) 을 담아야 하므로 EW = EF+1
    initial begin
        if (EW != EF+1)
            $display("ERROR: Exp2_Unit EW(%0d) must be EF+1(%0d)", EW, EF+1);
        if (EXP_BW != GF+1)
            $display("ERROR: Exp2_Lut.vh EXP_BW(%0d) != GF+1(%0d) — LUT 재생성 필요",
                     EXP_BW, GF+1);
    end
endmodule
