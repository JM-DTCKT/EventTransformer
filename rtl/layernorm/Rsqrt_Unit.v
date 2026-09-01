// -----------------------------------------------------------------------------
// Rsqrt_Unit : 1/sqrt(var) 유닛 (정규화 + 짝수지수 분해 + PWL-LUT)
//
// LayerNorm 의 나눗셈+제곱근 y = (x-mu)/sqrt(var+eps) 를 "역제곱근 한 번 +
// 곱셈 D 번" 으로 바꾸기 위한 1/sqrt 계산기.  행당 1회만 돌면 된다.
//
// var 는 [eps, 65536] 로 동적범위가 2^33 이나 되므로 그대로 근사할 수 없다.
// 2의 거듭제곱으로 정규화해 LUT 정의역을 [1,2) 로 좁힌다.
//
//    v = m * 2^q ,  m in [1,2)              <- q = v 의 leading-one 위치
//    var = v * 2^-VF = m * 2^k ,  k = q - VF
//
// ## 핵심
//
// sqrt 는 지수를 반으로 나누므로 **k 가 홀수면 시프트로 못 뺀다.**
// k = 2e + par (par = k mod 2) 로 쪼개서 홀수분 2^par 은 가수쪽으로 흡수시키고,
// 짝수분만 시프트로 처리한다.
//
//    1/sqrt(var) = (m * 2^par)^-0.5 * 2^-e
//                   └── PWL-LUT (par 로 뱅크 선택) ──┘ └ 호출부 시프트 ┘
//
// LUT 은 par=0 : 1/sqrt(m)  in (0.7071, 1.0]
//        par=1 : 1/sqrt(2m) in (0.5,    0.7071]
// 둘 다 동적범위가 2배 이내라 뱅크당 128 세그먼트 선형보간으로 상대오차
// 1.5e-5 면 충분하다 (Newton-Raphson 반복 불필요).
// 두 뱅크를 합쳐도 256 x 28b = 7 kbit 뿐이고, 이 유닛은 **행당 1개** 뿐이라
// distributed ROM 으로 깔면 된다.
//
// VF 가 짝수이면 (q - VF) 의 패리티 = q 의 패리티이므로 par = q[0] 이다.
// floor 나눗셈이 필요한 e = floor(k/2) 는 **산술 우시프트 한 번**이면 된다
// (k 가 음수여도 >>> 는 floor 로 내려간다).
//
// ## 고정소수점 포맷
//
//    v : unsigned (VW)      var * 2^VF,  v > 0 (호출부가 eps 를 더해서 넣는다)
//    r : unsigned UQ1.RF    (m*2^par)^-0.5, 범위 (2^(RF-1), 2^RF],  RW = RF+1
//    e : signed   (EW)      호출부 시프트량,  1/sqrt(var) = r * 2^-RF * 2^-e
//
// ## 파이프라인
//
// 3 stage, 행당 1회
//    S1 leading-one 검출 | S2 정규화 & seg/frac/par/e 분리 | S3 LUT+보간
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Rsqrt_Unit #(
    parameter integer VW = 61,   // v 폭
    parameter integer VF = 44,   // v 소수부 비트수 (**짝수여야 함**)
    parameter integer RW = 18,   // r 폭 (unsigned UQ1.17, = RF+1)
    parameter integer RF = 17,   // r 소수부 비트수
    parameter integer QW = 6,    // q 폭 (0..VW-1)
    parameter integer EW = 6     // e 폭 (signed)
)(
    input                        clk,
    input                        rst_n,
    input                        in_valid,
    input             [VW-1:0]   v,          // var * 2^VF,  v > 0
    output reg                   out_valid,
    output reg        [RW-1:0]   r,          // (m*2^par)^-0.5  (UQ1.RF)
    output reg signed [EW-1:0]   e           // 시프트량 (signed)
);
    localparam integer SEGB = 7;    // LUT 인덱스 비트 (뱅크당 128 세그먼트)
    localparam integer FRB  = 12;   // 세그먼트 내 위치 비트 (LUT 생성기와 일치)
    localparam integer KW   = QW + 2;   // k = q - VF 폭 (signed)
    localparam signed [KW-1:0] VFS = VF;   // signed 상수 (부호 섞임 방지)

    // ---- PWL-LUT : rsq_base_rom / rsq_delta_rom + 폭 상수 RSQ_BW / RSQ_DW ----
    //      ROM 폭은 생성기(gen_lut.py)가 실제 값 범위에 맞춰 최소화해 emit 한다.
    `include "Rsqrt_Lut.vh"

    // ======================= Stage 1 : leading-one 검출 =====================
    integer      i;
    reg [QW-1:0] q_c;
    always @(*) begin
        q_c = {QW{1'b0}};
        for (i = 0; i < VW; i = i + 1)
            if (v[i]) q_c = i[QW-1:0];       // 마지막으로 걸린 1 = 최상위 1
    end
    wire v_zero = (v == {VW{1'b0}});

    reg          v1, z1;
    reg [VW-1:0] vv1;
    reg [QW-1:0] q1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; z1 <= 1'b0; vv1 <= {VW{1'b0}}; q1 <= {QW{1'b0}};
        end else begin
            v1 <= in_valid; z1 <= v_zero; vv1 <= v; q1 <= q_c;
        end
    end

    // ======================= Stage 2 : 정규화 + seg/frac/par/e ===============
    //  mq = v << (VW-1-q)  =>  MSB 가 항상 bit(VW-1),  m = mq / 2^(VW-1) in [1,2)
    wire [VW-1:0]      mq = vv1 << ((VW-1) - q1);
    wire signed [KW-1:0] kk = $signed({{(KW-QW){1'b0}}, q1}) - VFS;

    reg               v2, z2, par2;
    reg [SEGB-1:0]    seg2;      // m 소수부의 상위 SEGB 비트 : LUT 인덱스
    reg [FRB-1:0]     frac2;     // 그 아래 FRB 비트 : 세그먼트 내 위치
    reg signed [EW-1:0] e2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0; z2 <= 1'b0; par2 <= 1'b0;
            seg2 <= {SEGB{1'b0}}; frac2 <= {FRB{1'b0}}; e2 <= {EW{1'b0}};
        end else begin
            v2    <= v1;
            z2    <= z1;
            par2  <= q1[0];                       // VF 짝수 -> k 패리티 = q 패리티
            seg2  <= mq[VW-2 -: SEGB];
            frac2 <= mq[VW-2-SEGB -: FRB];
            e2    <= (kk >>> 1);                  // floor(k/2), 산술 시프트
        end
    end

    // ======================= Stage 3 : LUT 조회 + 선형보간 ==================
    //  r = base[idx] + (delta[idx]*frac) / 2^FRB      (delta 는 항상 음수)
    wire [SEGB:0]              idx    = {par2, seg2};
    wire signed [RSQ_DW+FRB:0] mul_r  = rsq_delta_rom[idx] * $signed({1'b0, frac2});
    wire signed [RF+2:0]       interp = (mul_r + (1 << (FRB-1))) >>> FRB;   // 반올림
    wire signed [RF+2:0]       r_s    = $signed({1'b0, rsq_base_rom[idx]}) + interp;
                                                        // r in (2^(RF-1), 2^RF]

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0; r <= {RW{1'b0}}; e <= {EW{1'b0}};
        end else begin
            out_valid <= v2;
            r         <= z2 ? {RW{1'b0}} : r_s[RW-1:0];   // v==0 -> r=0 (안전값)
            e         <= e2;
        end
    end

    // ---- 포맷 정합성 체크 (합성 무관) --------------------------------------
    initial begin
        if (RW != RF+1)
            $display("ERROR: Rsqrt_Unit RW(%0d) must be RF+1(%0d)", RW, RF+1);
        if (RSQ_BW != RF+1)
            $display("ERROR: Rsqrt_Lut.vh RSQ_BW(%0d) != RF+1(%0d) - LUT 재생성 필요",
                     RSQ_BW, RF+1);
        if (VF % 2 != 0)
            $display("ERROR: Rsqrt_Unit VF(%0d) must be even (par = q[0] 가정)", VF);
        if ((1 << QW) < VW)
            $display("ERROR: Rsqrt_Unit QW(%0d) too small for VW(%0d)", QW, VW);
    end
endmodule
