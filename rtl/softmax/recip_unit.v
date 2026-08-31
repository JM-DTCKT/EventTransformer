// ============================================================================
//  recip_unit.v  --  분모 역수 유닛 (정규화 + PWL-LUT)
// ----------------------------------------------------------------------------
//  softmax 의 나눗셈 y_i = e_i / S 를 "역수 한 번 + 곱셈 N 번" 으로 바꾸기 위한
//  1/S 계산기.  S 는 96개 exp 결과의 합이라 [1, 96] 범위로 넓다.
//
//  [핵심 아이디어]  S 를 2의 거듭제곱으로 정규화해서 LUT 정의역을 [1,2) 로 좁힌다.
//
//     S = m * 2^p ,  m in [1,2)          <- p = S 의 MSB 위치 (leading-one)
//     1/S = (1/m) * 2^-p
//            └ PWL-LUT ┘ └ 시프트 ┘
//
//  1/m in (0.5, 1] 이므로 128 세그먼트 선형보간으로 최대오차 1.6e-5
//  (상대오차 3.2e-5) 이면 충분하다.  Newton-Raphson 반복이 필요 없다.
//  — 이 오차는 출력 UQ1.15 로 환산하면 0.5 LSB 로, 최종 반올림 오차에 묻힌다.
//    (256 세그먼트로 늘려도 softmax 출력 오차가 전혀 개선되지 않음을 실측 확인)
//
//  [고정소수점 포맷]
//     s   : unsigned UQ8.EF        분모 (exp 결과 UQ1.EF 를 N개 누산)
//     m_q : unsigned        (SWb)  s 를 MSB 가 bit(SW-1) 에 오도록 좌시프트한 값
//                                  = m * 2^(SW-1),  m = 1.f  (f = m_q[SW-2:0])
//     r   : unsigned UQ1.RF        1/m, 범위 (2^(RF-1), 2^RF],  RW = RF+1
//     p   : MSB 위치 (0..23)       호출부가 최종 시프트량 계산에 사용
//
//  [호출부 사용법]  s_real = s / 2^SF (SF=17) 일 때
//     1/s_real = r * 2^-RF * 2^(SF-p)
//     y = e * (1/s_real) 을 UQ1.OF 로 얻으려면  (e * r) >> (p + RF - OF)
//
//  [파이프라인]  3 stage, 벡터당 1회만 사용
//     S1 leading-one 검출 | S2 정규화 시프트 & seg/frac 분리 | S3 LUT+보간
// ============================================================================
`timescale 1ns/1ps

module recip_unit #(
    parameter integer SW = 24,   // 분모 s 폭
    parameter integer RW = 18,   // r 폭 (unsigned UQ1.17, = RF+1)
    parameter integer RF = 17,   // r 소수부 비트수
    parameter integer PW = 5     // p 폭 (0..SW-1)
)(
    input                   clk,
    input                   rst_n,
    input                   in_valid,
    input      [SW-1:0]     s,          // 분모 (UQ7.17), s > 0
    output reg              out_valid,
    output reg [RW-1:0]     r,          // 1/m  (UQ1.RF)
    output reg [PW-1:0]     p           // s 의 MSB 위치
);
    localparam integer SEGB = 7;             // LUT 인덱스 비트 (128 세그먼트)
    localparam integer FRB  = (SW-1) - SEGB; // 세그먼트 내 위치 비트 (SW=25 -> 17)

    // ---- PWL-LUT : rcp_base_rom / rcp_delta_rom + 폭 상수 RCP_BW / RCP_DW ----
    //      ROM 폭은 생성기가 실제 값 범위에 맞춰 최소화해 emit 한다.
    `include "recip_lut.vh"

    // ======================= Stage 1 : leading-one 검출 =====================
    //  s 의 최상위 '1' 위치 p 를 찾는다 (우선순위 인코더).  s==0 이면 zero 플래그.
    integer         k;
    reg [PW-1:0]    p_c;
    always @(*) begin
        p_c = {PW{1'b0}};
        for (k = 0; k < SW; k = k + 1)
            if (s[k]) p_c = k[PW-1:0];       // 마지막으로 걸린 1 = 최상위 1
    end
    wire s_zero = (s == {SW{1'b0}});

    reg          v1, z1;
    reg [SW-1:0] s1;
    reg [PW-1:0] p1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; z1 <= 1'b0; s1 <= {SW{1'b0}}; p1 <= {PW{1'b0}};
        end else begin
            v1 <= in_valid; z1 <= s_zero; s1 <= s; p1 <= p_c;
        end
    end

    // ======================= Stage 2 : 정규화 + seg/frac 분리 ===============
    //  m_q = s << (SW-1-p)  =>  MSB 가 항상 bit(SW-1),  m = m_q / 2^23 in [1,2)
    wire [SW-1:0] m_q = s1 << ((SW-1) - p1);

    reg            v2, z2;
    reg [PW-1:0]   p2;
    reg [SEGB-1:0] seg2;    // m 소수부(SW-1 비트) 상위 SEGB b : LUT 인덱스
    reg [FRB-1:0]  frac2;   // 나머지 하위 FRB b : 세그먼트 내 위치
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0; z2 <= 1'b0; p2 <= {PW{1'b0}};
            seg2 <= {SEGB{1'b0}}; frac2 <= {FRB{1'b0}};
        end else begin
            v2    <= v1;
            z2    <= z1;
            p2    <= p1;
            seg2  <= m_q[SW-2 -: SEGB];   // f = m_q[SW-2:0] 의 상위 SEGB 비트
            frac2 <= m_q[FRB-1:0];        // f 의 하위 FRB 비트
        end
    end

    // ======================= Stage 3 : LUT 조회 + 선형보간 ==================
    //  r = base[seg] + (delta[seg]*frac) / 2^FRB         (delta 는 항상 음수)
    wire signed [RCP_DW+FRB:0] mul_r  = rcp_delta_rom[seg2] * $signed({1'b0, frac2});
    wire signed [RF+2:0]       interp = (mul_r + (1 << (FRB-1))) >>> FRB;   // 반올림
    wire signed [RF+2:0]       r_s    = $signed({1'b0, rcp_base_rom[seg2]}) + interp;
                                                        // r in (2^(RF-1), 2^RF]

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0; r <= {RW{1'b0}}; p <= {PW{1'b0}};
        end else begin
            out_valid <= v2;
            r         <= z2 ? {RW{1'b0}} : r_s[RW-1:0];   // s==0 -> r=0 (안전값)
            p         <= p2;
        end
    end

    // 포맷 정합성 체크
    initial begin
        if (RW != RF+1)   // r 은 UQ1.RF, 최대값 2^RF 를 담아야 함
            $display("ERROR: recip_unit RW(%0d) must be RF+1(%0d)", RW, RF+1);
        if (RCP_BW != RF+1)
            $display("ERROR: recip_lut.vh RCP_BW(%0d) != RF+1(%0d) — LUT 재생성 필요",
                     RCP_BW, RF+1);
    end
endmodule
