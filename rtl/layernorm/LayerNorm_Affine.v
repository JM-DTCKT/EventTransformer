// -----------------------------------------------------------------------------
// LayerNorm_Affine : LayerNorm 의 affine 단계 — 곱셈기 1개 + 고정 시프트
//
//   y = sat( (xhat * gamma + (beta << BSHIFT) + (1 << (OSHIFT-1))) >> OSHIFT )
//
// 세 피연산자가 전부 고정소수점 코드라 스케일이 정적으로 정해집니다:
//
//   xhat  Q4.11   →  2^-11        (|xhat|  < 9.02,  실측)
//   gamma Q1.14   →  2^-14        (|gamma| < 1.26,  실측)
//   beta  Q4.11   →  2^-11        (|beta|  < 0.16,  실측)
//   곱          →  2^-25         = XHAT_FRAC + GAMMA_FRAC
//   출력  Q4.11   →  2^-11        (|out|   < 9.45,  실측 → ±16 에 1.7배 여유)
//
//   BSHIFT = 25 - 11 = 14   (beta 를 곱의 스케일로 좌시프트)
//   OSHIFT = 25 - 11 = 14   (곱을 출력 스케일로 우시프트)
//
// 둘 다 **전 사이트 공통 상수**이므로 `M` 곱수가 필요 없습니다 — requant 곱셈기
// 대신 고정 배선 시프트 하나로 끝납니다.
//
// gamma 를 Q4.11 로 내리지 않는 이유: gamma 는 xhat 에 *곱해지므로* 오차가
// 증폭됩니다. Q4.11(lsb 4.88e-4)이면 출력 오차가 9.02*2.44e-4 ≈ 4.5 LSB,
// Q1.14(lsb 6.1e-5)면 0.55 LSB. 저장 비용은 둘 다 int16 으로 같고, 하드웨어는
// OSHIFT 가 11 대신 14 가 될 뿐입니다.
// beta 는 *더해지기만* 하므로 출력 LSB 보다 정밀할 이유가 없어 Q4.11 입니다.
//
// 비트폭: xhat 15b × gamma 12b → 26b (DSP48E2 27×18 하나에 들어감).
// 반올림: round-half-up. 파이프라인 2단 (곱+정렬 / 반올림+시프트+포화).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module LayerNorm_Affine #(
    parameter XHAT_W     = 16, parameter XHAT_FRAC  = 11,   // Q4.11
    parameter GAMMA_W    = 16, parameter GAMMA_FRAC = 14,   // Q1.14
    parameter BETA_W     = 16, parameter BETA_FRAC  = 11,   // Q4.11
    parameter OUT_W      = 16, parameter OUT_FRAC   = 11    // Q4.11
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      in_valid,
    input  wire signed [XHAT_W-1:0]  xhat,
    input  wire signed [GAMMA_W-1:0] gamma,
    input  wire signed [BETA_W-1:0]  beta,
    output reg                       out_valid,
    output reg  signed [OUT_W-1:0]   y
);
    localparam integer PFRAC  = XHAT_FRAC + GAMMA_FRAC;   // 곱의 소수 비트 (25)
    localparam integer BSHIFT = PFRAC - BETA_FRAC;        // beta 정렬 좌시프트 (14)
    localparam integer OSHIFT = PFRAC - OUT_FRAC;         // 출력 우시프트 (14)
    localparam integer ACC_W  = XHAT_W + GAMMA_W + 2;     // 곱 + beta 여유
    localparam integer OMAX   =  (1 << (OUT_W-1)) - 1;
    localparam integer OMIN   = -(1 << (OUT_W-1));

    // ---------------- stage 1 : 곱 + beta 정렬 덧셈 ----------------
    reg signed [ACC_W-1:0] acc_q;
    reg                    v_q;

    always @(posedge clk) begin
        if (rst) begin
            v_q <= 1'b0; acc_q <= {ACC_W{1'b0}};
        end else begin
            v_q <= in_valid;
            if (in_valid)
                acc_q <= ($signed(xhat) * $signed(gamma))
                       + ($signed(beta) <<< BSHIFT);
        end
    end

    // ---------------- stage 2 : 반올림 + 시프트 + 포화 ----------------
    // 주의: concatenation 은 Verilog 에서 항상 unsigned 라, 인라인으로 쓰면
    // 덧셈 전체가 unsigned 가 되고 `>>>` 가 논리 시프트로 동작합니다.
    // 반드시 signed reg 에 담아 두 피연산자를 모두 signed 로 유지할 것.
    reg signed [ACC_W-1:0] round_add, sum, sr;

    always @* begin
        round_add = {{(ACC_W-1){1'b0}}, 1'b1} <<< (OSHIFT-1);
        sum       = acc_q + round_add;
        sr        = sum >>> OSHIFT;
    end

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; y <= {OUT_W{1'b0}};
        end else begin
            out_valid <= v_q;
            if (v_q) begin
                if (sr > OMAX)      y <= OMAX[OUT_W-1:0];
                else if (sr < OMIN) y <= OMIN[OUT_W-1:0];
                else                y <= sr[OUT_W-1:0];
            end
        end
    end
endmodule
