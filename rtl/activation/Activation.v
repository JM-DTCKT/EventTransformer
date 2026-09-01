// -----------------------------------------------------------------------------
// Activation : requantizer 뒤에 붙는 활성함수 — N 레인 병렬, 순수 조합
//
// 레이어마다 런타임으로 골라야 하므로(마지막 레이어는 활성함수가 없음)
// 파라미터가 아니라 **입력 포트**로 선택합니다.
//
//   act_sel                       act_parm         회로
//   ------------------------------------------------------------------------
//   0  ACT_NONE    y = x          -                (배선만)
//   1  ACT_RELU    y = max(0,x)   -                레인당 AND 게이트 DATA_W 개
//   2  ACT_LEAKY   y = x<0 ? x>>>k : x   k=parm[2:0]   가변 산술시프트 (3비트)
//   3  ACT_CLAMP   y = min(b, max(0,x))  b=parm[6:0]   비교기 2개
//
// ## 왜 부호비트 마스킹으로 끝나는가
//
// 이 데이터패스는 **대칭 양자화**(zero-point = 0)라 실수 0 이 코드 0 에 정확히
// 대응합니다. 따라서 ReLU 에 비교기도, 뺄셈도, zero-point 보정도 필요 없습니다.
//
// ## 포화와의 순서 — ReLU 는 정확, 나머지는 근사
//
// `Requant_Int` 이 **부호 있는 포화**로 내보낸 뒤 여기서 활성함수를 씌웁니다.
// ReLU 는 이 순서가 정확합니다:
//
//     relu( sat_[-128,127](v) )  ==  sat_[0,127](v)        모든 v 에 대해
//
// LEAKY / CLAMP 는 **포화가 일어나는 입력에서만** 앞뒤 순서가 갈립니다
// (예: v=-1000, k=1 → 여기서는 sat 후 -128>>>1 = -64, 이론값은 -500 포화 후 -128).
// 캘리브레이션이 제대로면 포화 자체가 드물어 무시할 수 있지만, **정확한 등가는
// 아닙니다.** 새 활성함수를 실제로 쓰기 전에 골든 모델과 대조하십시오.
//
// ## 여기 없는 것
//
// GELU · softmax 는 int8 코드가 아니라 **Qm.n 고정소수(Q4.11 / Q6.9)** 에서 LUT 로
// 계산합니다 — 포맷도 자원 규모도 달라 별도 모듈입니다 (`Gelu_Pwl`, `Exp2_Unit`).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Activation #(
    parameter DATA_W = 8,        // 데이터 비트폭 (배포 경로는 INT8)
    parameter N      = 32,       // 병렬 레인 수
    parameter SEL_W  = 2,
    parameter PARM_W = 8
)(
    input  wire [SEL_W-1:0]    act_sel,
    input  wire [PARM_W-1:0]   act_parm,     // 의미는 act_sel 에 따라 다름
    input  wire [N*DATA_W-1:0] din,
    output wire [N*DATA_W-1:0] dout
);
    localparam [1:0] ACT_NONE  = 2'd0,
                     ACT_RELU  = 2'd1,
                     ACT_LEAKY = 2'd2,
                     ACT_CLAMP = 2'd3;

    wire [2:0]  k     = act_parm[2:0];                       // LEAKY : 기울기 2^-k
    wire signed [DATA_W-1:0] bnd = {1'b0, act_parm[DATA_W-2:0]};  // CLAMP : 상한

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : LANE
            wire signed [DATA_W-1:0] x   = din[i*DATA_W +: DATA_W];
            wire                     neg = x[DATA_W-1];
            reg  signed [DATA_W-1:0] y;

            always @* begin
                case (act_sel)
                    ACT_RELU:  y = neg ? {DATA_W{1'b0}} : x;
                    ACT_LEAKY: y = neg ? (x >>> k) : x;
                    ACT_CLAMP: y = neg ? {DATA_W{1'b0}} : ((x > bnd) ? bnd : x);
                    default:   y = x;                        // ACT_NONE
                endcase
            end

            assign dout[i*DATA_W +: DATA_W] = y;
        end
    endgenerate
endmodule
