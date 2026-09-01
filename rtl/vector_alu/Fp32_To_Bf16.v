// -----------------------------------------------------------------------------
// Fp32_To_Bf16 : fp32 → bf16, round-to-nearest-even (조합)
//
// LayerNorm 에서 **fp32 누산을 bf16 으로 내리는 유일한 지점**입니다:
//
//     mu  = bf16( fp32_sum(x)   / E )
//     var = bf16( fp32_sum(ctr²) / E )
//     ctr = bf16( x - mu )                     ← fp32 뺄셈 결과를 내림
//
// bf16 은 fp32 의 상위 16비트라 **가수를 자르고 한 번 반올림**하면 끝입니다.
// 잘리는 16비트 중 최상위가 guard, 나머지가 sticky 입니다.
//
//     up = g & (sticky | mant[0])              ← 동점이면 짝수로
//
// 반올림이 가수를 넘치면 지수를 하나 올립니다 (가수는 0 이 됩니다).
//
// ## E 로 나누기
//
// `log2e` 만큼 지수를 빼서 나눗셈을 대신합니다. E 가 2의 거듭제곱이므로 **오차가
// 없습니다** (LayerNorm 의 특징 수는 128 / 256 등). 0 을 주면 나누지 않습니다.
// 지수가 0 이하로 내려가면 flush-to-zero 입니다.
//
// denormal 입력은 0 으로 봅니다 (다른 bf16 블록과 같은 정책).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Fp32_To_Bf16 (
    input  wire [31:0] f,
    input  wire [5:0]  log2e,     // 이 값만큼 2^-log2e 를 곱함 (E 로 나누기)
    output reg  [15:0] y
);
    wire        s = f[31];
    wire [7:0]  e = f[30:23];
    wire [22:0] m = f[22:0];

    wire        g   = m[15];              // guard
    wire        stk = |m[14:0];           // sticky
    wire [6:0]  mt  = m[22:16];           // 남길 가수 7비트
    wire        up  = g & (stk | mt[0]);

    wire [7:0]  mt_r  = {1'b0, mt} + {7'd0, up};
    wire        carry = mt_r[7];

    reg signed [9:0] e_adj;
    always @* begin
        if (e == 8'd0) begin                                  // 0 / denormal
            y = 16'd0;
        end else if (e == 8'hFF) begin                        // Inf / NaN
            y = {s, 8'hFF, mt};
        end else begin
            e_adj = $signed({2'b00, e}) + {9'd0, carry} - $signed({4'd0, log2e});
            if (e_adj <= 10'sd0)        y = 16'd0;             // 언더플로 → FTZ
            else if (e_adj >= 10'sd255) y = {s, 8'hFF, 7'd0};  // 오버플로 → Inf
            else                        y = {s, e_adj[7:0], mt_r[6:0]};
        end
    end
endmodule
