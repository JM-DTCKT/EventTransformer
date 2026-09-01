// -----------------------------------------------------------------------------
// Requant_Int : 정수 재양자화기 — 데이터패스의 유일한 requantizer 형태
//
//   out = sat( (acc * M + (1 << (sh-1))) >> sh )
//
//   acc : GEMM 누산기 (INT32) 또는 앞단 Qm.n 코드
//   M   : 오프라인 상수. M[c] = round(s_x * s_w[c] / lsb_out * 2^sh)
//         → 역양자화(× s_x·s_w)와 재양자화(÷ lsb_out)가 곱수 하나로 접힘.
//   sh  : 레이어당 상수 (uint6, 실측 31~46)
//
// 이 모듈 하나가 다음을 전부 처리합니다 (파라미터만 다름):
//   INT32 → Q4.11  (GELU 입력)          OUT_W=16, UNSIGNED_OUT=0
//   INT32 → Q6.9   (softmax 입력)       OUT_W=16, UNSIGNED_OUT=0
//   INT32 → INT8   (ReLU / MAC 피연산자) OUT_W=8,  UNSIGNED_OUT=0
//   Q1.14 → uint8  (softmax 출력)       OUT_W=8,  UNSIGNED_OUT=1  → [0,127]
//
// 반올림: round-half-up (`+2^(sh-1)` 후 산술 우시프트). 음수에서 동점은 +∞ 방향.
// 포화  : 2의 보수 전체 범위 [-2^(OUT_W-1), 2^(OUT_W-1)-1]
//         (UNSIGNED_OUT=1 이면 [0, 2^(OUT_W-1)-1])
//
// ## bias (USE_BIAS=1)
//
// `acc[c] += b_int[c]` 를 곱 **앞에서** 합니다. b_int[c] 와 M[c] 는 같은 채널
// 인덱스라 ROM 주소생성기도 공유합니다.
//
// ## 파이프라인
//
//   PIPE_PRE=0  2단 : 곱 / 반올림+시프트+포화
//   PIPE_PRE=1  3단 : 프리애더를 따로 끊음   ← **이 프로젝트는 전부 이쪽**
//
// [타이밍] `(acc + b_int) * M` 은 DSP48E2 의 27비트 pre-adder 에 매핑되기를
// 기대할 만하지만, acc 를 32비트로 두면 포트에 안 들어가 **패브릭 가산기**가
// 됩니다. 그러면 32비트 CARRY 체인이 곱셈기 앞에 그대로 붙습니다 — 그래서
// 한 단을 끊습니다. `mult`/`shift` 도 같이 등록해 레인 안으로 끌어들이므로
// DSP 64개를 먹던 상수 팬아웃이 레인별 FF 로 나뉩니다.
//
// 단수를 바꾸면 쓰는 쪽의 지연 상수도 같이 봐야 합니다
// (`Format_Cast_Act` 머리말 → `EvT_Engine.FCA_LAT_*`).
//
// 주의: acc(23b 유효) x M(31b) = 54비트라 DSP48E2 의 48비트 P 를 넘습니다.
//       합성 시 2 DSP 캐스케이드(PCIN>>17)로 매핑되며, 그것을 의도한 것입니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Requant_Int #(
    parameter ACC_W        = 32,   // 누산기 폭 (signed)
    parameter MUL_W        = 32,   // 곱수 M 폭 (signed, |M| < 2^31)
    parameter SH_W         = 6,    // 시프트 폭 (uint6)
    parameter OUT_W        = 16,   // 소비자 폭
    parameter UNSIGNED_OUT = 0,    // 1 = [0, 2^(OUT_W-1)-1] 로 클램프
    parameter USE_BIAS     = 0,    // 1 = folded INT32 bias 를 누산기에 더함
    parameter BIAS_W       = 32,   // b_int[c] 폭 (signed, 실측 최대 |b_int| = 6,204)
    parameter PIPE_PRE     = 0     // 1 = 프리애더를 한 단 끊어 3단 (고Fmax 용)
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     in_valid,
    input  wire signed [ACC_W-1:0]  acc,
    input  wire signed [BIAS_W-1:0] bias,    // b_int[c]; USE_BIAS=0 이면 무시
    input  wire signed [MUL_W-1:0]  mult,    // M[c]
    input  wire        [SH_W-1:0]   shift,   // sh
    output reg                      out_valid,
    output reg  signed [OUT_W-1:0]  out
);
    localparam integer PRE_W  = ((ACC_W > BIAS_W) ? ACC_W : BIAS_W) + 1;  // 프리애더 폭
    localparam integer PROD_W = PRE_W + MUL_W;
    localparam integer OMAX   =  (1 << (OUT_W-1)) - 1;
    localparam integer OMIN   = -(1 << (OUT_W-1));

    // ---------------- stage 0 : 프리애더 (PIPE_PRE=1 일 때만) ----------------
    // `USE_BIAS=0` 이면 이 덧셈은 통째로 빠져 합성됩니다.
    // 왜 여기서 레지스터를 끊는지는 머리말 [타이밍] 참조.
    wire signed [PRE_W-1:0] pre = (USE_BIAS != 0)
                                ? ($signed(acc) + $signed(bias))
                                : $signed(acc);

    wire signed [PRE_W-1:0]  pre_q;
    wire signed [MUL_W-1:0]  mul_q;
    wire        [SH_W-1:0]   sh_q0;
    wire                     v_q0;

    generate
    if (PIPE_PRE != 0) begin : g_pre
        reg  signed [PRE_W-1:0]  pre_r;
        reg  signed [MUL_W-1:0]  mul_r;
        reg         [SH_W-1:0]   sh_r;
        reg                      v_r;
        always @(posedge clk) begin
            if (rst) begin
                v_r <= 1'b0; pre_r <= {PRE_W{1'b0}};
                mul_r <= {MUL_W{1'b0}}; sh_r <= {SH_W{1'b0}};
            end else begin
                v_r <= in_valid;
                if (in_valid) begin
                    pre_r <= pre;  mul_r <= mult;  sh_r <= shift;
                end
            end
        end
        assign pre_q = pre_r; assign mul_q = mul_r;
        assign sh_q0 = sh_r;  assign v_q0  = v_r;
    end else begin : g_pre
        assign pre_q = pre;  assign mul_q = mult;
        assign sh_q0 = shift; assign v_q0 = in_valid;
    end
    endgenerate

    // ---------------- stage 1 : 곱 ----------------
    reg  signed [PROD_W-1:0] prod_q;
    reg         [SH_W-1:0]   sh_q;
    reg                      v_q;

    always @(posedge clk) begin
        if (rst) begin
            v_q <= 1'b0; prod_q <= {PROD_W{1'b0}}; sh_q <= {SH_W{1'b0}};
        end else begin
            v_q <= v_q0;
            if (v_q0) begin
                prod_q <= $signed(pre_q) * $signed(mul_q);
                sh_q   <= sh_q0;
            end
        end
    end

    // ---------------- stage 2 : 반올림 + 시프트 + 포화 ----------------
    reg signed [PROD_W-1:0] round_add, sum, sr;

    always @* begin
        // sh = 0 이면 반올림 항이 없음 (1 << -1 방지)
        round_add = (sh_q == {SH_W{1'b0}})
                  ? {PROD_W{1'b0}}
                  : ({{(PROD_W-1){1'b0}}, 1'b1} <<< (sh_q - 1'b1));
        sum = prod_q + round_add;
        sr  = sum >>> sh_q;                       // 산술 우시프트
    end

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; out <= {OUT_W{1'b0}};
        end else begin
            out_valid <= v_q;
            if (v_q) begin
                if (UNSIGNED_OUT != 0) begin
                    if (sr < 0)          out <= {OUT_W{1'b0}};
                    else if (sr > OMAX)  out <= OMAX[OUT_W-1:0];
                    else                 out <= sr[OUT_W-1:0];
                end else begin
                    if (sr > OMAX)       out <= OMAX[OUT_W-1:0];
                    else if (sr < OMIN)  out <= OMIN[OUT_W-1:0];
                    else                 out <= sr[OUT_W-1:0];
                end
            end
        end
    end
endmodule
