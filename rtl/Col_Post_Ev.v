// -----------------------------------------------------------------------------
// Col_Post_Ev : GEMM 컬럼(32레인 INT32) → 소비자 포맷  (EvT 판)
// GEMM 출력을 consumer에 따라 format casting

// `fpga_nl/Col_Post` 에 **활성함수 분기**를 더한 것입니다. EvT 는 int8 소비자
// 앞에 ReLU 가 붙는 자리가 있습니다:
//
//   proc_events.seq_init.1 / .4      Linear → ReLU → int8
//   proc_embs_block.linear1          Linear → ReLU → int8
//   models_clf.0.linear_1            Linear → ReLU → int8
//
// GELU 자리(Q4.11 경로)와 bf16/Q6.9 경로는 `fpga_nl` 과 동일합니다.
//
// ## 소비자
//
//   consumer            경로                                        출력
//   -------------------------------------------------------------------------
//   INT8 (0)  requant(acc+b, M, sh) → [ReLU] → int8                A_Mem 하위 8b
//   Q411 (1)  requant → Q4.11 → GELU → requant → int8              A_Mem 하위 8b
//             raw16=1 이면 마지막 재양자화를 건너뛰고 Q4.11 16b 그대로         A_Mem 16b
//   BF16 (2)  bf16( (acc+b) · step[n] )                            A_Mem 16b
//   Q69  (3)  requant → Q6.9                                       Softmax_Attn 로 직결
//
// ## in_proj 은 여기서 특별대우가 없습니다
//
// `in_proj` 은 384행이 Q/K/V 세 구간이지만 **M 이 행마다 다르므로** 채널별 곱수
// 하나로 전부 표현됩니다(구간마다 출력 step 이 달라도 M[c] 에 접혀 있음). 다만
// Q 는 `layer_norm_1`, K·V 는 `layer_norm_x` 를 읽으므로 **A 피연산자가 달라**
// 하드웨어에서는 GEMM 두 번입니다 (rows 0-127 / rows 128-383). 그건 스케줄러가
// `a_base` 와 `b_base` 를 달리 주는 것으로 끝나고, 이 모듈은 관여하지 않습니다.
//
// (manifest 의 `bands[].input_step` 은 이름과 달리 **출력** step 입니다 —
//  `export_fpga.py:216` 이 각 밴드의 출력 스케일을 그 필드에 넣습니다.)
//
// ## V 는 전치가 필요합니다
//
// `attn·V` 의 reduce 축이 토큰이라 V 만 워드[j] 레인=d 가 필요합니다. 그 축 변환은
// `Transpose32` 가 하고, 이 모듈은 평소처럼 컬럼을 내보내기만 합니다 — 어디로
// 보낼지는 스케줄러가 정합니다.
//
// ## 지연
//
//   BF16          : 1 사이클
//   INT8/Q69      : 3 사이클  (ReLU 는 조합이라 늘지 않음)
//   INT8+req2     : 6 사이클
//   Q411          : 3(requant) + 3(GELU) + 3(requant) = 9 사이클
//
// `Requant_Int` 가 2단 → **3단** 이 되면서(100 MHz 를 맞추려 프리애더를 끊음)
// 전부 한 단씩 늘었습니다. 바꿀 때 `EvT_Engine` 의 `CP_LAT_*` 도 같이 봐야 합니다.
// -----------------------------------------------------------------------------
module Col_Post_Ev #(
    parameter N       = 32,
    parameter ACT_W   = 8,
    parameter PSUM_W  = 32,
    parameter GELU_LUT_FILE = "gelu.hex"
)(
    input  wire                     clk,
    input  wire                     rst,

    input  wire [1:0]               consumer,   // 0 int8  1 Q4.11+GELU  2 bf16  3 Q6.9
    input  wire signed [PSUM_W-1:0] bias,       // b_int[n]
    input  wire signed [PSUM_W-1:0] mult,       // M[n]  또는  step(bf16, 하위 16b)
    input  wire [5:0]               shift,
    input  wire signed [PSUM_W-1:0] g_mult,     // GELU 뒤 int8 재양자화 (스칼라)
    input  wire [5:0]               g_shift,
    input  wire [1:0]               act_sel,    // 0 없음  1 ReLU  (Activation.v 인코딩)
    input  wire [7:0]               act_parm,
    // Q4.11 소비자에서 **int8 재양자화를 건너뛰고 16비트를 그대로** 내보냅니다.
    // `linear1` 뒤가 LayerNorm 이라 int8 격자가 없는 자리에 씁니다. LayerNorm 은
    // 스케일 불변이라 골든(Q4.11 실수 입력)과 결과가 완전히 같습니다.
    input  wire                     raw16,
    // int8 소비자에서 **활성함수 뒤 한 번 더** 재양자화합니다 (g_mult/g_shift).
    // 골든이 ReLU 앞뒤로 격자를 달리 쓰는 자리(`proc_ev.1`, `clf.linear_1`) 용.
    input  wire                     req2,

    input  wire                     in_valid,
    input  wire [N*PSUM_W-1:0]      acc,

    output wire                     out_valid,
    output wire [N*16-1:0]          out_data,   // A_Mem 워드 (int8 은 부호확장)
    output wire [N*16-1:0]          out_q69     // Softmax 로 가는 Q6.9 (consumer=3)
);
    localparam C_INT8=2'd0, C_Q411=2'd1, C_BF16=2'd2, C_Q69=2'd3;

    wire is_q411 = (consumer == C_Q411);
    wire is_bf16 = (consumer == C_BF16);

    wire [N-1:0] v_i8, v_16, v_bf, v_g2;

    // requant 직후(활성함수 전) / 직후(활성함수 후) 버스
    wire [N*ACT_W-1:0] i8r_bus, i8a_bus;
    Activation #(.DATA_W(ACT_W), .N(N)) u_act (
        .act_sel(act_sel), .act_parm(act_parm), .din(i8r_bus), .dout(i8a_bus));

    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : LANE
            wire signed [PSUM_W-1:0] a = acc[g*PSUM_W +: PSUM_W];

            // ---- INT8 직행 (consumer 0) + 활성함수 ----
            wire signed [ACT_W-1:0] y_i8r;
            Requant_Int #(.ACC_W(PSUM_W), .MUL_W(PSUM_W), .SH_W(6), .OUT_W(ACT_W),
                          .PIPE_PRE(1), .UNSIGNED_OUT(0), .USE_BIAS(1), .BIAS_W(PSUM_W)) u_i8 (
                .clk(clk), .rst(rst), .in_valid(in_valid && (consumer == C_INT8)),
                .acc(a), .bias(bias), .mult(mult), .shift(shift),
                .out_valid(v_i8[g]), .out(y_i8r));

            // 활성함수는 **32레인 버스 단위**라 루프 밖에서 한 번 겁니다.
            assign i8r_bus[g*ACT_W +: ACT_W] = y_i8r;
            wire signed [ACT_W-1:0] y_i8 = i8a_bus[g*ACT_W +: ACT_W];

            // ---- 16비트 Q포맷 (consumer 1 = Q4.11, 3 = Q6.9) ----
            wire signed [15:0] y_16;
            Requant_Int #(.ACC_W(PSUM_W), .MUL_W(PSUM_W), .SH_W(6), .OUT_W(16),
                          .PIPE_PRE(1), .UNSIGNED_OUT(0), .USE_BIAS(1), .BIAS_W(PSUM_W)) u_16 (
                .clk(clk), .rst(rst),
                .in_valid(in_valid && (is_q411 || (consumer == C_Q69))),
                .acc(a), .bias(bias), .mult(mult), .shift(shift),
                .out_valid(v_16[g]), .out(y_16));

            // ---- GELU (Q4.11 → Q4.11) → int8 ----
            wire               vg;
            wire signed [15:0] y_g;
            // ---- GELU : 64세그먼트 PWL (`GELU/verilog/gelu_pwl.v`) ----
            // 전수 LUT(16,384 x 16b = 8 BRAM36, 32레인이면 256 BRAM)을
            // **base/delta 64쌍 = 2 Kb** 로 줄인 판입니다. Q4.11 전 코드에서
            // 정확한 GELU 대비 **최대 1 LSB** (65,536 중 96.6%는 정확).
            //
            // 통합 시 다른 점 둘:
            //   · 리셋이 **active-low(`rst_n`)** — 프로젝트 나머지는 active-high
            //   · 지연이 **3사이클** (기존 Gelu 는 2) → CP_LAT_G 6 → 7 → (requant 3단화로) 9
            gelu_pwl #(.W(16), .QF(11), .FR(14)) u_gelu (
                .clk(clk), .rst_n(~rst), .in_valid(v_16[g] && is_q411), .x(y_16),
                .out_valid(vg), .y(y_g));

            // 2차 재양자화기 하나를 **GELU 경로와 int8 경로가 나눠** 씁니다 —
            // 두 소비자가 동시에 도는 일이 없어 곱수/시프트도 같이 씁니다.
            wire signed [ACT_W-1:0] y_g8;
            wire signed [15:0] r2_in = is_q411 ? y_g : {{8{y_i8[ACT_W-1]}}, y_i8};
            Requant_Int #(.ACC_W(16), .MUL_W(PSUM_W), .SH_W(6), .OUT_W(ACT_W),
                          .PIPE_PRE(1), .UNSIGNED_OUT(0), .USE_BIAS(0), .BIAS_W(PSUM_W)) u_g8 (
                .clk(clk), .rst(rst),
                .in_valid(is_q411 ? vg : (v_i8[g] && req2)),
                .acc(r2_in), .bias({PSUM_W{1'b0}}), .mult(g_mult), .shift(g_shift),
                .out_valid(v_g2[g]), .out(y_g8));

            // raw16 은 재양자화(**3단**)를 안 거치므로 그만큼 늦춰 같은 지연으로
            // 맞춥니다. 소비자마다 지연이 다르면 쓰기 주소 파이프(`CP_LAT_*`)가
            // 하나 더 갈라져 `fpga_nl` 에서 겪은 "첫 워드가 안 써짐"이 재발합니다.
            reg signed [15:0] y_g_d1, y_g_d2, y_g_d3;
            always @(posedge clk) begin
                y_g_d1 <= y_g;
                y_g_d2 <= y_g_d1;
                y_g_d3 <= y_g_d2;
            end

            // ---- bf16 : bias 를 먼저 더하고 step 을 곱함 ----
            //   골든은 bf16(acc · s_x·s_w[c]) 이고 acc 에 b_int 가 이미 들어 있음
            wire signed [PSUM_W:0] a_b = $signed(a) + $signed(bias);
            wire [15:0]            y_bf;
            Requant_Bf16 #(.ACC_W(PSUM_W+1)) u_bf (
                .clk(clk), .rst(rst), .in_valid(in_valid && is_bf16),
                .acc(a_b), .scale(mult[15:0]),
                .out_valid(v_bf[g]), .out(y_bf));

            // ---- 출력 먹싱 ----
            assign out_data[g*16 +: 16] =
                   is_bf16 ? y_bf
                 : is_q411 ? (raw16 ? y_g_d3 : {{8{y_g8[ACT_W-1]}}, y_g8})
                 : req2     ? {{8{y_g8[ACT_W-1]}}, y_g8}
                 :             {{8{y_i8[ACT_W-1]}}, y_i8};
            assign out_q69[g*16 +: 16] = y_16;
        end
    endgenerate

    // 소비자마다 지연이 다릅니다 (Q4.11 은 GELU 를 거쳐 9단)
    assign out_valid = (consumer == C_INT8) ? (req2 ? v_g2[0] : v_i8[0])
                     : (consumer == C_Q411) ? v_g2[0]
                     : (consumer == C_BF16) ? v_bf[0]
                     :                        v_16[0];
endmodule
