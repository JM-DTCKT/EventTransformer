// -----------------------------------------------------------------------------
// Format_Cast_Act : GEMM 컬럼(32레인 INT32) → 출력 포맷 + 활성함수
//
// bias 덧셈 · 역양자화/재양자화 · 포맷 변환 · 활성함수를 한 파이프라인에
// 융합합니다 — 별도 패스 없이 GEMM 출력에 그대로 붙어 흐릅니다.
//
// ## 출력 포맷 (`fmt`)
//
//   FMT_INT8 (0)  requant(acc+bias, mult, shift) → [활성함수] → int8
//                 req2=1 이면 활성함수 뒤 한 번 더 재양자화       A_Mem 하위 8b
//   FMT_Q411 (1)  requant → Q4.11 → GELU → requant → int8        A_Mem 하위 8b
//                 raw16=1 이면 마지막 재양자화를 건너뜀           A_Mem 16b
//   FMT_BF16 (2)  bf16( (acc+bias) · scale[n] )                   A_Mem 16b
//   FMT_Q69  (3)  requant → Q6.9                                  Softmax_Top 직결
//
// ## 지연 — `EvT_Engine.FCA_LAT_*` 와 한 벌입니다
//
//   FMT_BF16       2 사이클
//   FMT_INT8/Q69   3 사이클        (활성함수는 조합이라 안 늘어남)
//   FMT_INT8+req2  6 사이클
//   FMT_Q411       9 사이클        3(requant) + 3(GELU) + 3(requant)
//
// 단수를 바꾸면 `EvT_Engine` 의 `FCA_LAT_*` 도 **반드시 같이** 고쳐야 합니다 —
// 쓰기 주소 파이프가 그만큼 밀려 결과가 한 칸 앞 주소에 써집니다.
//
// ## 여기서 특별대우가 없는 것 둘
//
// **in_proj** 은 384행이 Q/K/V 세 구간이지만 구간마다 다른 출력 scale 이 채널별
// 곱수 M[c] 에 이미 접혀 있어 여기서는 한 덩어리입니다. (Q 는 `layer_norm_1`,
// K·V 는 `layer_norm_x` 를 읽어 A 피연산자가 다르므로 하드웨어에서는 GEMM 이
// 두 번이지만, 그건 스케줄러가 `a_base`/`b_base` 를 달리 주는 것으로 끝납니다.)
//
// **V 의 전치**도 여기서 안 합니다 — 컬럼을 평소처럼 내보내고 `Transpose32` 가
// 축을 돌립니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Format_Cast_Act #(
    parameter N       = 32,
    parameter ACT_W   = 8,
    parameter PSUM_W  = 32,
    // 재양자화 상수를 몇 레인마다 다시 뜰지 (아래 "상수 복제" 참고)
    parameter LANES_PER_COPY = 8
)(
    input  wire                     clk,
    input  wire                     rst,

    input  wire [1:0]               fmt,   // 0 int8  1 Q4.11+GELU  2 bf16  3 Q6.9
    // [타이밍] 아래 5개는 **레지스터를 거치지 않은 채** 받습니다. 여기서 레인
    // 그룹마다 다시 뜨기 때문입니다 (`bias_l[]` 등, 아래 "상수 복제" 참고).
    input  wire signed [PSUM_W-1:0] bias,       // b_int[n]
    input  wire signed [PSUM_W-1:0] mult,       // M[n]  또는  scale(bf16, 하위 16b)
    input  wire [5:0]               shift,
    input  wire signed [PSUM_W-1:0] g_mult,     // GELU 뒤 int8 재양자화 (스칼라)
    input  wire [5:0]               g_shift,
    input  wire [1:0]               act_sel,    // 0 없음  1 ReLU  (Activation.v 인코딩)
    input  wire [7:0]               act_parm,
    // FMT_Q411 에서 **int8 재양자화를 건너뛰고 16비트 그대로** 내보냅니다.
    // `linear1` 뒤가 LayerNorm 이라 int8 격자가 없는 자리에 씁니다 (LayerNorm 은
    // 스케일 불변이라 골든과 결과가 완전히 같습니다).
    input  wire                     raw16,
    // FMT_INT8 에서 **활성함수 뒤 한 번 더** 재양자화합니다 (g_mult/g_shift).
    // 골든이 ReLU 앞뒤로 격자를 달리 쓰는 자리(`proc_ev.1`, `clf.linear_1`) 용.
    input  wire                     req2,

    input  wire                     in_valid,
    input  wire [N*PSUM_W-1:0]      acc,

    output wire                     out_valid,
    output wire [N*16-1:0]          out_data,   // A_Mem 워드 (int8 은 부호확장)
    output wire [N*16-1:0]          out_q69     // Softmax 로 가는 Q6.9 (fmt=3)
);
    localparam FMT_INT8=2'd0, FMT_Q411=2'd1, FMT_BF16=2'd2, FMT_Q69=2'd3;

    wire is_q411 = (fmt == FMT_Q411);
    wire is_bf16 = (fmt == FMT_BF16);

    wire [N-1:0] v_i8, v_16, v_bf, v_g2;

    // 활성함수 앞 / 뒤 버스 (활성함수는 32레인 단위라 루프 밖에서 겁니다)
    wire [N*ACT_W-1:0] i8r_bus, i8a_bus;
    Activation #(.DATA_W(ACT_W), .N(N)) u_act (
        .act_sel(act_sel), .act_parm(act_parm), .din(i8r_bus), .dout(i8a_bus));

    // =========================================================================
    // 상수 복제 — 재양자화 상수를 LANES_PER_COPY 레인마다 다시 뜹니다
    //
    // 엔진에서 한 번만 뜨면 그 플롭 하나가 32레인 x DSP 2계통으로 뻗어,
    // `bias -> DSP 프리애더 -> 곱셈기 -> ALU` 앞에 배선 2.3 ns 가 붙습니다
    // (실측 WNS -0.187 ns, LANE[20].u_bf 경로). CLB 96 % 라 `max_fanout` 복제
    // 지시만으로는 복제본이 레인 근처에 자리를 못 잡습니다.
    //
    // 레인마다(32벌) 뜨면 배선은 가장 짧지만 FF 가 2,000개 넘게 늘어 오히려
    // 혼잡을 키웁니다. 8레인당 1벌 = 4벌이면 팬아웃이 1/4 로 줄면서 FF 증가는
    // 300개 아래입니다.
    //
    // 엔진의 플롭을 **여기로 옮긴 것**이라 bias/mult 의 지연은 그대로입니다
    // (컬럼마다 바뀌는 값이라 한 사이클도 어긋나면 안 됩니다). shift 계열은
    // 명령어 내내 상수라 한 단 늦어도 무해합니다.
    // `dont_touch` 가 없으면 합성기가 복제본을 도로 하나로 합칩니다.
    // =========================================================================
    localparam NCOPY = (N + LANES_PER_COPY - 1) / LANES_PER_COPY;

    (* dont_touch = "true" *) reg signed [PSUM_W-1:0] bias_l   [0:NCOPY-1];
    (* dont_touch = "true" *) reg signed [PSUM_W-1:0] mult_l   [0:NCOPY-1];
    (* dont_touch = "true" *) reg signed [PSUM_W-1:0] g_mult_l [0:NCOPY-1];
    (* dont_touch = "true" *) reg        [5:0]        shift_l  [0:NCOPY-1];
    (* dont_touch = "true" *) reg        [5:0]        g_shift_l[0:NCOPY-1];

    integer c;
    always @(posedge clk) begin
        for (c = 0; c < NCOPY; c = c + 1) begin
            bias_l  [c] <= bias;     mult_l   [c] <= mult;    shift_l[c] <= shift;
            g_mult_l[c] <= g_mult;   g_shift_l[c] <= g_shift;
        end
    end

    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : LANE
            localparam CP = g / LANES_PER_COPY;    // 이 레인이 쓸 복제본
            wire signed [PSUM_W-1:0] a = acc[g*PSUM_W +: PSUM_W];

            // ---- INT8 직행 (fmt 0) + 활성함수 ----
            wire signed [ACT_W-1:0] y_i8r;
            Requant_Int #(.ACC_W(PSUM_W), .MUL_W(PSUM_W), .SH_W(6), .OUT_W(ACT_W),
                          .PIPE_PRE(1), .UNSIGNED_OUT(0), .USE_BIAS(1), .BIAS_W(PSUM_W)) u_i8 (
                .clk(clk), .rst(rst), .in_valid(in_valid && (fmt == FMT_INT8)),
                .acc(a), .bias(bias_l[CP]), .mult(mult_l[CP]), .shift(shift_l[CP]),
                .out_valid(v_i8[g]), .out(y_i8r));

            // 활성함수는 **32레인 버스 단위**라 루프 밖에서 한 번 겁니다.
            assign i8r_bus[g*ACT_W +: ACT_W] = y_i8r;
            wire signed [ACT_W-1:0] y_i8 = i8a_bus[g*ACT_W +: ACT_W];

            // ---- 16비트 Q포맷 (fmt 1 = Q4.11, 3 = Q6.9) ----
            wire signed [15:0] y_16;
            Requant_Int #(.ACC_W(PSUM_W), .MUL_W(PSUM_W), .SH_W(6), .OUT_W(16),
                          .PIPE_PRE(1), .UNSIGNED_OUT(0), .USE_BIAS(1), .BIAS_W(PSUM_W)) u_16 (
                .clk(clk), .rst(rst),
                .in_valid(in_valid && (is_q411 || (fmt == FMT_Q69))),
                .acc(a), .bias(bias_l[CP]), .mult(mult_l[CP]), .shift(shift_l[CP]),
                .out_valid(v_16[g]), .out(y_16));

            // ---- GELU (Q4.11 → Q4.11) → int8 ----
            // 64세그먼트 PWL — 전수 LUT(레인당 8 BRAM36 = 256개)을 base/delta
            // 64쌍 = 2 Kb 로 줄인 것입니다. 정확한 GELU 대비 최대 1 LSB.
            // 지연 3사이클, 리셋만 **active-low** 입니다 (나머지는 active-high).
            wire               vg;
            wire signed [15:0] y_g;
            Gelu_Pwl #(.W(16), .QF(11), .FR(14)) u_gelu (
                .clk(clk), .rst_n(~rst), .in_valid(v_16[g] && is_q411), .x(y_16),
                .out_valid(vg), .y(y_g));

            // 2차 재양자화기 하나를 **GELU 경로와 int8 경로가 나눠** 씁니다 —
            // 두 포맷이 동시에 도는 일이 없어 곱수/시프트도 같이 씁니다.
            wire signed [ACT_W-1:0] y_g8;
            wire signed [15:0] r2_in = is_q411 ? y_g : {{8{y_i8[ACT_W-1]}}, y_i8};
            Requant_Int #(.ACC_W(16), .MUL_W(PSUM_W), .SH_W(6), .OUT_W(ACT_W),
                          .PIPE_PRE(1), .UNSIGNED_OUT(0), .USE_BIAS(0), .BIAS_W(PSUM_W)) u_g8 (
                .clk(clk), .rst(rst),
                .in_valid(is_q411 ? vg : (v_i8[g] && req2)),
                .acc(r2_in), .bias({PSUM_W{1'b0}}), .mult(g_mult_l[CP]), .shift(g_shift_l[CP]),
                .out_valid(v_g2[g]), .out(y_g8));

            // raw16 은 마지막 재양자화(3단)를 안 거치므로 그만큼 늦춰 지연을
            // 맞춥니다 — 안 맞추면 `FCA_LAT_*` 가 하나 더 갈라집니다.
            reg signed [15:0] y_g_d1, y_g_d2, y_g_d3;
            always @(posedge clk) begin
                y_g_d1 <= y_g;
                y_g_d2 <= y_g_d1;
                y_g_d3 <= y_g_d2;
            end

            // ---- bf16 : bias 를 먼저 더하고 scale 을 곱함 ----
            //   골든은 bf16(acc · s_x·s_w[c]) 이고 acc 에 b_int 가 이미 들어 있음
            wire signed [PSUM_W:0] a_b = $signed(a) + $signed(bias_l[CP]);
            wire [15:0]            y_bf;
            Requant_Bf16 #(.ACC_W(PSUM_W+1)) u_bf (
                .clk(clk), .rst(rst), .in_valid(in_valid && is_bf16),
                .acc(a_b), .scale(mult_l[CP][15:0]),
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
    assign out_valid = (fmt == FMT_INT8) ? (req2 ? v_g2[0] : v_i8[0])
                     : (fmt == FMT_Q411) ? v_g2[0]
                     : (fmt == FMT_BF16) ? v_bf[0]
                     :                        v_16[0];
endmodule
