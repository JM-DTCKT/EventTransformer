// ============================================================================
//  softmax_top.v  --  96-D softmax unit   (입력 Q6.9 x96 병렬 / 출력 UQ1.15 스트림)
// ----------------------------------------------------------------------------
//  y_i = exp(x_i - max) / sum_j exp(x_j - max)        (i = 0 .. N-1, N = 96)
//
//  [입력 인터페이스]  벡터 전체(N x DW = 1536-bit)를 1 clk 에 받는다.
//    upstream (QK^T score) 이 벡터를 한 번에 내주는 구조에 맞춘 것으로,
//    직렬로 96 clk 동안 긁어모으던 LOAD pass 가 통째로 사라진다.
//
//  [왜 그래도 버퍼가 필요한가]
//    softmax 는 max 와 sum 이라는 "전역" 의존성이 있어 x_95 를 보기 전에는
//    y_0 을 낼 수 없다.  1 clk 에 다 받더라도 1 clk 에 96개 exp 를 계산하지
//    않는 이상 96 word 를 어딘가 들고 있어야 한다 -> xbuf 는 제거 불가.
//    단, 96개를 동시에 써야 하므로 xbuf 는 SRAM 이 아니라 96x16 flip-flop 이다.
//
//  [P-lane exp 병렬]  exp2_unit 을 P개(기본 8) 깔아 매 clk P원소씩 처리한다.
//    lane j 는 step k 에서 원소 (k*P + j) 를 담당한다 (원소 i -> step i/P, lane i%P).
//    xbuf / ebuf 를 [step][lane] 2차원으로 두면 lane 당 NS:1 mux 만으로 접근된다.
//    exp 결과 P개는 가산 트리로 합쳐 분모 S 에 한 번에 누산한다.
//
//  [단계별 사이클]  (N=96, P=8 기준)
//    ST_LOAD :  in_vec 전체를 xbuf 에 래치 + max tree stage A         1 clk
//    ST_MAX  :  max tree stage B -> maxv 확정                         1 clk
//    ST_EXP  :  z = x - max (<=0) 를 P-lane exp2_unit 에 흘리고
//               결과를 ebuf 저장 + 가산트리로 S 누산    NS + 4 =     16 clk
//    ST_RCP  :  R = 1/S 를 recip_unit 으로 한 번만 계산                4 clk
//    ST_MUL  :  y_i = e_i * R  (나눗셈 대신 역수 곱셈)   NS + 1 =     13 clk
//                                                          합계      35 clk
//    * exp 와 출력 곱셈 모두 P-lane 이라 남은 최대 항목은 ST_EXP(16 clk) 다.
//      출력은 매 clk P원소씩 NS(=12) beat 로 나간다.
//    * [주의] 원소 단위는 완전 파이프라인이지만 "벡터 단위 중첩은 없다".
//      FSM 이 순차로 돌므로 in_ready 는 35 clk 중 34 clk 동안 low 이고,
//      35 clk 이 latency 이자 처리주기다 (평균 2.74 elem/clk, 8/clk 은 버스트).
//      지속 8 elem/clk 을 내려면 xbuf/ebuf 를 ping-pong 으로 2벌 두고
//      벡터 단위 파이프라이닝을 해야 한다 -> 하한 12 clk/vector.
//
//  * max 뺄셈 : exp 입력을 항상 <=0 으로 만들어 오버플로를 원천 차단하고
//               (e_i in (0,1]) exp LUT 의 정의역도 한쪽으로 고정시킨다.
//               최대 원소가 exp(0)=1.0 정확히 나오므로 S >= 1.0 이 보장된다.
//  * 나눗셈   : 분모 역수를 1회 구해 96번 곱한다.  나눗셈기 없이 곱셈기 1개.
//               역수는 벡터당 1회뿐이라 lane 을 늘릴 대상이 아니다.
//
//  [max tree]  N 을 2의 거듭제곱(NP=128)으로 패딩해 균일한 7-레벨 트리로 만든다.
//    패딩값은 표현 가능한 최소값(-64.0)이라 실제 max 를 절대 이기지 못하고,
//    상수끼리의 비교라 합성 시 전부 제거된다 (실질 비교기 N-1 = 95개).
//    7 레벨을 한 clk 에 태우면 500MHz 에서 빠듯하므로 4+3 레벨 2단으로 쪼갠다.
//
//  [고정소수점 포맷]
//    x (in_vec)   signed   Q6.9   16b   범위 [-64, 64),   LSB 2^-9  = 1.95e-3
//    z (내부)     signed   Q7.9   17b   = x - max,  범위 (-128, 0]
//    e (내부)     unsigned UQ1.16 17b   exp 결과 (0,1],   1.0 = 2^16
//    S (내부)     unsigned UQ8.16 24b   sum(e), 범위 [1, 96]
//    R (내부)     unsigned UQ1.17 18b   1/m  (m = S/2^p, m in [1,2))
//    y (out_data) unsigned UQ1.15 16b   범위 [0,1],       1.0 = 16'h8000
//
//    y = e/S = e * R * 2^-p  이므로  y15 = (e*R) >> (p + RF - OF) = >> (p+2)
//    * e(17b) x R(18b) = 18s x 19s -> DSP48 (25x18 / 27x18) 1개에 정확히 들어간다.
//
//  [핸드셰이크 / 타이밍]  (시뮬레이션 실측치, N=96 / P=8)
//    입력 : in_valid / in_ready 로 벡터 1개 = 1 handshake.
//           in_ready 는 idle 일 때만 high 이고, 연산 내내 low 를 유지하다가
//           마지막 출력(out_last)과 같은 사이클에 복귀한다 -> back-to-back 무버블.
//           별도 busy 출력은 정확히 ~in_ready 라 중복이므로 두지 않는다.
//    출력 : out_valid 가 NS(=12) clk 연속 high, 매 beat 마다 P(=8)원소.
//           마지막 beat 에서 out_last.  (출력측 back-pressure 없음)
//    latency  = 24 clk  (in_valid handshake -> 첫 out_valid, 첫 P원소)
//    벡터주기 = 35 clk  (2 load/max + 16 exp + 4 recip + 13 mul)
// ============================================================================
`timescale 1ns/1ps

module softmax_top #(
    parameter integer N  = 96,   // softmax 차원
    parameter integer P  = 8,    // exp lane 수 (N 의 약수, 2의 거듭제곱)
    parameter integer DW = 16,   // 입력 원소 폭 (signed Q6.9)
    parameter integer IF = 9,    // 입력 소수부 비트수
    parameter integer OW = 16,   // 출력 폭 (unsigned UQ1.15)
    parameter integer OF = 15,   // 출력 소수부 비트수
    parameter integer EW = 17,   // 내부 exp 폭 (unsigned UQ1.16, = EF+1)
    parameter integer EF = 16,   // 내부 exp 소수부 비트수
    parameter integer RW = 18,   // 역수 폭 (unsigned UQ1.17, = RF+1)
    parameter integer RF = 17    // 역수 소수부 비트수
)(
    input                        clk,
    input                        rst_n,
    // ---- 입력 : 벡터 전체를 1 clk 에 (원소 i = in_vec[i*DW +: DW]) ----
    input                        in_valid,
    output                       in_ready,
    input       [N*DW-1:0]       in_vec,      // Q6.9 x N, 원소 0 이 LSB 쪽
    // ---- 출력 : 매 clk P원소씩 NS beat (원소 j = out_data[j*OW +: OW]) ----
    output reg                   out_valid,
    output reg  [P*OW-1:0]       out_data,    // UQ1.15 x P, beat k 는 y[k*P .. k*P+P-1]
    output reg                   out_last
);
    // ---- width 파라미터 ----------------------------------------------------
    localparam integer AW      = $clog2(N);        // = 7,  원소 인덱스 폭
    localparam integer NS      = N / P;            // = 12, exp step 수
    localparam integer SW1     = $clog2(NS);       // = 4,  step 인덱스 폭
    localparam integer PL      = $clog2(P);        // = 3,  lane 인덱스 폭
    localparam integer ZW      = DW + 1;           // = 17, z = x - max 폭
    localparam integer SW      = EF + AW + 1;      // = 24, S = sum(e) 폭
                                                   //   S <= N*2^EF < 2^(EF+AW) 이므로 충분
    localparam integer ESW     = EW + PL;          // = 20, P개 exp 합(가산트리) 폭
    localparam integer PW      = 5;                // p (S 의 MSB 위치) 폭, p in [16,22]
    localparam integer SH_OFF  = RF - OF;          // = 2,  최종 시프트 오프셋
    localparam integer PDW     = EW + RW;          // = 35, e*R 곱 폭 (17b x 18b)

    // max tree : NP 로 패딩 후 LVL 레벨, LA + LB 두 단으로 파이프라인
    // Tree 구조로 max값을 찾기 위해 최솟값으로 padding
    localparam integer LVL     = $clog2(N);        // = 7,  전체 레벨 수
    localparam integer NP      = 1 << LVL;         // = 128, 패딩된 원소 수
    localparam integer LA      = (LVL + 1) / 2;    // = 4,  stage A 레벨 수
    localparam integer LB      = LVL - LA;         // = 3,  stage B 레벨 수
    localparam integer NA      = NP >> LA;         // = 8,  stage A 출력 개수
    localparam signed [DW-1:0] XPAD = {1'b1, {(DW-1){1'b0}}};  // 패딩값 = 최소값

    localparam [2:0] ST_LOAD = 3'd0,   // 벡터 래치 + max tree stage A
                     ST_MAX  = 3'd1,   // max tree stage B -> maxv 확정
                     ST_EXP  = 3'd2,   // P-lane exp 계산 + 분모 누산
                     ST_RCP  = 3'd3,   // 분모 역수
                     ST_MUL  = 3'd4;   // e * (1/S) 출력

    genvar  gl, gi, gj;
    integer bk, bj;

    // ---- 상태 / 버퍼 / 포인터 ---------------------------------------------
    //  버퍼는 [step][lane] 2차원.  원소 i <-> (step i/P, lane i%P).
    //  덕분에 P개 동시 접근이 lane 당 NS:1 mux 하나로 끝난다.
    reg [2:0]           state;
    reg signed [DW-1:0] xbuf [0:NS-1][0:P-1];  // 입력 버퍼 — N개 동시 write 라 FF
    reg        [EW-1:0] ebuf [0:NS-1][0:P-1];  // exp 결과 버퍼 (UQ1.17)
    reg signed [DW-1:0] maxv;                  // 벡터 최대값
    reg        [SW-1:0] sum;                   // 분모 S
    reg        [SW1:0]  sptr;                  // ST_EXP  exp 투입 step 포인터
    reg        [SW1:0]  cptr;                  // ST_EXP  exp 결과 수집 step 포인터
    reg        [SW1:0]  mptr;                  // ST_MUL  곱셈 step 포인터
    reg                 rcp_go;                // recip_unit 1-cycle start pulse

    // in_ready 하나로 상태가 다 드러난다 (high = idle, low = 연산 중).
    assign in_ready = (state == ST_LOAD);
    wire   load_hs  = in_valid & in_ready;

    // ======================================================================
    //  입력 언팩 : 1536-bit 버스 -> 원소 배열
    // ======================================================================
    wire signed [DW-1:0] xin [0:N-1];
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_unpack
            assign xin[gi] = in_vec[gi*DW +: DW];
        end
    endgenerate

    // ======================================================================
    //  max reduction tree  (stage A : 조합 LA 레벨,  NP -> NA)
    //    입력을 in_vec 에서 바로 받으므로 xbuf 래치와 같은 clk 에 처리된다.
    // ======================================================================
    wire signed [DW-1:0] ta [0:LA][0:NP-1];
    generate
        for (gi = 0; gi < NP; gi = gi + 1) begin : g_ta0
            if (gi < N) assign ta[0][gi] = xin[gi];
            else        assign ta[0][gi] = XPAD;      // 패딩 (합성 시 소거됨)
        end
        for (gl = 0; gl < LA; gl = gl + 1) begin : g_ta
            for (gi = 0; gi < (NP >> (gl+1)); gi = gi + 1) begin : g_node
                assign ta[gl+1][gi] = (ta[gl][2*gi] > ta[gl][2*gi+1])
                                    ?  ta[gl][2*gi] : ta[gl][2*gi+1];
            end
        end
    endgenerate

    reg signed [DW-1:0] ma_q [0:NA-1];   // stage A 결과 파이프라인 레지스터

    // ======================================================================
    //  max reduction tree  (stage B : 조합 LB 레벨,  NA -> 1)
    // ======================================================================
    wire signed [DW-1:0] tb [0:LB][0:NA-1];
    generate
        for (gi = 0; gi < NA; gi = gi + 1) begin : g_tb0
            assign tb[0][gi] = ma_q[gi];
        end
        for (gl = 0; gl < LB; gl = gl + 1) begin : g_tb
            for (gi = 0; gi < (NA >> (gl+1)); gi = gi + 1) begin : g_node
                assign tb[gl+1][gi] = (tb[gl][2*gi] > tb[gl][2*gi+1])
                                    ?  tb[gl][2*gi] : tb[gl][2*gi+1];
            end
        end
    endgenerate
    wire signed [DW-1:0] max_c = tb[LB][0];

    // ======================================================================
    //  Pass 1 : z = x - max  ->  P-lane exp2_unit  ->  ebuf / sum
    // ======================================================================
    // sptr 은 NS 까지 올라가므로 읽기 인덱스는 범위 안으로 묶어둔다 (X 전파 방지)
    wire [SW1-1:0] rd_step = (sptr < NS) ? sptr[SW1-1:0] : {SW1{1'b0}};
    wire           exp_iv  = (state == ST_EXP) && (sptr < NS);

    wire signed [ZW-1:0] z_w    [0:P-1];
    wire        [P-1:0]  exp_ov_v;
    wire        [EW-1:0] exp_e_v [0:P-1];

    generate
        for (gj = 0; gj < P; gj = gj + 1) begin : g_lane
            // 17비트로 부호확장 후 뺄셈 : z <= 0 보장 (max 가 벡터 최대값이므로)
            wire signed [DW-1:0] x_rd = xbuf[rd_step][gj];
            assign z_w[gj] = {x_rd[DW-1], x_rd} - {maxv[DW-1], maxv};

            exp2_unit #(.ZW(ZW), .ZF(IF), .EW(EW), .EF(EF)) u_exp (
                .clk(clk), .rst_n(rst_n),
                .in_valid(exp_iv), .z(z_w[gj]),
                .out_valid(exp_ov_v[gj]), .e(exp_e_v[gj])
            );
        end
    endgenerate
    wire exp_ov = exp_ov_v[0];      // 모든 lane 이 동일 valid 로 lockstep 동작

    // ---- P개 exp 결과 가산 트리 (PL 레벨) : S 에 한 번에 누산 ----
    wire [ESW-1:0] et [0:PL][0:P-1];
    generate
        for (gj = 0; gj < P; gj = gj + 1) begin : g_et0
            assign et[0][gj] = {{PL{1'b0}}, exp_e_v[gj]};
        end
        for (gl = 0; gl < PL; gl = gl + 1) begin : g_et
            for (gj = 0; gj < (P >> (gl+1)); gj = gj + 1) begin : g_node
                assign et[gl+1][gj] = et[gl][2*gj] + et[gl][2*gj+1];
            end
        end
    endgenerate
    wire [ESW-1:0] esum = et[PL][0];        // 이번 step 의 P원소 합

    // ======================================================================
    //  Pass 2-a : 분모 역수  R = 1/m,  S = m * 2^p   (벡터당 1회, lane 없음)
    // ======================================================================
    wire            rcp_ov;
    wire [RW-1:0]   rcp_r;
    wire [PW-1:0]   rcp_p;
    recip_unit #(.SW(SW), .RW(RW), .RF(RF), .PW(PW)) u_rcp (
        .clk(clk), .rst_n(rst_n),
        .in_valid(rcp_go), .s(sum),
        .out_valid(rcp_ov), .r(rcp_r), .p(rcp_p)
    );

    reg [RW-1:0] r_reg;     // 확정된 1/m
    reg [PW-1:0] p_reg;     // 확정된 p

    // ======================================================================
    //  Pass 2-b : y_i = e_i * R  (P-lane, 2-stage 출력 파이프라인)
    //     Stage A : ebuf 한 step(P원소) 읽기   Stage B : P개 곱셈 + 시프트/반올림
    //  R 과 시프트량 sh 는 벡터당 하나뿐이라 전 lane 이 공유(브로드캐스트)한다.
    //  exp lane 과 달리 복제할 LUT 이 없어서 곱셈기 P개만 늘면 된다.
    // ======================================================================
    wire            mul_iss = (state == ST_MUL) && (mptr < NS);
    reg             va, la;             // Stage A valid / last
    reg  [EW-1:0]   ea [0:P-1];         // Stage A 로 읽어온 e (P개)

    wire [5:0]     sh   = p_reg + SH_OFF;  // = p + 2, p in [16,22] -> sh in [18,24]
    wire [PDW-1:0] half = {{(PDW-1){1'b0}}, 1'b1} << (sh - 1);   // 0.5 LSB (공유)
    wire [PDW-1:0] prod [0:P-1];
    wire [PDW-1:0] yshf [0:P-1];
    generate
        for (gj = 0; gj < P; gj = gj + 1) begin : g_omul
            // e(17b) x R(18b) -> DSP48 (25x18 / 27x18) 1개, lane 당 1개씩
            assign prod[gj] = ea[gj] * r_reg;
            assign yshf[gj] = (prod[gj] + half) >> sh;           // UQ1.15 반올림
        end
    endgenerate

    // ======================================================================
    //  메인 FSM
    // ======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_LOAD;
            sptr      <= {(SW1+1){1'b0}};
            cptr      <= {(SW1+1){1'b0}};
            mptr      <= {(SW1+1){1'b0}};
            maxv      <= {DW{1'b0}};
            sum       <= {SW{1'b0}};
            rcp_go    <= 1'b0;
            r_reg     <= {RW{1'b0}};
            p_reg     <= {PW{1'b0}};
            va        <= 1'b0;
            la        <= 1'b0;
            for (bk = 0; bk < P; bk = bk + 1) ea[bk] <= {EW{1'b0}};
            out_valid <= 1'b0;
            out_data  <= {(P*OW){1'b0}};
            out_last  <= 1'b0;
            for (bk = 0; bk < NA; bk = bk + 1) ma_q[bk] <= {DW{1'b0}};
        end else begin
            rcp_go    <= 1'b0;      // 기본값 : 1-cycle pulse
            va        <= 1'b0;
            out_valid <= 1'b0;
            out_last  <= 1'b0;

            case (state)
            // ------------------------------- 벡터 래치 + max tree stage A
            ST_LOAD: begin
                if (load_hs) begin
                    for (bk = 0; bk < NS; bk = bk + 1)
                        for (bj = 0; bj < P; bj = bj + 1)
                            xbuf[bk][bj] <= xin[bk*P + bj];
                    for (bk = 0; bk < NA; bk = bk + 1) ma_q[bk] <= ta[LA][bk];
                    state <= ST_MAX;
                end
            end
            // ------------------------------- max tree stage B -> maxv 확정
            ST_MAX: begin
                maxv  <= max_c;
                state <= ST_EXP;
                sptr  <= {(SW1+1){1'b0}};
                cptr  <= {(SW1+1){1'b0}};
                sum   <= {SW{1'b0}};
            end
            // --------------------------- P-lane exp + 분모 누산 (NS step)
            ST_EXP: begin
                if (sptr < NS) sptr <= sptr + 1'b1;         // 매 clk P원소 투입
                if (exp_ov) begin                            // 4 clk 뒤부터 결과 도착
                    for (bj = 0; bj < P; bj = bj + 1)
                        ebuf[cptr[SW1-1:0]][bj] <= exp_e_v[bj];
                    sum  <= sum + {{(SW-ESW){1'b0}}, esum};  // 가산트리 결과를 누산
                    cptr <= cptr + 1'b1;
                    if (cptr == NS-1) begin                  // 전체 누산 완료
                        state  <= ST_RCP;
                        rcp_go <= 1'b1;                      // 이 시점 sum 이 최종값
                    end
                end
            end
            // ------------------------------------------------- 분모 역수
            ST_RCP: begin
                if (rcp_ov) begin
                    r_reg <= rcp_r;
                    p_reg <= rcp_p;
                    state <= ST_MUL;
                    mptr  <= {(SW1+1){1'b0}};
                end
            end
            // -------------------------------------- e * R -> 출력 (P원소/clk)
            ST_MUL: begin
                // Stage A : ebuf 한 step(P원소) 읽기
                if (mul_iss) begin
                    va <= 1'b1;
                    for (bj = 0; bj < P; bj = bj + 1)
                        ea[bj] <= ebuf[mptr[SW1-1:0]][bj];
                    la   <= (mptr == NS-1);
                    mptr <= mptr + 1'b1;
                end
                // Stage B : P개 곱셈 결과를 한 beat 로 출력
                if (va) begin
                    out_valid <= 1'b1;
                    for (bj = 0; bj < P; bj = bj + 1)
                        out_data[bj*OW +: OW] <= yshf[bj][OW-1:0];
                    out_last  <= la;
                    if (la) state <= ST_LOAD;                // 마지막 -> 다음 벡터
                end
            end
            default: state <= ST_LOAD;
            endcase
        end
    end

    // ---- 파라미터 정합성 체크 (합성에는 영향 없음) ------------------------
    initial begin
        // y15 = (e17 * R20) >> (p + RF - OF) 이므로 RF > OF 여야 한다
        if (SH_OFF < 1)
            $display("ERROR: softmax_top RF(%0d) must be > OF(%0d)", RF, OF);
        // S = sum(e) 가 SW 비트에 담겨야 한다 : N * 2^EF < 2^SW
        if (SW <= EF + AW)
            $display("ERROR: softmax_top SW(%0d) too small for N=%0d, EF=%0d", SW, N, EF);
        // lane 분할이 딱 떨어져야 하고, P 는 2의 거듭제곱이어야 한다
        if (N % P != 0)
            $display("ERROR: softmax_top N(%0d) must be divisible by P(%0d)", N, P);
        if ((1 << PL) != P)
            $display("ERROR: softmax_top P(%0d) must be a power of 2", P);
        // e, R 은 각각 1.0(=2^EF, 2^RF) 을 담아야 하므로 폭이 소수부+1
        if (EW != EF+1) $display("ERROR: softmax_top EW must be EF+1");
        if (RW != RF+1) $display("ERROR: softmax_top RW must be RF+1");
    end
endmodule
