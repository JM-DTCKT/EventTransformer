// -----------------------------------------------------------------------------
// Gemm_Core : C = A · B  (output-stationary, 32x32 systolic)
//
// **B 를 어느 메모리에서 읽을지는 코어가 정하지 않습니다** — 주소만 내보내고
// 답하는 쪽은 `EvT_Engine` 이 고릅니다. EvT 의 GEMM 세 종류가 그 하나만 다르고
// 회로는 완전히 같기 때문입니다:
//
//   ① Linear     C[m][n] = Σ_k A[m][k]·W[n][k]      B = 가중치  (W_Mem)
//   ② Q·Kᵀ       C[m][n] = Σ_d Q[m][d]·K[n][d]      B = 활성값  (A_Mem)
//   ③ attn·V     C[m][n] = Σ_j attn[m][j]·V[j][n]   B = 활성값  (A_Mem)
//
// ## 레이아웃이 저절로 맞습니다
//
// 코어는 A·B 둘 다 "워드 = reduce 인덱스, 레인 = non-reduce 인덱스" 로 읽습니다.
//
//     A_Mem[a_base + mt*K + k] 레인 i = A[mt*32+i][k]
//     B    [b_base + nt*K + k] 레인 j = B[k][nt*32+j]
//
//   ② in_proj 은 출력채널 c 마다 32행 컬럼을 뱉습니다 → 워드[c] 레인 = 토큰.
//      Q·Kᵀ 의 reduce 가 d(=c) 이므로 Q·K 둘 다 그대로 맞고, head 분할도
//      공짜입니다 (head h 는 `a_base + h*32`, K=32).
//   ③ score GEMM 은 키 n 마다 컬럼을 뱉습니다 → 워드[j] 레인 = 쿼리. attn·V 의
//      A 가 정확히 이 모양입니다. **V 만** 워드[j] 레인 = d 가 필요해
//      `Transpose32` 로 축을 돌립니다 (reduce 축이 d→j 로 바뀌는 유일한 자리).
//
// ③의 A 는 softmax 출력 uint8 [0,127] 이라 int8 해석과 값이 같습니다 — `PE_OS`
// 의 signed x signed 를 그대로 씁니다.
//
// ## M > 32
//
// 토큰 최대 123(4타일), latent 96(3타일)이라 행 타일링이 필수입니다. 아래
// 순서기가 `mt` 를 내므로 **소비자가 `col_mt` 로 행 타일을 구분해 되쓰면**
// 됩니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Gemm_Core #(
    parameter N      = 32,
    parameter ACT_W  = 8,
    parameter PSUM_W = 32,
    parameter DIM_W  = 16,
    parameter AW_A   = 12,      // A_Mem 워드 주소폭
    parameter AW_B   = 14,      // B 주소폭 (W_Mem 기준, A_Mem 도 담김)
    parameter PK_S   = 21,      // DSP 패킹 : 상위 곱의 자리
    parameter SH_W   = 41,      // shadow 폭
    parameter RD_LEAD = 1       // 읽기 시점의 여유 사이클 (0 이 이론 하한)
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    output wire                    all_done,

    // 1 = B 가 int4 두 개씩 담긴 W_Mem (Linear). 타일이 32x64 가 되고 컬럼을
    //     64개 뽑습니다. 0 = B 가 A_Mem 의 int8 (attention), 32x32 그대로.
    input  wire                    pack,

    // ---- 형상 / 베이스 (레이어마다 다름, start 이전에 안정) ----
    input  wire [DIM_W-1:0]        M,
    input  wire [DIM_W-1:0]        K,          // in_features
    input  wire [DIM_W-1:0]        Nout,       // out_features
    input  wire [AW_A-1:0]         a_base,
    input  wire [AW_B-1:0]         b_base,

    // ---- A_Mem 읽기 포트 ----
    output wire                    a_rd_en,
    output wire [AW_A-1:0]         a_rd_addr,
    input  wire [N*16-1:0]         a_rd_data,   // 16비트 레인 (하위 8b 사용)

    // ---- B 읽기 포트 — **어느 메모리인지는 코어가 모릅니다** ----
    //   Linear   → W_Mem
    //   Q·Kᵀ     → A_Mem (하위 8비트)
    //   attn·V   → A_Mem 의 전치된 V 영역
    output wire                    b_rd_en,
    output wire [AW_B-1:0]         b_rd_addr,
    input  wire [N*ACT_W-1:0]      b_rd_data,   // pack 이면 레인당 {w1,w0}

    // ---- 완성된 컬럼 (출력채널 n 하나에 대한 32개 행) ----
    output reg                     col_valid,
    output reg  [N*PSUM_W-1:0]     col_data,   // col_data[i] = acc[mt*32+i][n]
    output reg  [DIM_W-1:0]        col_n,      // 전역 출력채널 n
    output reg  [DIM_W-1:0]        col_mt      // 출력 행 타일
);
    // =========================================================================
    // ping-pong 누산기 + 타일 파이프라인 — 타일 주기 max(K+4, 36)
    //
    // ## 파면 시각
    //
    //   주소 c 를 tile_ph 에 내고 BRAM 이 다음 사이클에 답함  → skew din 의 k
    //   Skew_Buf 레인 i 는 i 사이클 뒤    → 레인 i 가 k 를 내는 시각 k + 1 + i
    //   PE[i][j] 는 a_reg/b_reg 를 j/i 홉 지나서               → k + 1 + i + j
    //   DSP48E2 (AREG=BREG=0, MREG=1, PREG=1) → P 반영은 +2
    //
    //   ⇒ PE[i][j] 확정 = K + i + j + 2
    //
    // PE 마다 끝나는 시각이 다릅니다. `PE_OS` 는 그 시각에 `shadow <= P` 를 하고
    // (snap 파면) 다음 사이클에 `P <= 0` 을 합니다(clr 파면). 읽기는 shadow 에서
    // 하므로 **배열은 곧바로 다음 타일을 받습니다.**
    //
    // ## 주기를 정하는 부등식
    //
    //   SNAP_PH = K + 3                  PE[i][j] 스냅 = SNAP_PH + i + j
    //   컬럼 j 가 shadow 에 다 차는 시각 = SNAP_PH + (N-1) + j = K + 34 + j
    //   컬럼 j 를 읽는 시각              = K + 34 + j + RD_LEAD
    //
    //   ① clr 는 snap 뒤                 TILE_P >= SNAP_PH + 1 = K + 4
    //   ② 다음 타일이 shadow 를 덮기 전에 읽기 (i=0 최악, RD_LEAD=1) ⇒ TILE_P > 32
    //   ③ 컬럼 32개를 1/사이클로 소비     TILE_P >= 32
    //
    //   ⇒ TILE_P = max(K + 4, 36)        K=128 → 132,  K=32 → 36
    //
    // ## 읽기는 한 타일 뒤에서 돕니다
    //
    // 컬럼을 뽑는 시점(K+35)은 이미 다음 타일의 주기 안입니다. 그래서 읽기는
    // `mt_rd_pend`/`nt_rd_pend`(직전 타일)을 쓰고, 마지막 타일을 위해 꼬리에서
    // 주기를 한 번 더 돕니다. 타일이 겹쳐 흐르므로 순서기도 여기 전용입니다.
    // =========================================================================
    // 패킹이면 컬럼 타일이 64 폭입니다.
    wire [DIM_W-1:0] col_per_tile = pack ? 16'd64 : 16'd32;
    wire [5:0]       rd_last      = pack ?  6'd63 :  6'd31;
    wire [DIM_W-1:0] num_mt  = (M + N - 1) / N;
    wire [DIM_W-1:0] num_nt  = pack ? ((Nout + 16'd63) >> 6) : ((Nout + 16'd31) >> 5);
    // DSP 에 ADREG/BREG 가 붙어 지연이 3 입니다 (전에는 2) — 한 사이클씩 밀립니다.
    wire [DIM_W-1:0] snap_tile_ph = K + 16'd4;
    // ① snap 펄스가 주기 **안에** 떨어져야 합니다 — `tile_ph` 는 0..TILE_P-1 만
    //    돌므로 TILE_P >= snap_tile_ph + 1 = K + 5 입니다. (K+4 로 두면 K=32 인
    //    attention 에서 tile_ph 가 36 에 닿지 못해 snap 이 **한 번도 안 뜹니다**.)
    // ② 컬럼을 col_per_tile 개 소비해야 하므로 그만큼도 필요합니다.
    wire [DIM_W-1:0] k_plus5     = K + 16'd5;
    wire [DIM_W-1:0] tp_min      = col_per_tile + 16'd4;
    wire [DIM_W-1:0] tile_period = (k_plus5 > tp_min) ? k_plus5 : tp_min;
    // 타일 T 의 컬럼 0 은 T*tile_period + K+34+RD_LEAD 에 뽑습니다. 그때는 이미
    // 타일 T+1 의 주기이므로 위상은 그만큼 뺀 값입니다.
    wire [DIM_W-1:0] rd_trig_tile_ph = K + (35 + RD_LEAD - 1) - tile_period;

    localparam ST_IDLE=2'd0, ST_ISSUE=2'd1, ST_TAIL=2'd2, ST_DONE=2'd3;
    reg [1:0]        state;
    reg [DIM_W-1:0]  tile_ph;                 // 발행 중인 타일의 위상
    reg [DIM_W-1:0]  mt_iss,  nt_iss;        // 발행 중인 타일 idx
    reg [DIM_W-1:0]  mt_rd_pend, nt_rd_pend;       // Core에서 읽을 차례인(발행이 끝난) 타일 idx
    reg              tile_start;             // clr 파면 주입
    reg              rd_arm;             // 첫 주기에는 읽을 타일이 없습니다 -> 첫 타일 끝에서 1이 된다.
    reg              all_done_r;
    reg [AW_A-1:0]   a_tile_base;
    reg [AW_B-1:0]   b_tile_base;

    // 다음 타일 인덱스를 미리 계산합니다 — 베이스 곱셈을 주소 경로에서 빼내려면
    // `mt` 가 바뀌는 **그 엣지에** 베이스도 같이 확정돼야 합니다.
    wire             nt_last   = (nt_iss == num_nt - 1'b1);
    wire [DIM_W-1:0] mt_next     = nt_last ? (mt_iss + 1'b1) : mt_iss;
    wire [DIM_W-1:0] nt_next     = nt_last ? {DIM_W{1'b0}} : (nt_iss + 1'b1);
    wire             last_tile = nt_last && (mt_iss == num_mt - 1'b1);

    reg              rd_run;
    reg [5:0]        rd_cnt;
    reg [DIM_W-1:0]  mt_rd, nt_rd; // Core에서 읽는 중인 Tile idx

    always @(posedge clk) begin
        tile_start <= 1'b0;
        if (rst) begin
            state <= ST_IDLE; tile_ph <= {DIM_W{1'b0}};
            mt_iss  <= {DIM_W{1'b0}}; nt_iss  <= {DIM_W{1'b0}};
            mt_rd_pend <= {DIM_W{1'b0}}; nt_rd_pend <= {DIM_W{1'b0}};
            rd_arm <= 1'b0; all_done_r <= 1'b0;
            a_tile_base <= {AW_A{1'b0}}; b_tile_base <= {AW_B{1'b0}};
        end else begin
            case (state)
                ST_IDLE: begin
                    all_done_r <= 1'b0; rd_arm <= 1'b0;
                    if (start) begin
                        mt_iss <= {DIM_W{1'b0}}; nt_iss <= {DIM_W{1'b0}};
                        tile_ph   <= {DIM_W{1'b0}}; tile_start <= 1'b1;
                        a_tile_base <= a_base;
                        b_tile_base <= b_base;
                        state <= ST_ISSUE;
                    end
                end
                ST_ISSUE: begin
                    if (tile_ph == tile_period - 1'b1) begin // Tile 경계
                        tile_ph     <= {DIM_W{1'b0}};
                        mt_rd_pend  <= mt_iss;  nt_rd_pend <= nt_iss;   // 읽기는 한 타일 뒤
                        rd_arm <= 1'b1;
                        if (last_tile) state <= ST_TAIL;
                        else begin
                            mt_iss <= mt_next; nt_iss <= nt_next; tile_start <= 1'b1;
                            a_tile_base <= a_base + mt_next * K;
                            b_tile_base <= b_base + nt_next * K;
                        end
                    end else tile_ph <= tile_ph + 1'b1;
                end
                // 마지막 타일을 읽기 위해 주기를 한 번 더 (발행은 없음)
                ST_TAIL: begin
                    tile_ph <= tile_ph + 1'b1;
                    if ((tile_ph > rd_trig_tile_ph) && !rd_run) begin
                        all_done_r <= 1'b1; state <= ST_DONE;
                    end
                end
                // `all_done` 은 start 가 내려갈 때까지 유지하는 규약입니다
                ST_DONE: if (!start) state <= ST_IDLE;
            endcase
        end
    end

    assign all_done = all_done_r;

    // =========================================================================
    // 주소 생성 :  a_tile_base + tile_ph   /   b_tile_base + tile_ph
    // =========================================================================
    wire issuing = (state == ST_ISSUE) && (tile_ph < K);

    assign a_rd_en   = issuing;
    assign b_rd_en   = issuing;
    assign a_rd_addr = a_tile_base + tile_ph[AW_A-1:0];
    assign b_rd_addr = b_tile_base + tile_ph[AW_B-1:0];

    // BRAM 1클럭 지연 흡수 — 데이터는 tile_ph+1 에 옵니다
    reg issuing_d1, array_ce;
    always @(posedge clk) begin
        if (rst) begin issuing_d1 <= 1'b0; array_ce <= 1'b0; end
        else begin
            issuing_d1 <= issuing;
            array_ce <= (state == ST_ISSUE) || (state == ST_TAIL);
        end
    end

    // =========================================================================
    // 두 파면 : 펄스를 레인 i 만큼 늦춰 배열 왼쪽에 넣습니다
    //   clr  = 타일 시작(tile_ph 0)      → PE[i][j] 에 i+j 뒤 도착
    //   snap = 확정 시점(tile_ph K+3)    → 마찬가지
    // (A 의 `Skew_Buf` 와 같은 삼각형 지연을 1비트로 한 것)
    // =========================================================================
    wire snap_pulse = (state == ST_ISSUE) && (tile_ph == snap_tile_ph);

    reg  [N-1:0] clr_wave,  snap_wave;
    wire [N-1:0] clr_edge  = {clr_wave [N-2:0], tile_start};
    wire [N-1:0] snap_edge = {snap_wave[N-2:0], snap_pulse};
    always @(posedge clk) begin
        if (rst) begin clr_wave <= {N{1'b0}}; snap_wave <= {N{1'b0}}; end
        else     begin clr_wave <= clr_edge;  snap_wave <= snap_edge;  end
    end

    // =========================================================================
    // edge mask (skew **앞**) → skew
    //
    // [함정] 마스크는 skew **앞**에서 걸어야 합니다. 뒤에서 걸면 두 타일이 동시에
    // 흐를 때 배열 안 데이터는 이전 타일 것인데 `mt`/`nt` 는 이미 다음 것이라
    // 어긋납니다. 마스크는 레인마다 상수라 주입 시점에 걸어도 결과가 같습니다.
    //
    // A_Mem 은 16비트 레인이지만 PE 는 int8 을 받습니다 — 활성값이 하위 8비트에
    // 부호확장돼 저장되므로 그대로 잘라 씁니다.
    // =========================================================================
    wire [N*ACT_W-1:0] a_masked, b_masked;
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : g_edge_mask
            wire row_ok = ((mt_iss << 5) + g) < M;
            // 패킹이면 레인 g 가 출력채널 두 개를 담습니다 — 하위 니블 = nt*64+g,
            // 상위 니블 = nt*64+32+g. Nout 이 64 의 배수가 아니면 한쪽만 삽니다.
            wire col_ok    = ((nt_iss << 5) + g) < Nout;
            wire col_ok_lo = ((nt_iss << 6) + g) < Nout;
            wire col_ok_hi = ((nt_iss << 6) + 32 + g) < Nout;
            assign a_masked[g*ACT_W +: ACT_W] =
                   row_ok ? a_rd_data[g*16 +: ACT_W]    : {ACT_W{1'b0}};
            assign b_masked[g*ACT_W +: ACT_W] =
                   pack ? {col_ok_hi ? b_rd_data[g*ACT_W+4 +: 4] : 4'd0,
                           col_ok_lo ? b_rd_data[g*ACT_W   +: 4] : 4'd0}
                        : (col_ok ? b_rd_data[g*ACT_W +: ACT_W] : {ACT_W{1'b0}});
        end
    endgenerate

    wire [N*ACT_W-1:0] a_skew, b_skew;

    // `clr` 를 안 겁니다 — 타일이 연달아 흐르므로 flush 하면 직전 타일의 꼬리가
    // 잘립니다. `din_valid` 가 0 인 구간은 Skew_Buf 가 알아서 0 을 냅니다.
    Skew_Buf #(.N(N), .W(ACT_W)) u_askew (
        .clk(clk), .rst(rst), .clr(1'b0),
        .din(a_masked), .din_valid(issuing_d1), .dout(a_skew), .dout_valid());

    Skew_Buf #(.N(N), .W(ACT_W)) u_bskew (
        .clk(clk), .rst(rst), .clr(1'b0),
        .din(b_masked), .din_valid(issuing_d1), .dout(b_skew), .dout_valid());

    // =========================================================================
    // PE 배열 — `acc` 는 **shadow** 라 읽는 동안 값이 안 변합니다
    // =========================================================================
    wire [N*N*SH_W-1:0] acc;
    PE_Array #(.N(N), .ACT_W(ACT_W), .PK_S(PK_S), .SH_W(SH_W)) u_array (
        .clk(clk), .rst(rst), .ce(array_ce), .pack(pack),
        .clr_edge(clr_edge), .snap_edge(snap_edge),
        .a_edge(a_skew), .b_edge(b_skew), .acc_out(acc));

    // =========================================================================
    // 컬럼 읽어내기 — shadow 에서, 다음 타일이 도는 중에
    // =========================================================================
    wire rd_trig = rd_arm && (tile_ph == rd_trig_tile_ph)
                && ((state == ST_ISSUE) || (state == ST_TAIL));

    always @(posedge clk) begin
        if (rst) begin
            rd_run <= 1'b0; rd_cnt <= 6'd0;
            mt_rd <= {DIM_W{1'b0}}; nt_rd <= {DIM_W{1'b0}};
        end else if (rd_trig) begin
            rd_run <= 1'b1; rd_cnt <= 6'd0; mt_rd <= mt_rd_pend; nt_rd <= nt_rd_pend;
        end else if (rd_run) begin
            if (rd_cnt == rd_last) rd_run <= 1'b0;
            else rd_cnt <= rd_cnt + 1'b1;
        end
    end

    // =========================================================================
    // 컬럼 j 를 뽑아 레지스터에 잡음 (32:1 mux x 32 를 파이프라인 밖으로)
    //
    // 패킹이면 rd_cnt 는 0..63 이고 상위 비트가 **어느 필드인지**를 고릅니다:
    //   rd_cnt[4:0] = PE 컬럼,  rd_cnt[5] = 0 -> 하위 곱(n = nt*64 + j)
    //                                     1 -> 상위 곱(n = nt*64 + 32 + j)
    //
    // 필드 분리는 shadow 를 자르는 것뿐입니다. **하위가 음수면 상위 자리에서
    // 1을 빌려갔으므로** 되돌려 줘야 합니다 — `+ sh[PK_S-1]` 이 그것입니다.
    //
    // attention(pack=0) 은 int8 하나를 두 니블로 쪼개 곱한 것이라 자리값으로
    // 다시 합칩니다 : int8 = 상위*16 + 하위(부호없음) → acc = hi*16 + lo.
    // =========================================================================
    wire [4:0]       rd_col   = rd_cnt[4:0];
    wire             rd_hi    = rd_cnt[5];
    wire [DIM_W-1:0] col_n_glb = pack
        ? ((nt_rd << 6) + {{(DIM_W-6){1'b0}}, rd_cnt})
        : ((nt_rd << 5) + {{(DIM_W-6){1'b0}}, rd_cnt});

    integer i;
    reg  [SH_W-1:0]      sh_i;
    reg  signed [PSUM_W-1:0] lo_i, hi_i;
    always @(posedge clk) begin
        if (rst) begin
            col_valid <= 1'b0;
            col_n <= {DIM_W{1'b0}}; col_mt <= {DIM_W{1'b0}};
            col_data <= {N*PSUM_W{1'b0}};
        end else begin
            col_valid <= rd_run && (col_n_glb < Nout);
            col_n     <= col_n_glb;
            col_mt    <= mt_rd;
            for (i = 0; i < N; i = i + 1) begin
                sh_i = acc[(i*N + rd_col)*SH_W +: SH_W];
                lo_i = $signed(sh_i[PK_S-1:0]);
                // [함정] `$signed(...) + sh_i[..]` 로 쓰면 피연산자 하나가
                // unsigned 라 **식 전체가 unsigned** 가 되어 부호확장이 사라집니다
                // (오차가 정확히 2^(SH_W-PK_S-1)). 빌림 보정도 signed 로 만듭니다.
                hi_i = $signed(sh_i[SH_W-1:PK_S]) + $signed({1'b0, sh_i[PK_S-1]});
                col_data[i*PSUM_W +: PSUM_W] <=
                    pack ? (rd_hi ? hi_i : lo_i)
                         : ((hi_i <<< 4) + lo_i);
            end
        end
    end
endmodule
