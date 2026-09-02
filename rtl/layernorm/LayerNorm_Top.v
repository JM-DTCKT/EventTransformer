// -----------------------------------------------------------------------------
// LayerNorm_Top : `LayerNorm_Unit` 래퍼 — A_Mem 읽기 + affine + 재양자화
//
// 코어는 **정규화만** 합니다 (bf16 또는 Q4.11 열 in → signed Q4.11 열 out).
// EvT 가 필요로 하는 나머지 셋을 여기서 붙입니다:
//
//   ① A_Mem 읽기   코어는 스트림만 받습니다. 행타일 M/32 개를 **끊지 않고**
//                  연달아 밀어 넣어야 코어의 3단 Tile 파이프라인이 겹칩니다.
//   ② affine       gamma/beta 를 `LayerNorm_Affine` 으로 적용합니다. 다음 Linear
//                  의 가중치에 접어 넣을 수도 있지만, 그러면 가중치 양자화
//                  격자가 바뀌어 골든과 어긋납니다.
//   ③ 재양자화     Q4.11 → int8 (스칼라 mult, shift)
//
// ## in_shift — 고정소수점 창
//
// 코어 내부는 Q8.15(±256)입니다. `layer_norm_1` 처럼 latent 가 20 타임스텝
// 누적된 자리는 값이 훨씬 커서 그대로 넣으면 포화합니다. `in_shift` 로 창을
// 옮깁니다 (value·2^xsh). LayerNorm 은 스케일 불변이라 결과가 안 바뀌고
// **정밀도만** 달라집니다. 값은 골든이 그 자리에서 쓰는 Qm.n 으로 정합니다:
//
//     골든 `.in` frac_bits = f  →  표현 범위 2^(15-f)  →  xsh = f - 8
//
// (여유 1비트. `sw/schedule_evt.py` 가 manifest 에서 계산해 SHIFT2 로 실어 옵니다.)
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module LayerNorm_Top #(
    parameter N      = 32,          // 레인 = 행
    parameter E      = 128,         // 정규화 축 (컴파일타임 — 코어 D)
    parameter DIM_W  = 16,
    parameter AW     = 14,
    parameter XSW    = 6
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 start,
    output reg                  done,
    input  wire [DIM_W-1:0]     M,          // 행 수 (32 의 배수가 아니어도 됨)
    input  wire [AW-1:0]        a_base,
    input  wire signed [XSW-1:0] in_shift,
    input  wire                  in_q411,    // 1 = rd_data 가 Q4.11 정수 코드, 0 = bf16

    // A_Mem 읽기 (bf16 x N)
    output wire                 rd_en,
    output wire [AW-1:0]        rd_addr,
    input  wire [N*16-1:0]      rd_data,

    // gamma/beta (특징 k 별, 전 레인 공유)
    output wire [DIM_W-1:0]     af_addr,
    input  wire signed [15:0]   af_gamma,    // Q1.14
    input  wire signed [15:0]   af_beta,     // Q4.11

    // Q4.11 → int8
    input  wire signed [31:0]   mult,
    input  wire [5:0]           shift,

    output wire                 out_valid,
    output wire [5:0]           out_mt,     // 행타일
    output wire [DIM_W-1:0]     out_k,      // 특징
    output wire [N*8-1:0]       out_data
);
    wire [5:0] mt_last = (M > 0) ? ((M - 1'b1) >> 5) : 6'd0;
    wire       rd_en_w;
    wire [AW-1:0] rd_addr_w;
    assign rd_en   = rd_en_w;
    assign rd_addr = rd_addr_w;

    // =========================================================================
    // 읽기 : 타일 mt 의 특징 k 를 순서대로. in_ready 가 내려가면 잠깐 멈춥니다.
    // (슬롯이 3개라 우리 M<=96 에서는 사실상 안 멈춥니다)
    // =========================================================================
    reg  [5:0]       rmt;
    reg  [DIM_W-1:0] rk;
    reg              rrun, rd_v;
    wire             core_iready;

    // **주소를 조합으로 내고 매 사이클 전진합니다** — `in_ready` 를 보고 멈추지
    // 않습니다. 이 설계에서는 멈출 일이 없기 때문입니다: 코어 슬롯이 3개이고
    // LN 한 번의 행타일이 최대 3개(M<=96)라 `in_ready` 가 안 내려갑니다.
    // [함정] 멈추게 만들면 주소가 레지스터라 데이터가 2사이클 뒤에 와서 수락
    // 판정과 어긋나 첫 열이 중복되고 이후가 한 칸씩 밀립니다. 혹시 내려가면
    // `stall_err` 로 드러나게 해 두었습니다.
    assign rd_en_w   = rrun;
    assign rd_addr_w = a_base + rmt * E[AW-1:0] + rk[AW-1:0];

    reg stall_err;
    reg [N*16-1:0] rd_data_q;
    reg            rd_v_q;
    // [함정] `rrun` 은 마지막 주소를 낸 순간 내려가는데 `done` 은 출력이 5단 뒤라
    // 아직 0 입니다. 그 사이에 `start` 가 계속 1 이면 같은 명령어가 **다시
    // 시작**합니다. 한 번 걸면 `start` 가 내려갈 때까지 다시 안 걸리게 래치합니다.
    reg armed;
    always @(posedge clk) begin
        if (rst) begin
            rmt <= 0; rk <= 0; rrun <= 1'b0; rd_v <= 1'b0; stall_err <= 1'b0;
            rd_v_q <= 1'b0;
            armed <= 1'b0;
        end else begin
            if (!start) armed <= 1'b0;
            rd_v <= rd_en_w;                     // A_Mem 1사이클 뒤 데이터
            // [타이밍] BRAM 출력 → 배선 → 코어의 포맷 변환기 → 첫 파이프
            // 레지스터가 한 사이클이었습니다 (7.2 ns). 여기서 끊으면 BRAM·배선과
            // 포맷 변환이 갈립니다. LN 명령어당 1사이클만 늘어납니다 (0.015 %).
            rd_v_q    <= rd_v;
            rd_data_q <= rd_data;
            if (rd_v_q && !core_iready) stall_err <= 1'b1;  // 있으면 안 되는 일

            if (start && !armed) begin
                rmt <= 0; rk <= 0; rrun <= 1'b1; armed <= 1'b1;
            end else if (rrun) begin
                if (rk == E - 1) begin
                    rk <= 0;
                    if (rmt == mt_last) rrun <= 1'b0;
                    else rmt <= rmt + 1'b1;
                end else rk <= rk + 1'b1;
            end
        end
    end

    // =========================================================================
    // 코어
    // =========================================================================
    wire            core_ov, core_olast;
    wire [N*16-1:0] core_ocol;

    // [타이밍] SRAM_LAT=2 : BRAM 의 **선택적 출력 레지스터**(DOx_REG) 를 켭니다.
    // 읽기 지연이 1 -> 2 사이클이 되는 대신 CLK->DOUT 이 0.826 ns 에서 절반 수준
    // 으로 떨어지고, 뒤따르는 변환기 18 단이 온전히 한 사이클을 씁니다.
    // (실측 WNS -0.102, xbuf -> g_norm[*].diff_r 계통)
    // 비용은 타일당 1 사이클 = 전체 39 사이클(0.06 %) 이고 FF 는 BRAM 안이라 0 입니다.
    LayerNorm_Unit #(.LANE(N), .D(E), .DLOG(7), .XSW(XSW), .SRAM_LAT(2), .NB(3))
    u_core (
        .clk(clk), .rst_n(~rst),
        .in_valid(rd_v_q), .in_ready(core_iready),
        .in_col(rd_data_q), .in_shift(in_shift), .in_q411(in_q411),
        .out_valid(core_ov), .out_col(core_ocol), .out_last(core_olast),
        .ovf());

    // =========================================================================
    // affine → 재양자화. 출력 순서가 곧 (타일, 특징) 이라 세면 됩니다.
    // =========================================================================
    // 명령어마다 0 부터 세야 합니다 — 안 그러면 두 번째 LN 부터 타일 번호가
    // 이어져 쓰기 주소도 `done` 판정도 어긋납니다.
    reg [5:0]       omt;
    reg [DIM_W-1:0] ok;
    always @(posedge clk) begin
        if (rst || !start) begin omt <= 0; ok <= 0; end
        else if (core_ov) begin
            if (ok == E - 1) begin ok <= 0; omt <= omt + 1'b1; end
            else ok <= ok + 1'b1;
        end
    end
    assign af_addr = ok;        // gamma/beta 는 특징 k 로 색인

    // Affine_Mem 은 주소를 준 **다음** 사이클에 답합니다. 코어 출력을 한 단 늦춰
    // gamma/beta 와 짝을 맞춥니다 (안 맞추면 한 특징씩 밀린 계수를 곱합니다).
    reg              core_ov_d;
    reg [N*16-1:0]   core_ocol_d;
    always @(posedge clk) begin
        core_ov_d   <= core_ov;
        core_ocol_d <= core_ocol;
    end

    wire [N-1:0] av, qv;
    wire [N*8-1:0] q_bus;
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : LANE
            wire signed [15:0] yaff;
            LayerNorm_Affine u_aff (
                .clk(clk), .rst(rst), .in_valid(core_ov_d),
                .xhat(core_ocol_d[g*16 +: 16]), .gamma(af_gamma), .beta(af_beta),
                .out_valid(av[g]), .y(yaff));

            wire signed [7:0] y8;
            Requant_Int #(.ACC_W(16), .MUL_W(32), .SH_W(6), .OUT_W(8),
                          .PIPE_PRE(1), .UNSIGNED_OUT(0), .USE_BIAS(0), .BIAS_W(32)) u_rq (
                .clk(clk), .rst(rst), .in_valid(av[g]),
                .acc(yaff), .bias(32'sd0), .mult(mult), .shift(shift),
                .out_valid(qv[g]), .out(y8));
            assign q_bus[g*8 +: 8] = y8;
        end
    endgenerate

    // 지연만큼 (타일, 특징) 을 늦춥니다 — 정렬 1 + LayerNorm_Affine 2 + Requant_Int 3
    localparam LAT = 6;   // 정렬 1 + LayerNorm_Affine 2 + Requant_Int **3**
    reg [5:0]       mq [0:LAT-1];
    reg [DIM_W-1:0] kq [0:LAT-1];
    integer z;
    always @(posedge clk) begin
        mq[0] <= omt;  kq[0] <= ok;
        for (z = 1; z < LAT; z = z + 1) begin
            mq[z] <= mq[z-1];  kq[z] <= kq[z-1];
        end
    end
    assign out_valid = qv[0];
    assign out_mt    = mq[LAT-1];
    assign out_k     = kq[LAT-1];
    assign out_data  = q_bus;

    // 마지막 타일의 마지막 특징까지 나오면 완료
    always @(posedge clk) begin
        if (rst) done <= 1'b0;
        else if (out_valid && out_mt == mt_last && out_k == E - 1) done <= 1'b1;
        else if (!start) done <= 1'b0;
    end
endmodule
