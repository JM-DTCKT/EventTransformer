// -----------------------------------------------------------------------------
// Pos_Gather : positional encoding 을 **온칩 표**에서 모아 PIN 뒤쪽에 씁니다
//
// 전에는 호스트가 타임스텝마다 pos enc 를 워드 레이아웃으로 펴서 DDR 에 올렸고
// 그 이미지가 **96.7 MB** 였습니다. 정보량은 표 27.6 KB + `pos_idx`(토큰당 2B,
// 전체 0.47 MB) 뿐이라 200배를 펼쳐 보낸 셈입니다. 이 모듈이 그걸 없앱니다.
//
// ## 왜 쌌나 (처음엔 비싸다고 봤습니다)
//
// A_Mem 워드는 32레인이 서로 다른 토큰이라 "레인마다 다른 주소"가 필요합니다.
// 워드 하나를 만들 때마다 32번 읽으면 타일당 64x32 = 2,048 사이클이라 비쌉니다.
//
// 그런데 표를 **행 단위로 넓게**(한 행 = 64특징 = 512비트) 두면 **한 토큰의 64개
// 값이 한 번의 읽기**로 나옵니다. 그 뒤는 축을 돌리는 문제이고, 그건 이미
// `Transpose32`(V 전용 corner-turn)로 풀어 둔 것과 같습니다.
//
//     타일 하나(토큰 32개)
//       ① 표에서 32행 읽기       96 사이클   (토큰당 3위상 — BRAM 지연)
//       ② 32x64 버퍼에 쌓기
//       ③ A_Mem 에 64워드 쓰기   64 사이클   (레인 = 토큰)
//                              ~160 사이클/타일
//
// 타임스텝당 최대 4타일 = 650 사이클 남짓. 타임스텝 하나가 53만 사이클이니
// **0.1 %** 입니다.
//
// ## 쓰는 곳
//
//   PIN[a_base + mt*ostr + 96 + d]  레인 i = TBL[pos_idx[mt*32+i]][d]
//
// 앞 96워드는 `event_projection` 이 채웁니다 — 그래서 stride 가 160 입니다.
// 유효 토큰(n_tok)을 넘는 레인은 0 으로 둡니다.
//
// ## pos_idx 는 A_Mem 에서
//
// 타임스텝당 최대 123개(246 B)라 따로 포트를 두지 않고 A_Mem 의 작은 영역을
// 씁니다. 워드 mt 의 레인 i = 토큰 mt*32+i 의 인덱스 (int16).
// -----------------------------------------------------------------------------
module Pos_Gather #(
    parameter N      = 32,        // 레인 = 토큰
    parameter FEAT   = 64,        // pos enc 특징 수
    parameter AW_A   = 14,        // A_Mem 주소폭
    parameter AW_T   = 9,         // 표 주소폭 (21*21 = 441)
    parameter DIM_W  = 16
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 start,
    output reg                  done,

    input  wire [DIM_W-1:0]     n_tok,      // 유효 토큰 수
    input  wire [AW_A-1:0]      a_base,     // PIN 베이스
    input  wire [DIM_W-1:0]     ostr,       // PIN stride (160)
    input  wire [AW_A-1:0]      idx_base,   // pos_idx 영역 베이스

    // ---- 표 적재 (로더 sel 5) ----
    input  wire                 ld_we,
    input  wire [AW_T-1:0]      ld_addr,
    input  wire [FEAT*8-1:0]    ld_data,

    // ---- A_Mem 읽기 (pos_idx) ----
    output reg                  rd_en,
    output reg  [AW_A-1:0]      rd_addr,
    input  wire [N*16-1:0]      rd_data,

    // ---- A_Mem 쓰기 ----
    output reg                  we_en,
    output reg  [AW_A-1:0]      we_addr,
    output reg  [N*16-1:0]      we_data
);
    // =========================================================================
    // 표 : 441행 x 64바이트. 행 하나가 한 토큰의 pos enc 전부입니다.
    // =========================================================================
    reg [FEAT*8-1:0] tbl [0:(1<<AW_T)-1];
    reg [FEAT*8-1:0] tbl_q;
    reg [AW_T-1:0]   tbl_a;
    always @(posedge clk) begin
        if (ld_we) tbl[ld_addr] <= ld_data;
        tbl_q <= tbl[tbl_a];        // 주소를 잡은 **다음** 사이클에 나옵니다
    end

    // 32행 x 64바이트 전치 버퍼. 열 d 를 읽으면 32레인이 나옵니다.
    reg [FEAT*8-1:0] buf_r [0:N-1];

    localparam S_IDLE=3'd0, S_IDX=3'd1, S_ROW=3'd2, S_OUT=3'd3, S_DONE=3'd4;
    reg [2:0]        st;
    reg [5:0]        mt;           // 행타일
    reg [5:0]        li;           // 타일 안 레인
    reg [6:0]        di;           // 특징
    reg [1:0]        ph;
    reg [N*16-1:0]   idx_q;        // 이 타일의 pos_idx 32개
    wire [5:0]       mt_last = (n_tok > 0) ? ((n_tok - 1'b1) >> 5) : 6'd0;

    // 열 d 를 32레인으로 (유효 토큰 밖은 0)
    integer c;
    reg [N*16-1:0] col_w;
    always @* begin
        col_w = {(N*16){1'b0}};
        for (c = 0; c < N; c = c + 1)
            if ({mt, 5'd0} + c[5:0] < n_tok)
                col_w[c*16 +: 16] = {{8{buf_r[c][di[5:0]*8+7]}},
                                     buf_r[c][di[5:0]*8 +: 8]};
    end

    integer z;
    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; done <= 1'b0; mt <= 0; li <= 0; di <= 0; ph <= 0;
            rd_en <= 1'b0; we_en <= 1'b0;
        end else begin
            rd_en <= 1'b0; we_en <= 1'b0;
            case (st)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin mt <= 0; ph <= 0; st <= S_IDX; end
                end
                // 이 타일의 pos_idx 워드 하나를 읽습니다.
                // `rd_en`/`rd_addr` 이 **레지스터 출력**이라 BRAM 이 주소를 보는
                // 것이 한 사이클 뒤입니다. 그래서 3위상 (발행 → 대기 → 수신)
                // 입니다 — 두 위상으로 두면 직전 주소의 데이터를 잡습니다.
                S_IDX: begin
                    if (ph == 2'd0) begin
                        rd_en   <= 1'b1;
                        rd_addr <= idx_base + {{(AW_A-6){1'b0}}, mt};
                    end
                    ph <= ph + 1'b1;
                    if (ph == 2'd2) begin
                        idx_q <= rd_data;
                        li <= 0; ph <= 0; st <= S_ROW;
                    end
                end
                // 토큰마다 표에서 한 행씩 (같은 이유로 3위상)
                S_ROW: begin
                    if (ph == 2'd0) tbl_a <= idx_q[li*16 +: AW_T];
                    ph <= ph + 1'b1;
                    if (ph == 2'd2) begin
                        buf_r[li[4:0]] <= tbl_q;
                        ph <= 0;
                        if (li == N - 1) begin di <= 0; st <= S_OUT; end
                        else li <= li + 1'b1;
                    end
                end
                // 특징마다 워드 하나 (레인 = 토큰)
                S_OUT: begin
                    we_en   <= 1'b1;
                    // `di` 는 7비트입니다 — `di[AW_A-1:0]` 처럼 범위를 넘겨
                    // 잘라 쓰면 Verilog 는 **조용히 X** 를 줍니다 (주소 전체가 X
                    // 가 돼 PIN 뒤쪽이 통째로 안 써졌습니다). 이 프로젝트에서
                    // 세 번째로 만난 같은 함정입니다.
                    we_addr <= a_base + mt * ostr[AW_A-1:0]
                             + (ostr[AW_A-1:0] - FEAT)
                             + {{(AW_A-7){1'b0}}, di};
                    we_data <= col_w;
                    if (di == FEAT - 1) begin
                        if (mt == mt_last) st <= S_DONE;
                        else begin mt <= mt + 1'b1; ph <= 0; st <= S_IDX; end
                    end else di <= di + 1'b1;
                end
                S_DONE: begin
                    done <= 1'b1;
                    if (!start) st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
