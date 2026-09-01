// -----------------------------------------------------------------------------
// Pos_Gather : positional encoding 을 **온칩 표**에서 모아 PIN 뒤쪽에 씁니다
//
// pos enc 는 `pos_idx` 만으로 정해지고 표는 고정입니다. 호스트가 워드 레이아웃
// 으로 펴서 보내면 96.7 MB 인데, 표(27.6 KB) 를 PL 에 두고 타임스텝마다
// `pos_idx`(최대 246 B) 만 받으면 됩니다.
//
// ## 레인마다 다른 주소가 필요한데도 싼 이유
//
// A_Mem 워드는 32레인이 서로 다른 토큰이라, 워드 하나를 만들 때마다 표를 32번
// 읽으면 타일당 64x32 = 2,048 사이클입니다. 대신 표를 **행 단위로 넓게**
// (한 행 = 64특징 = 512비트) 두면 **한 토큰의 64개 값이 한 번의 읽기**로
// 나옵니다. 그 뒤는 `Transpose32` 와 같은 축 변환 문제입니다.
//
//     타일 하나(토큰 32개)
//       ① 표에서 32행 읽기       96 사이클   (토큰당 3위상 — BRAM 지연)
//       ② 32x64 버퍼에 쌓기
//       ③ A_Mem 에 64워드 쓰기   64 사이클   (레인 = 토큰)
//                              ~160 사이클/타일
//
// 타임스텝당 최대 4타일 = 650 사이클 남짓 — 타임스텝 하나가 53만 사이클이니
// **0.1 %** 입니다.
//
// ## 쓰는 곳
//
//   PIN[a_base + row_tile*ostr + 96 + d]  레인 i = 표[pos_idx[row_tile*32+i]][d]
//
// 앞 96워드는 `event_projection` 이 채웁니다 — 그래서 stride 가 160 입니다.
// 유효 토큰(n_tok)을 넘는 레인은 0 으로 둡니다.
//
// `pos_idx` 는 타임스텝당 최대 123개(246 B)라 따로 포트를 두지 않고 A_Mem 의
// 작은 영역에서 읽습니다 (워드 row_tile 의 레인 i = 토큰 row_tile*32+i 의 인덱스).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

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
    reg [FEAT*8-1:0] pos_tbl [0:(1<<AW_T)-1];
    reg [FEAT*8-1:0] pos_tbl_q;
    reg [AW_T-1:0]   pos_tbl_addr;
    always @(posedge clk) begin
        if (ld_we) pos_tbl[ld_addr] <= ld_data;
        pos_tbl_q <= pos_tbl[pos_tbl_addr];        // 주소를 잡은 **다음** 사이클에 나옵니다
    end

    // 32행 x 64바이트 전치 버퍼. 열 d 를 읽으면 32레인이 나옵니다.
    reg [FEAT*8-1:0] row_buf [0:N-1];

    localparam ST_IDLE=3'd0, ST_IDX=3'd1, ST_ROW=3'd2, ST_OUT=3'd3, ST_DONE=3'd4;
    reg [2:0]        state;
    reg [5:0]        row_tile;           // 행타일
    reg [5:0]        lane;           // 타일 안 레인
    reg [6:0]        feat;           // 특징
    reg [1:0]        ph;
    reg [N*16-1:0]   idx_word;        // 이 타일의 pos_idx 32개
    wire [5:0]       row_tile_last = (n_tok > 0) ? ((n_tok - 1'b1) >> 5) : 6'd0;

    // 열 d 를 32레인으로 (유효 토큰 밖은 0)
    integer c;
    reg [N*16-1:0] col_word;
    always @* begin
        col_word = {(N*16){1'b0}};
        for (c = 0; c < N; c = c + 1)
            if ({row_tile, 5'd0} + c[5:0] < n_tok)
                col_word[c*16 +: 16] = {{8{row_buf[c][feat[5:0]*8+7]}},
                                     row_buf[c][feat[5:0]*8 +: 8]};
    end

    integer z;
    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE; done <= 1'b0; row_tile <= 0; lane <= 0; feat <= 0; ph <= 0;
            rd_en <= 1'b0; we_en <= 1'b0;
        end else begin
            rd_en <= 1'b0; we_en <= 1'b0;
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin row_tile <= 0; ph <= 0; state <= ST_IDX; end
                end
                // 이 타일의 pos_idx 워드 하나를 읽습니다. `rd_en`/`rd_addr` 이
                // 레지스터 출력이라 BRAM 이 주소를 보는 것이 한 사이클 뒤입니다
                // → 3위상 (발행 → 대기 → 수신).
                ST_IDX: begin
                    if (ph == 2'd0) begin
                        rd_en   <= 1'b1;
                        rd_addr <= idx_base + {{(AW_A-6){1'b0}}, row_tile};
                    end
                    ph <= ph + 1'b1;
                    if (ph == 2'd2) begin
                        idx_word <= rd_data;
                        lane <= 0; ph <= 0; state <= ST_ROW;
                    end
                end
                // 토큰마다 표에서 한 행씩 (같은 이유로 3위상)
                ST_ROW: begin
                    if (ph == 2'd0) pos_tbl_addr <= idx_word[lane*16 +: AW_T];
                    ph <= ph + 1'b1;
                    if (ph == 2'd2) begin
                        row_buf[lane[4:0]] <= pos_tbl_q;
                        ph <= 0;
                        if (lane == N - 1) begin feat <= 0; state <= ST_OUT; end
                        else lane <= lane + 1'b1;
                    end
                end
                // 특징마다 워드 하나 (레인 = 토큰)
                ST_OUT: begin
                    we_en   <= 1'b1;
                    // [함정] `feat` 는 7비트입니다. `feat[AW_A-1:0]` 처럼 폭을
                    // 넘겨 잘라 쓰면 Verilog 가 **조용히 X** 를 줍니다.
                    we_addr <= a_base + row_tile * ostr[AW_A-1:0]
                             + (ostr[AW_A-1:0] - FEAT)
                             + {{(AW_A-7){1'b0}}, feat};
                    we_data <= col_word;
                    if (feat == FEAT - 1) begin
                        if (row_tile == row_tile_last) state <= ST_DONE;
                        else begin row_tile <= row_tile + 1'b1; ph <= 0; state <= ST_IDX; end
                    end else feat <= feat + 1'b1;
                end
                ST_DONE: begin
                    done <= 1'b1;
                    if (!start) state <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
