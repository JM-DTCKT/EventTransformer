// -----------------------------------------------------------------------------
// Tile_Ctrl : Mac_OS/Tiling_Controller 의 보드용 판
//
// 원본과 다른 점은 딱 둘입니다.
//
//  1) WRITE 가 1사이클 병렬 복사가 아니라 **핸드셰이크**입니다.
//     원본 Output_Collector 는 32x32 타일을 한 사이클에 y_out 배열로 복사했는데,
//     그건 레지스터 32K 개짜리 배열이라 합성이 안 됩니다. 보드에서는 컬럼 하나씩
//     32 사이클에 걸쳐 빼내며 곧바로 requantize 하므로, 그동안 컨트롤러가 기다려야
//     합니다. → RD 상태에서 tile_rdy 를 올리고 rd_done 을 기다립니다.
//
//  2) `readout` 출력이 없습니다 (위 핸드셰이크가 대신함).
//
// 타일 순회와 stream_len 은 원본 그대로입니다:
//     stream_len = K + 2N + 4      (skew fill/drain 2N, DSP48E2 MREG+PREG flush 4)
//
// BRAM 지연 1사이클은 Gemm_Core 가 clr/ce 를 통째로 1클럭 늦춰 흡수합니다. 상대
// 타이밍이 그대로라 stream_len 은 손댈 필요가 없습니다.
// -----------------------------------------------------------------------------
module Tile_Ctrl #(
    parameter N     = 32,
    parameter DIM_W = 16
)(
    input  wire             clk,
    input  wire             rst,
    input  wire             start,
    input  wire [DIM_W-1:0] M,
    input  wire [DIM_W-1:0] in_features,     // K
    input  wire [DIM_W-1:0] out_features,    // Nout

    output reg  [DIM_W-1:0] mt,              // 현재 행 타일
    output reg  [DIM_W-1:0] nt,              // 현재 열 타일
    output wire [DIM_W-1:0] c,               // 스트리밍 사이클
    output wire             clr,             // 타일 시작 (PE 누산기 클리어)
    output wire             ce,              // 스트리밍 (누적 진행)
    output wire             k_valid,         // 이 사이클의 c 가 유효한 k 인가
    output wire             tile_rdy,        // 타일 완료 → 읽어가라
    input  wire             rd_done,         // 읽기 끝났다
    output reg              all_done
);
    localparam IDLE=0, CLR=1, STREAM=2, RD=3, NEXT=4, DONE=5;
    reg [2:0] state;
    reg [DIM_W-1:0] cyc;

    wire [DIM_W-1:0] num_mt = (M            + N - 1) / N;
    wire [DIM_W-1:0] num_nt = (out_features + N - 1) / N;
    wire [DIM_W-1:0] stream_len = in_features + 2*N + 4;

    assign clr      = (state == CLR);
    assign ce       = (state == STREAM);
    assign k_valid  = (state == STREAM) && (cyc < in_features);
    assign tile_rdy = (state == RD);
    assign c        = cyc;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; mt <= 0; nt <= 0; cyc <= 0; all_done <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    all_done <= 1'b0;
                    if (start) begin mt <= 0; nt <= 0; state <= CLR; end
                end
                CLR: begin
                    cyc <= 0; state <= STREAM;
                end
                STREAM: begin
                    cyc <= cyc + 1'b1;
                    if (cyc == stream_len - 1) state <= RD;
                end
                RD: begin                        // 컬럼 32개를 빼내는 동안 대기
                    if (rd_done) state <= NEXT;
                end
                NEXT: begin
                    if (nt == num_nt - 1) begin
                        nt <= 0;
                        if (mt == num_mt - 1) state <= DONE;
                        else begin mt <= mt + 1'b1; state <= CLR; end
                    end else begin
                        nt <= nt + 1'b1; state <= CLR;
                    end
                end
                DONE: begin
                    all_done <= 1'b1;
                    if (!start) state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
