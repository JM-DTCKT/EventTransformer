// -----------------------------------------------------------------------------
// Axis_Loader : AXI-Stream(128b) → 온칩 메모리 순차 적재
//
// **명령어 프로그램(Inst_Mem)도 DMA 목적지**입니다 — 123 명령어 x 32바이트 =
// 3.9 KB 라 AXI-Lite 레지스터로 넣으면 쓰기가 976번입니다.
//
//   sel 0 W_Mem   256b → 2 beat/word
//   sel 1 A_Mem   512b → 4 beat/word      ← 32레인 x 16b (int8/bf16/Q4.11 공용)
//   sel 2 Requant_Mem 256b → 2 beat    ← 채널별 {mult, bias}   (구 PB_Mem)
//   sel 3 Affine_Mem  256b → 2 beat    ← 특징별 {gamma, beta}  (구 PG_Mem)
//   sel 4 Inst_Mem   256b → 2 beat    ← 명령어 프로그램       (구 Step_Mem)
//   sel 5 POS 표   512b → 4 beat      ← 21x21 행 x 64특징 (Pos_Gather 안)
//
// 스트림 바이트 순서가 곧 메모리 이미지입니다 — 재배치는 전부 `sw/pack_evt.py`
// 와 `sw/schedule_evt.py` 가 미리 해 둡니다. tready 는 항상 1 (목적지가 BRAM).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Axis_Loader #(
    parameter SW = 128,          // 스트림 폭
    parameter DW = 512,          // 가장 넓은 메모리 (A_Mem)
    parameter AW = 14            // 가장 큰 메모리의 워드 주소폭
)(
    input  wire            clk,
    input  wire            rst,

    input  wire            arm,           // 1클럭 펄스: sel/base 를 잡고 카운터 리셋
    input  wire [2:0]      arm_sel,
    input  wire [AW-1:0]   arm_base,

    input  wire            s_tvalid,
    output wire            s_tready,
    input  wire [SW-1:0]   s_tdata,
    input  wire            s_tlast,

    output reg             ld_we,
    output reg  [2:0]      ld_sel,
    output reg  [AW-1:0]   ld_addr,
    output reg  [DW-1:0]   ld_data,

    output reg  [AW-1:0]   words_written,
    output wire            busy
);
    // 목적지 인코딩 — `EvT_Engine.LD_*` / Top.v 의 LOAD_SEL 과 같습니다
    localparam [2:0] LD_W = 3'd0, LD_A = 3'd1, LD_RQ = 3'd2,
                     LD_AF = 3'd3, LD_INST = 3'd4, LD_POS = 3'd5;

    assign s_tready = 1'b1;

    reg [1:0]    beat_last;      // 워드의 마지막 비트 번호 (2beat→1, 4beat→3)
    reg [1:0]    beat;
    reg [AW-1:0] wr_ptr;
    reg [DW-1:0] shreg;
    reg          armed;

    assign busy = armed;

    always @(posedge clk) begin
        if (rst) begin
            ld_we <= 1'b0; ld_sel <= 3'd0; ld_addr <= {AW{1'b0}};
            ld_data <= {DW{1'b0}}; beat <= 2'd0; wr_ptr <= {AW{1'b0}};
            beat_last <= 2'd1; words_written <= {AW{1'b0}}; armed <= 1'b0;
        end else begin
            ld_we <= 1'b0;

            if (arm) begin
                ld_sel        <= arm_sel;
                wr_ptr          <= arm_base;
                beat         <= 2'd0;
                // A_Mem 과 POS 표만 512b — 나머지는 256b
                beat_last     <= (arm_sel == LD_A || arm_sel == LD_POS) ? 2'd3 : 2'd1;
                words_written <= {AW{1'b0}};
                armed         <= 1'b1;
            end else if (s_tvalid) begin
                // 하위 비트가 먼저 옵니다 — 오른쪽으로 밀어 넣습니다.
                shreg <= {s_tdata, shreg[DW-1:SW]};
                if (beat == beat_last) begin
                    // 2비트 모드면 방금 것이 상위 절반이라 위쪽 256b 에 있습니다.
                    ld_data <= (beat_last == 2'd1)
                             ? {{(DW/2){1'b0}}, s_tdata, shreg[DW-1:DW-SW]}
                             : {s_tdata, shreg[DW-1:SW]};
                    ld_addr <= wr_ptr;
                    ld_we   <= 1'b1;
                    wr_ptr    <= wr_ptr + 1'b1;
                    beat   <= 2'd0;
                    words_written <= words_written + 1'b1;
                end else begin
                    beat <= beat + 1'b1;
                end
            end

            if (s_tvalid && s_tlast) begin
                beat <= 2'd0;          // 워드 경계가 안 맞으면 버립니다
                armed <= 1'b0;
            end
        end
    end
endmodule
