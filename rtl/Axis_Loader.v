// -----------------------------------------------------------------------------
// Axis_Loader : AXI-Stream(128b) → 온칩 메모리(256b) 순차 적재
//
// AXI DMA 의 MM2S 가 DDR 에서 읽어 흘려보낸 바이트를 그대로 메모리에 쌓습니다.
// 스트림 128b, 메모리 256b 라 **2비트씩 모아 한 워드**를 씁니다.
//
//   워드 w 의 바이트 b  =  스트림의 (32w + b) 번째 바이트
//
// 즉 소프트웨어가 넘긴 버퍼가 곧 메모리 이미지입니다 — 순서 재배치는 전부
// 호스트 쪽 `pack.py` 가 미리 해 둡니다. RTL 에 주소 계산 로직이 없다는 뜻이라
// 디버깅할 게 없습니다.
//
// tready 는 항상 1 입니다. 목적지가 BRAM 이고 2비트당 1워드라 백프레셔가 생길
// 이유가 없습니다 (DMA 쪽이 병목).
//
// 전송 시작 전에 AXI-Lite 로 `arm` 을 쳐서 목적지(sel)와 시작 워드(base)를 잡습니다.
// -----------------------------------------------------------------------------
module Axis_Loader #(
    parameter SW = 128,          // 스트림 폭
    parameter DW = 256,          // 메모리 폭
    parameter AW = 12            // 메모리 워드 주소폭 (가장 큰 메모리 기준)
)(
    input  wire            clk,
    input  wire            rst,

    // ---- 제어 (AXI-Lite) ----
    input  wire            arm,          // 1클럭 펄스: sel/base 를 잡고 카운터 리셋
    input  wire [1:0]      arm_sel,
    input  wire [AW-1:0]   arm_base,

    // ---- AXI-Stream slave ----
    input  wire            s_tvalid,
    output wire            s_tready,
    input  wire [SW-1:0]   s_tdata,
    input  wire            s_tlast,

    // ---- 메모리 write ----
    output reg             ld_we,
    output reg  [1:0]      ld_sel,
    output reg  [AW-1:0]   ld_addr,
    output reg  [DW-1:0]   ld_data,

    // ---- 상태 ----
    output reg  [AW-1:0]   words_written
);
    localparam integer BEATS = DW/SW;        // 워드당 비트 수 (=2)

    assign s_tready = 1'b1;

    reg [SW-1:0] hold;
    reg          phase;
    reg [AW-1:0] wptr;

    always @(posedge clk) begin
        if (rst) begin
            ld_we <= 1'b0; ld_sel <= 2'd0; ld_addr <= {AW{1'b0}};
            ld_data <= {DW{1'b0}}; phase <= 1'b0; wptr <= {AW{1'b0}};
            words_written <= {AW{1'b0}};
        end else begin
            ld_we <= 1'b0;

            if (arm) begin
                ld_sel <= arm_sel;
                wptr   <= arm_base;
                phase  <= 1'b0;
                words_written <= {AW{1'b0}};
            end else if (s_tvalid) begin
                if (phase == 1'b0) begin
                    hold  <= s_tdata;
                    phase <= 1'b1;
                end else begin
                    ld_data <= {s_tdata, hold};      // 하위 비트가 먼저 온 쪽
                    ld_addr <= wptr;
                    ld_we   <= 1'b1;
                    wptr    <= wptr + 1'b1;
                    phase   <= 1'b0;
                    words_written <= words_written + 1'b1;
                end
            end

            if (s_tvalid && s_tlast) phase <= 1'b0;   // 홀수 비트로 끝나면 버림
        end
    end
endmodule
