// -----------------------------------------------------------------------------
// Axis_Dump : A_Mem 구간 → AXI-Stream(128b) → DMA S2MM → DDR
//
// **디버그용**입니다 — 최종 결과는 AXI-Lite 레지스터 하나(RES_CLASS)로 나갑니다.
// 포맷 경계마다 골든과 대조하려면 중간 활성값 자체가 필요하므로, A_Mem 의 임의
// 구간을 DDR 로 떠서 호스트에서 memcmp 합니다 (VCS 테스트벤치가 스냅샷으로 하던
// 대조를 보드에서 그대로 재현).
//
// A_Mem 워드가 512b, 스트림이 128b 라 **워드당 4 beat** 입니다. 바이트 순서는
// 레인 0 의 하위 바이트부터 — `sw/pack_evt.py` 가 만든 이미지와 같은 규칙입니다.
//
// BRAM 읽기 지연이 1사이클이라 워드마다 (주소 → 대기 → 4 beat) 6사이클입니다.
// 1,424워드 전체를 떠도 100 MHz 에서 85 us 라 계측에 영향이 없습니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Axis_Dump #(
    parameter SW = 128,
    parameter DW = 512,
    parameter AW = 11
)(
    input  wire            clk,
    input  wire            rst,

    input  wire            start,          // 1클럭 펄스
    input  wire [AW-1:0]   base,
    input  wire [AW:0]     len,            // 워드 수 (0 이면 아무것도 안 함)

    output wire            rd_en,
    output wire [AW-1:0]   rd_addr,
    input  wire [DW-1:0]   rd_data,

    output wire            m_tvalid,
    input  wire            m_tready,
    output wire [SW-1:0]   m_tdata,
    output wire [SW/8-1:0] m_tkeep,
    output wire            m_tlast,

    output wire            busy
);
    localparam integer BEATS = DW/SW;                  // 워드당 beat 수 (=4)

    localparam ST_IDLE=2'd0, ST_ADDR=2'd1, ST_WAIT=2'd2, ST_EMIT=2'd3;
    reg [1:0]    state;
    reg [AW-1:0] rd_ptr;
    reg [AW:0]   left;
    reg [DW-1:0] word_q;
    reg [1:0]    beat;

    // 주소는 **조합**으로 냅니다. 레지스터로 내면 ST_ADDR 다음 사이클에 주소가
    // 실리고 그 다음에야 데이터가 와서, ST_WAIT 에서 뜨는 rd_data 가 한 워드
    // 전 것이 됩니다 (Bram_Sdp 는 주소를 준 **다음** 사이클에 유효).
    assign rd_en    = (state == ST_ADDR);
    assign rd_addr  = rd_ptr;

    assign busy     = (state != ST_IDLE);
    assign m_tvalid = (state == ST_EMIT);
    assign m_tdata  = word_q[beat*SW +: SW];
    assign m_tkeep  = {(SW/8){1'b1}};
    // 마지막 워드의 마지막 비트에만 tlast — DMA S2MM 이 이걸로 전송을 끝냅니다.
    assign m_tlast  = (state == ST_EMIT) && (beat == BEATS-1) && (left == 1);

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            rd_ptr <= {AW{1'b0}}; left <= {(AW+1){1'b0}}; beat <= 2'd0;
        end else begin
            case (state)
                ST_IDLE: if (start && len != 0) begin
                            rd_ptr <= base; left <= len; state <= ST_ADDR;
                        end
                ST_ADDR: state <= ST_WAIT;               // 이 사이클에 주소가 나감
                ST_WAIT: begin                       // BRAM 출력 레지스터 1사이클
                            word_q <= rd_data; beat <= 2'd0; state <= ST_EMIT;
                        end
                ST_EMIT: if (m_tready) begin
                            if (beat == BEATS-1) begin
                                if (left == 1) state <= ST_IDLE;
                                else begin
                                    left <= left - 1'b1;
                                    rd_ptr <= rd_ptr + 1'b1;
                                    state   <= ST_ADDR;
                                end
                            end else beat <= beat + 1'b1;
                        end
            endcase
        end
    end
endmodule
