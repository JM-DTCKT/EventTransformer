// -----------------------------------------------------------------------------
// Axis_Dump : A_Mem 구간 → AXI-Stream(128b) → DMA S2MM → DDR
//
// `fpga/` 의 MLP 는 결과가 이미지당 클래스 인덱스 하나뿐이라 `Axis_Sender` 로
// 128바이트만 내보내면 끝이었습니다. 여기는 **포맷 경계마다 대조**해야 하므로
// 중간 활성값 자체를 읽어와야 합니다:
//
//   step 0 → A_RES 128워드  = stem_bf16     step 1 → A_LN  128워드 = b0_norm_int8
//   step 2 → A_GE  256워드  = b0_gelu_int8  step 4 → A_RES 128워드 = b0_res_bf16
//   …
//
// 엔진의 `step_lo/step_hi` 로 한 step 만 돌리고 여기서 떠 보면, VCS 테스트벤치가
// 스냅샷으로 하던 대조를 보드에서 그대로 재현할 수 있습니다.
//
// A_Mem 워드가 512b, 스트림이 128b 라 **워드당 4비트**입니다. 바이트 순서는
// 레인 0 의 하위 바이트부터 — `pack_nl.py` 가 만든 이미지와 같은 규칙이라
// 호스트에서 memcmp 한 번이면 됩니다.
//
// BRAM 읽기 지연이 1사이클이라 워드마다 (주소 → 대기 → 4비트) 6사이클입니다.
// 1,424워드 전체를 떠도 100MHz 에서 85us 라 계측에 영향이 없습니다.
// -----------------------------------------------------------------------------
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
    localparam integer NB = DW/SW;                  // 워드당 비트 수 (=4)

    localparam S_IDLE=2'd0, S_ADDR=2'd1, S_WAIT=2'd2, S_EMIT=2'd3;
    reg [1:0]    st;
    reg [AW-1:0] wptr;
    reg [AW:0]   left;
    reg [DW-1:0] hold;
    reg [1:0]    beat;

    // 주소는 **조합**으로 냅니다. 레지스터로 내면 S_ADDR 다음 사이클에 주소가
    // 실리고 그 다음에야 데이터가 와서, S_WAIT 에서 뜨는 rd_data 가 한 워드
    // 전 것이 됩니다 (Bram_Sdp 는 주소를 준 **다음** 사이클에 유효).
    assign rd_en    = (st == S_ADDR);
    assign rd_addr  = wptr;

    assign busy     = (st != S_IDLE);
    assign m_tvalid = (st == S_EMIT);
    assign m_tdata  = hold[beat*SW +: SW];
    assign m_tkeep  = {(SW/8){1'b1}};
    // 마지막 워드의 마지막 비트에만 tlast — DMA S2MM 이 이걸로 전송을 끝냅니다.
    assign m_tlast  = (st == S_EMIT) && (beat == NB-1) && (left == 1);

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE;
            wptr <= {AW{1'b0}}; left <= {(AW+1){1'b0}}; beat <= 2'd0;
        end else begin
            case (st)
                S_IDLE: if (start && len != 0) begin
                            wptr <= base; left <= len; st <= S_ADDR;
                        end
                S_ADDR: st <= S_WAIT;               // 이 사이클에 주소가 나감
                S_WAIT: begin                       // BRAM 출력 레지스터 1사이클
                            hold <= rd_data; beat <= 2'd0; st <= S_EMIT;
                        end
                S_EMIT: if (m_tready) begin
                            if (beat == NB-1) begin
                                if (left == 1) st <= S_IDLE;
                                else begin
                                    left <= left - 1'b1;
                                    wptr <= wptr + 1'b1;
                                    st   <= S_ADDR;
                                end
                            end else beat <= beat + 1'b1;
                        end
            endcase
        end
    end
endmodule
