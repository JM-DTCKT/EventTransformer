// -----------------------------------------------------------------------------
// Bram_Sdp : simple dual-port BRAM (write 1 포트 + read 1 포트), byte write enable
//
// Vivado 가 block RAM 으로 추론하는 표준 형태입니다. 같은 주소를 동시에 read/write
// 하는 경우가 이 설계엔 없으므로(로드 단계와 연산 단계가 분리) collision 정책은
// 무관합니다.
//
// 읽기 지연 1 사이클 (rd_en 으로 주소를 준 다음 사이클에 rd_data 유효).
// -----------------------------------------------------------------------------
module Bram_Sdp #(
    parameter DW = 256,             // 데이터 폭 (8의 배수)
    parameter AW = 10               // 주소 폭 → 깊이 2^AW
)(
    input  wire                clk,

    input  wire                we_en,
    input  wire [AW-1:0]       we_addr,
    input  wire [DW/8-1:0]     we_be,      // 바이트별 write enable
    input  wire [DW-1:0]       we_data,

    input  wire                rd_en,
    input  wire [AW-1:0]       rd_addr,
    output reg  [DW-1:0]       rd_data
);
    localparam integer NB = DW/8;

    reg [DW-1:0] mem [0:(1<<AW)-1];

    integer b;
    always @(posedge clk) begin
        if (we_en)
            for (b = 0; b < NB; b = b + 1)
                if (we_be[b]) mem[we_addr][b*8 +: 8] <= we_data[b*8 +: 8];
        if (rd_en)
            rd_data <= mem[rd_addr];
    end
endmodule
