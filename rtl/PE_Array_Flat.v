// -----------------------------------------------------------------------------
// PE_Array_Flat : Mac_OS/PE_Array 와 **논리적으로 동일**, 포트만 packed 버스
//
// 원본 PE_Array 는 unpacked array 포트(`a_edge [0:N-1]`)를 씁니다. SystemVerilog
// 기능이라 시뮬은 되지만 합성 툴/버전 간 이식성이 떨어져, 보드용으로는 같은 mesh 를
// packed 버스로 다시 씁니다. **PE_OS 는 원본 그대로 인스턴스**합니다 — 검증된
// DSP48E2 매핑을 건드리지 않는 게 요점입니다.
//
//   A 는 좌→우, B 는 위→아래로 흐르고 각 PE 가 자기 출력 C[i][j] 를 로컬 누적.
//   acc_out[(i*N+j)*PSUM_W +: PSUM_W] = C[i][j]
// -----------------------------------------------------------------------------
module PE_Array_Flat #(
    parameter N      = 32,
    parameter ACT_W  = 8,
    parameter PSUM_W = 32
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    clr,
    input  wire                    ce,
    input  wire [N*ACT_W-1:0]      a_edge,     // 왼쪽 가장자리 (행 i)
    input  wire [N*ACT_W-1:0]      b_edge,     // 위쪽 가장자리 (열 j)
    output wire [N*N*PSUM_W-1:0]   acc_out
);
    // amesh[i][j] : 행 i 에서 열 j 로 들어가는 A,  j = 0..N (N 은 버려짐)
    // bmesh[i][j] : 열 j 에서 행 i 로 들어가는 B,  i = 0..N
    wire signed [ACT_W-1:0] amesh [0:N-1][0:N];
    wire signed [ACT_W-1:0] bmesh [0:N][0:N-1];

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : AEDGE
            assign amesh[i][0] = a_edge[i*ACT_W +: ACT_W];
        end
        for (j = 0; j < N; j = j + 1) begin : BEDGE
            assign bmesh[0][j] = b_edge[j*ACT_W +: ACT_W];
        end
        for (i = 0; i < N; i = i + 1) begin : ROW
            for (j = 0; j < N; j = j + 1) begin : COL
                wire signed [PSUM_W-1:0] acc_ij;
                PE_OS #(.ACT_W(ACT_W), .PSUM_W(PSUM_W)) u_pe (
                    .clk(clk), .rst(rst), .clr(clr), .ce(ce),
                    .a_in  (amesh[i][j]),   .b_in  (bmesh[i][j]),
                    .a_out (amesh[i][j+1]), .b_out (bmesh[i+1][j]),
                    .acc_out(acc_ij)
                );
                assign acc_out[(i*N + j)*PSUM_W +: PSUM_W] = acc_ij;
            end
        end
    endgenerate
endmodule
