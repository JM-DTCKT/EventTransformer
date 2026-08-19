// -----------------------------------------------------------------------------
// PE_Array_Pp : `PE_Array_Flat` 과 mesh 구조가 동일, **파면 두 개가 더 흐릅니다**
//
//   A     좌→우   amesh[i][j]
//   B     위→아래 bmesh[i][j]
//   clr   좌→우   cmesh[i][j]      ← A 와 같은 방향·같은 속도
//   snap  좌→우   smesh[i][j]      ← 마찬가지
//
// 둘 다 A 와 같은 속도로 흐르므로 PE[i][j] 는 **자기가 확정되는 사이클에** 값을
// shadow 로 복사하고, 그 다음 사이클에 P 를 비웁니다. 자세한 것은 `PE_OS_Pp.v`.
//
// `clr_edge` / `snap_edge` 는 `Gemm_Core_Ev` 의 32탭 시프트 레지스터가 만듭니다
// (A 의 `Skew_Buf` 와 같은 삼각형 지연을 1비트로 한 것).
// -----------------------------------------------------------------------------
module PE_Array_Pp #(
    parameter N      = 32,
    parameter ACT_W  = 8,
    parameter PSUM_W = 32
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    ce,
    input  wire [N-1:0]            clr_edge,   // 행 i 의 타일 시작   (i 사이클 지연)
    input  wire [N-1:0]            snap_edge,  // 행 i 의 확정 시점   (i 사이클 지연)
    input  wire [N*ACT_W-1:0]      a_edge,
    input  wire [N*ACT_W-1:0]      b_edge,
    output wire [N*N*PSUM_W-1:0]   acc_out
);
    wire signed [ACT_W-1:0] amesh [0:N-1][0:N];
    wire signed [ACT_W-1:0] bmesh [0:N][0:N-1];
    wire                    cmesh [0:N-1][0:N];
    wire                    smesh [0:N-1][0:N];

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : AEDGE
            assign amesh[i][0] = a_edge[i*ACT_W +: ACT_W];
            assign cmesh[i][0] = clr_edge[i];
            assign smesh[i][0] = snap_edge[i];
        end
        for (j = 0; j < N; j = j + 1) begin : BEDGE
            assign bmesh[0][j] = b_edge[j*ACT_W +: ACT_W];
        end
        for (i = 0; i < N; i = i + 1) begin : ROW
            for (j = 0; j < N; j = j + 1) begin : COL
                wire signed [PSUM_W-1:0] acc_ij;
                PE_OS_Pp #(.ACT_W(ACT_W), .PSUM_W(PSUM_W)) u_pe (
                    .clk(clk), .rst(rst), .ce(ce),
                    .clr_in (cmesh[i][j]), .clr_out (cmesh[i][j+1]),
                    .snap_in(smesh[i][j]), .snap_out(smesh[i][j+1]),
                    .a_in  (amesh[i][j]),   .b_in  (bmesh[i][j]),
                    .a_out (amesh[i][j+1]), .b_out (bmesh[i+1][j]),
                    .acc_out(acc_ij)
                );
                assign acc_out[(i*N + j)*PSUM_W +: PSUM_W] = acc_ij;
            end
        end
    endgenerate
endmodule
