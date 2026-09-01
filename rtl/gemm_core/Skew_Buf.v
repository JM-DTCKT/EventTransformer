// -----------------------------------------------------------------------------
// Skew_Buf : systolic 정렬용 삼각형 지연 버퍼
//
// 순진하게 짜면 레인마다 **다른 주소**를 읽어야 합니다.
//
//     a_edge[i] = a_mem[mt*N+i][c-i]        ← 레인 i 는 c-i 번째 k
//     b_edge[j] = b_mem[c-j][nt*N+j]        ← 레인 j 는 c-j 번째 k
//
// 그러면 뱅크 32개가 각각 독립 주소를 받아야 해서 메모리를 32조각으로 쪼개야
// 하고, AXI 로 채울 때도 포트가 32개 필요합니다.
//
// 대신 **skew 를 메모리 밖으로 빼냅니다.** 메모리는 폭 N*W 짜리 한 덩어리로 두고
// 주소 c 하나만 읽은 뒤(= k=c 의 32개 레인값이 한 워드), 레인 i 가
// **i 사이클 전 워드의 i 번째 레인값**을 집게 하면 원래 식과 정확히 같아집니다.
//
//     dout[i] = din_at_cycle(c-i)[i]
//
// stage 0 은 조합(현재 워드), stage 1..N-1 은 레지스터입니다. 따라서 din 을 넣은
// **같은 사이클에** dout[0] 이, 다음 사이클에 dout[1] 이 ... 나옵니다.
//
// 자원: stage d 는 레인 d..N-1 만 실제로 쓰이므로(하위 레인은 fanout 이 없음)
// 합성기가 정리하면 N(N+1)/2 = 528 바이트 ≈ 4.2K FF 로 줄어듭니다. 코드는 정사각형
// 으로 두고 트리밍에 맡깁니다 — 삼각형을 손으로 쓰면 폭이 제각각이라 읽기 어렵습니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Skew_Buf #(
    parameter N = 32,               // 레인 수
    parameter W = 8                 // 레인당 비트폭
)(
    input  wire            clk,
    input  wire            rst,
    input  wire            clr,          // 타일 시작: 파이프 비우기
    input  wire [N*W-1:0]  din,          // 현재 사이클의 워드 (레인 i = din[i*W +: W])
    input  wire            din_valid,    // 이 워드가 유효한 k 인가
    output wire [N*W-1:0]  dout,         // dout[i] = i 사이클 전 워드의 레인 i
    output wire [N-1:0]    dout_valid
);
    // stage 1..N-1 만 레지스터. stage 0 은 din 그 자체(지연 0).
    reg [N*W-1:0] s [1:N-1];
    reg [N-1:1]   v;

    integer d;
    always @(posedge clk) begin
        if (rst || clr) begin
            for (d = 1; d < N; d = d + 1) s[d] <= {N*W{1'b0}};
            v <= {(N-1){1'b0}};
        end else begin
            s[1] <= din;
            v[1] <= din_valid;
            for (d = 2; d < N; d = d + 1) begin
                s[d] <= s[d-1];
                v[d] <= v[d-1];
            end
        end
    end

    genvar i;
    generate
        // 레인 0 : 지연 없음
        assign dout[0 +: W]  = din_valid ? din[0 +: W] : {W{1'b0}};
        assign dout_valid[0] = din_valid;

        for (i = 1; i < N; i = i + 1) begin : LANE
            assign dout[i*W +: W] = v[i] ? s[i][i*W +: W] : {W{1'b0}};
            assign dout_valid[i]  = v[i];
        end
    endgenerate
endmodule
