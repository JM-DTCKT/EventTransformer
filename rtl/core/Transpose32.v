// -----------------------------------------------------------------------------
// Transpose32 : 32x32 바이트 corner-turn (행 쓰기 → 열 읽기)
//
//   쓰기 워드 d :  wdata 의 바이트 t  =  X[t][d]      (t = 0..31)
//   읽기 워드 t :  rdata 의 바이트 d  =  X[t][d]      (d = 0..31)
//
// ## 왜 필요한가 — attention 에서 V 만 전치가 필요합니다
//
// `Gemm_Core` 는 **A 와 B 둘 다 워드가 reduce 인덱스, 레인이 non-reduce 인덱스**
// 로 읽습니다. attention 의 두 MAC 을 그 형태에 맞춰 보면:
//
//   Q·Kᵀ   C[m][n] = Σ_d Q[m][d]·K[n][d]        reduce = d (head_dim 32)
//          A 워드[d] 레인 = query,  B 워드[d] 레인 = key
//          → in_proj 이 출력채널 c 마다 32행 컬럼을 뱉으므로 **Q·K 둘 다 그대로 맞음**
//
//   attn·V C[m][n] = Σ_j attn[m][j]·V[j][n]     reduce = j (Lk)
//          A 워드[j] 레인 = query  ← score GEMM 의 컬럼 출력이 정확히 이 모양
//          B 워드[j] 레인 = d      ← in_proj 은 워드[d] 레인=token 을 줌  **불일치**
//
// 즉 **V 하나만** 축을 돌려야 합니다. reduce 축이 head_dim(32)에서 토큰(Lk)으로
// 바뀌는 유일한 자리이기 때문입니다.
//
// ## 어디에 두는가
//
// in_proj GEMM 이 V 컬럼을 뱉는 자리에 바로 답니다. 타일 하나(32토큰 x 32채널)를
// 모은 뒤 전치해 A_Mem 에 쓰면, 별도의 읽기/쓰기 패스가 없습니다.
//
// ## 구현
//
// 레지스터 32x32 바이트. 쓰기는 행 하나(32바이트)를 한 사이클에, 읽기는 열 하나를
// 32:1 먹싱으로 뽑습니다. BRAM 으로는 못 합니다 — 열 읽기가 32개 주소를 동시에
// 요구하기 때문입니다. FF 8,192개 + 32:1 8비트 mux 32벌이면 충분합니다.
//
// 읽기는 **조합**입니다. 소비자(Skew_Buf)가 어차피 레지스터를 물고 있어서 여기서
// 한 단 더 잡으면 지연만 늘어납니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module Transpose32 #(
    parameter N = 32,
    parameter W = 8
)(
    input  wire              clk,
    input  wire              rst,

    // ---- 쓰기 : 워드 d (바이트 t = X[t][d]) ----
    input  wire              we,
    input  wire [4:0]        w_idx,        // d
    input  wire [N*W-1:0]    w_data,

    // ---- 읽기 : 워드 t (바이트 d = X[t][d]) ----
    input  wire [4:0]        r_idx,        // t
    output wire [N*W-1:0]    r_data
);
    // buf[d] = 쓰기 워드 d 그대로. buf[d] 의 바이트 t 가 X[t][d].
    reg [N*W-1:0] tbuf [0:N-1];

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < N; i = i + 1) tbuf[i] <= {N*W{1'b0}};
        end else if (we) begin
            tbuf[w_idx] <= w_data;
        end
    end

    // 읽기 워드 t 의 바이트 d = buf[d] 의 바이트 t
    genvar d;
    generate
        for (d = 0; d < N; d = d + 1) begin : COL
            assign r_data[d*W +: W] = tbuf[d][r_idx*W +: W];
        end
    endgenerate
endmodule
