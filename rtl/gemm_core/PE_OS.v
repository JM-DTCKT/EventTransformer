// -----------------------------------------------------------------------------
// PE_OS : output-stationary PE — DSP48E2 하나 + **ping-pong 누산기**
//
// 누산기가 DSP 의 P 하나뿐이면 컬럼을 다 뽑아낼 때까지 다음 타일을 시작할 수
// 없습니다. shadow 를 하나 더 두어 읽기와 계산을 분리합니다.
//
//   P       : 지금 타일을 누적                 (ping)
//   shadow  : 직전 타일의 최종값을 들고 있음   (pong)  ← 읽어내기는 여기서
//
// ## 두 신호 모두 systolic 입니다
//
// output-stationary 배열에서 PE[i][j] 는 대각 wavefront 를 따라 끝납니다:
//
//     k 를 받는 시각 = (타일 위상) + 1 + i + j
//     P 에 반영      = + 2
//     ⇒ PE[i][j] 확정 = K + i + j + 2
//
// 그래서 snap 도 clr 도 **전역이면 안 됩니다** — 전역이면 가장 늦게 끝나는
// PE[31][31](K+64)를 모두가 기다려야 합니다. 둘 다 A 와 같은 방향으로 흘리면
// PE 마다 자기 시각에 걸려 타일 주기가 **K+4** 까지 내려갑니다.
//
//     snap_in : shadow <= P   (그 PE 가 막 확정된 순간)
//     clr_in  : P <= 0        (그 다음 사이클, 새 타일 데이터 도착 직전)
//
// DSP48E2 매핑(AREG/BREG=0, MREG=1, PREG=1, OPMODE 상수)은 타이밍이 검증된
// 값입니다 — 건드리지 마십시오.
//
// 비용 : PE 당 shadow 32 FF + 전파 2 FF → 배열 전체 약 34.8K FF (+6.4 %p).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module PE_OS #(
    parameter ACT_W  = 8,
    parameter PSUM_W = 32
)(
    input  wire clk,
    input  wire rst,                    // 전역 리셋 (파워온 클리어)
    input  wire ce,
    input  wire clr_in,                 // 이 PE 의 타일 시작 (대각 wavefront)
    output wire clr_out,                // 오른쪽 PE 로 (1클럭 뒤)
    input  wire snap_in,                // 이 PE 의 확정 시점 → shadow 로 복사
    output wire snap_out,
    input  wire signed [ACT_W-1:0] a_in,
    input  wire signed [ACT_W-1:0] b_in,
    output wire signed [ACT_W-1:0] a_out,
    output wire signed [ACT_W-1:0] b_out,
    output wire signed [PSUM_W-1:0] acc_out   // = shadow (읽는 동안 안 바뀜)
);
    // 두 wavefront 전달 레지스터 — `ce` 와 무관하게 매 사이클 전진합니다.
    // 데이터가 없는 구간에도 파면은 계속 흘러야 하기 때문입니다.
    reg clr_r, snap_r;
    always @(posedge clk) begin
        if (rst) begin clr_r <= 1'b0; snap_r <= 1'b0; end
        else     begin clr_r <= clr_in; snap_r <= snap_in; end
    end
    assign clr_out  = clr_r;
    assign snap_out = snap_r;

    wire rc = rst | clr_in;
    reg signed [ACT_W-1:0] a_reg, b_reg;
    always @(posedge clk) begin
        if (rc) begin a_reg <= 0; b_reg <= 0; end
        else if (ce) begin a_reg <= a_in; b_reg <= b_in; end
    end
    assign a_out = a_reg;
    assign b_out = b_reg;

    wire [47:0] P;
    DSP48E2 #(
        .USE_MULT("MULTIPLY"),
        .USE_PATTERN_DETECT("NO_PATDET"),
        .AREG(0), .BREG(0), .ACASCREG(0), .BCASCREG(0),
        .ADREG(0), .DREG(0), .MREG(1), .PREG(1),
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .OPMODEREG(0), .ALUMODEREG(0), .INMODEREG(0),
        .USE_SIMD("ONE48")
    ) u_dsp (
        .CLK   (clk),
        .CEM   (ce), .CEP(ce),
        .RSTM  (rc), .RSTP(rc),

        .A     ({{(30-ACT_W){a_in[ACT_W-1]}}, a_in}),
        .B     ({{(18-ACT_W){b_in[ACT_W-1]}}, b_in}),
        .C     (48'd0),
        .PCIN  (48'd0),
        .CARRYIN(1'b0), .CARRYINSEL(3'b000), .INMODE(5'b00000),

        .OPMODE  (9'b000100101),
        .ALUMODE (4'b0000),

        .P     (P)
    );

    // ---- ping-pong : 확정 순간에 한 번 복사 ----
    // `snap_in` 과 다음 타일의 `clr_in` 이 같은 엣지에 와도 안전합니다 —
    // 둘 다 non-blocking 이라 shadow 는 **클리어 전의 P** 를 잡습니다.
    reg signed [PSUM_W-1:0] shadow;
    always @(posedge clk) begin
        if (rst)          shadow <= {PSUM_W{1'b0}};
        else if (snap_in) shadow <= P[PSUM_W-1:0];
    end
    assign acc_out = shadow;
endmodule
