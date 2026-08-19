module PE_OS #(
    parameter ACT_W  = 8,
    parameter PSUM_W = 32
)(
    input  wire clk,
    input  wire rst,                    // 전역 리셋 (파워온 클리어)
    input  wire clr,                    // 타일 시작: 누산기(P)·전달레지스터 0
    input  wire ce,                     // 스트리밍 중 전진/누적
    input  wire signed [ACT_W-1:0] a_in,   // 왼쪽에서
    input  wire signed [ACT_W-1:0] b_in,   // 위에서
    output wire signed [ACT_W-1:0] a_out,  // 오른쪽으로 (1클럭 지연)
    output wire signed [ACT_W-1:0] b_out,  // 아래로 (1클럭 지연)
    output wire signed [PSUM_W-1:0] acc_out
);
    // -----------------------------------------------------------------------
    // Output-Stationary PE : DSP48E2 accumulate 모드 (P = P + A*B, Z=P 피드백).
    //   각 PE 가 자기 출력 C[i][j] 를 로컬 누적 (WS 의 PCIN cascade 와 대비).
    //   A/B 입력 레지스터를 두지 않아(AREG=BREG=ADREG=0) A·B 가 같은 시점에 곱해짐.
    //   MREG+PREG = 2단 파이프라인 (컨트롤러 drain 이 flush 해줌).
    //   OPMODE 는 상수라 OPMODEREG=0 (제어 레지스터 로딩 transient 제거).
    //   ZCU102(UltraScale+) DSP48E2. 시뮬엔 Vivado unisims + glbl 필요.
    // -----------------------------------------------------------------------

    // systolic 전달 레지스터 (A 좌→우, B 위→아래) — DSP 와 별개
    wire rc = rst | clr;                 // 리셋 또는 타일 시작
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

        .A     ({{(30-ACT_W){a_in[ACT_W-1]}}, a_in}),   // 30-bit A (sign-ext)
        .B     ({{(18-ACT_W){b_in[ACT_W-1]}}, b_in}),   // 18-bit B (sign-ext)
        .C     (48'd0),
        .PCIN  (48'd0),
        .CARRYIN(1'b0), .CARRYINSEL(3'b000), .INMODE(5'b00000),

        // P = Z + X + Y = P + M(=A*B).  OPMODE[6:4]=010(Z=P) [3:2]=01 [1:0]=01
        .OPMODE  (9'b000100101),
        .ALUMODE (4'b0000),

        .P     (P)
    );

    assign acc_out = P[PSUM_W-1:0];
endmodule
