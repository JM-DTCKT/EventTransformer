// -----------------------------------------------------------------------------
// PE_OS : output-stationary PE — DSP48E2 하나 + **ping-pong 누산기**
//
// 누산기가 DSP 의 P 하나뿐이면 컬럼을 다 뽑아낼 때까지 다음 타일을 시작할 수
// 없습니다. shadow 를 하나 더 두어 읽기와 계산을 분리합니다.
//
//   P       : 지금 타일을 누적                 (ping)
//   shadow  : 직전 타일의 최종값을 들고 있음   (pong)  ← 읽어내기는 여기서
//
// ## DSP 패킹 — 한 DSP 가 곱셈 두 개
//
// DSP48E2 의 곱셈기는 27x18 입니다. 활성값(int8)을 18비트 B 포트에 두면 27비트
// 쪽에 **가중치 두 개를 얹을 수 있습니다**. 내장 pre-adder 로 공짜입니다:
//
//     A 포트 = w1 << PK_S      (b_in[7:4] 부호확장, 모드 무관)
//     D 포트 = w0              (b_in[3:0])
//     AD     = D + A = w1*2^PK_S + w0          ← pre-adder, LUT 0개
//     B 포트 = a               (활성값 int8)
//
//     P += a*w0 + (a*w1) * 2^PK_S
//
// 두 곱이 P 안에서 **자리로 분리**됩니다. PK_S=21 이면 하위 [20:0], 상위
// [40:21] 이고, K<=160 · |a|<=128 · |w|<=8 이면 |누산| <= 163,840 < 2^20 이라
// 겹치지 않습니다. 하위가 음수면 상위에서 1을 빌려가므로 읽을 때 되돌립니다
// (`Gemm_Core` 의 필드 추출 참고).
//
// ## 두 모드가 회로를 공유합니다
//
//   Linear     b_in = {w1, w0} 두 int4      → 하위=출력채널 n, 상위=n+32
//   attention  b_in = int8 하나              → int8 = b[7:4]*16 + b[3:0](부호없음)
//                                              이므로 하위*1 + 상위*16 로 복원
//
// 그래서 PE 안의 모드 분기는 **D 의 부호확장 1비트뿐**입니다 (`pack & b_in[3]`).
// 필드 추출과 복원은 컬럼 mux 뒤 32레인에서만 하므로 배열 밖입니다.
//
// ## 두 신호 모두 systolic 입니다
//
// output-stationary 배열에서 PE[i][j] 는 대각 wavefront 를 따라 끝납니다:
//
//     k 를 받는 시각 = (타일 위상) + 1 + i + j
//     P 에 반영      = + DSP_LAT (= 3 : ADREG/BREG -> MREG -> PREG)
//     ⇒ PE[i][j] 확정 = K + i + j + 3
//
// 그래서 snap 도 clr 도 **전역이면 안 됩니다** — 전역이면 가장 늦게 끝나는
// PE[31][31] 를 모두가 기다려야 합니다. 둘 다 A 와 같은 방향으로 흘리면
// PE 마다 자기 시각에 걸려 타일 주기가 **K+4** 까지 내려갑니다.
//
//     snap_in : shadow <= P   (그 PE 가 막 확정된 순간)
//     clr_in  : P <= 0        (그 다음 사이클, 새 타일 데이터 도착 직전)
//
// DSP48E2 매핑은 시뮬레이션으로 검증한 값입니다 — 특히 `INMODE=5'b00100` 이
// pre-adder 를 `D + A` 로 만드는 유일한 조합입니다. 건드리지 마십시오.
//
// 비용 : PE 당 shadow SH_W FF + 전파 2 FF. BREG/ADREG 는 DSP 내부라 공짜입니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module PE_OS #(
    parameter ACT_W  = 8,
    parameter PK_S   = 21,      // 상위 곱의 자리 (하위 누산이 넘지 않을 만큼)
    parameter SH_W   = 41       // shadow 폭 = PK_S + 20
)(
    input  wire clk,
    input  wire rst,                    // 전역 리셋 (파워온 클리어)
    input  wire ce,
    input  wire pack,                   // 1 = b_in 이 int4 두 개
    input  wire clr_in,                 // 이 PE 의 타일 시작 (대각 wavefront)
    output wire clr_out,                // 오른쪽 PE 로 (1클럭 뒤)
    input  wire snap_in,                // 이 PE 의 확정 시점 → shadow 로 복사
    output wire snap_out,
    input  wire signed [ACT_W-1:0] a_in,
    input  wire signed [ACT_W-1:0] b_in,
    output wire signed [ACT_W-1:0] a_out,
    output wire signed [ACT_W-1:0] b_out,
    output wire        [SH_W-1:0]  acc_out   // = shadow (읽는 동안 안 바뀜)
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

    // ---- DSP 입력 ----
    // A : 상위 니블을 PK_S 자리로. 모드와 무관하므로 mux 가 없습니다.
    wire [29:0] dsp_a = {{(30-PK_S-4){b_in[ACT_W-1]}}, b_in[ACT_W-1:ACT_W-4],
                         {PK_S{1'b0}}};
    // D : 하위 니블. Linear 은 int4 라 부호확장, attention 은 int8 의 하위
    //     니블이라 **부호없이** 채웁니다 (int8 = 상위*16 + 하위_unsigned).
    wire        d_sx  = pack & b_in[3];
    wire [26:0] dsp_d = {{23{d_sx}}, b_in[3:0]};
    wire [17:0] dsp_b = {{(18-ACT_W){a_in[ACT_W-1]}}, a_in};

    wire [47:0] P;
    DSP48E2 #(
        .USE_MULT("MULTIPLY"),
        .USE_PATTERN_DETECT("NO_PATDET"),
        .AREG(0), .BREG(1), .ACASCREG(0), .BCASCREG(1),
        .ADREG(1), .DREG(0), .MREG(1), .PREG(1),
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .AMULTSEL("AD"), .PREADDINSEL("A"),
        .OPMODEREG(0), .ALUMODEREG(0), .INMODEREG(0),
        .USE_SIMD("ONE48")
    ) u_dsp (
        .CLK   (clk),
        .CEA1(ce), .CEA2(ce), .CEB1(ce), .CEB2(ce), .CED(ce), .CEAD(ce),
        .CEM   (ce), .CEP(ce),
        .CEC(1'b0), .CECARRYIN(1'b0), .CECTRL(1'b0), .CEALUMODE(1'b0),
        .CEINMODE(1'b1),
        // **입력 레지스터는 전역 리셋만** 겁니다. `clr_in` 은 그 PE 의 타일 시작
        // 이라 k=0 이 BREG/ADREG 에 막 담기는 바로 그 엣지입니다 — 여기서 지우면
        // 첫 피연산자가 사라집니다. 누산기(M/P)만 비우면 충분합니다.
        .RSTA(rst), .RSTB(rst), .RSTD(rst), .RSTM(rc), .RSTP(rc), .RSTC(1'b0),
        .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),

        .A     (dsp_a),
        .B     (dsp_b),
        .D     (dsp_d),
        .C     (48'd0),
        .PCIN  (48'd0),
        .CARRYIN(1'b0), .CARRYINSEL(3'b000),
        .INMODE(5'b00100),               // AD = D + A  (검증된 유일 조합)

        .OPMODE  (9'b000100101),
        .ALUMODE (4'b0000),

        .P     (P)
    );

    // ---- ping-pong : 확정 순간에 한 번 복사 ----
    // `snap_in` 과 다음 타일의 `clr_in` 이 같은 엣지에 와도 안전합니다 —
    // 둘 다 non-blocking 이라 shadow 는 **클리어 전의 P** 를 잡습니다.
    reg [SH_W-1:0] shadow;
    always @(posedge clk) begin
        if (rst)          shadow <= {SH_W{1'b0}};
        else if (snap_in) shadow <= P[SH_W-1:0];
    end
    assign acc_out = shadow;
endmodule
