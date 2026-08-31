// -----------------------------------------------------------------------------
// softmax_top : attention softmax — **`SOFTMAX/softmax_unit` 래퍼**
//
// 엔진 쪽 인터페이스(start/C/done, in_valid/in_data, out_valid/out_c/out_data)는
// 예전 판과 **한 글자도 다르지 않습니다.** `EvT_Engine` 은 이 파일이 바뀐 줄
// 모릅니다. 안쪽만 새 코어로 갈아끼웠습니다.
//
// ## 왜 바꿨나 — 여기가 지배적 비용이었습니다
//
// 예전 판은 레인 하나를 시분할해 `4·N·C ≈ 6,800` 사이클(N=32, Lk=53)이 들었고,
// 이걸 (행타일 3) x (head 4) x (블록 3) = **36번** 도니 타임스텝 53만 사이클 중
// **약 24만**을 softmax 가 썼습니다.
//
// 새 코어는 열(32행)을 매 사이클 받아 max 를 수신 중에 확정하고, RECV/EXP/MUL
// 세 단계를 **서로 다른 Tile 로 겹칩니다**. 한 Tile 이 `T + ~40` 사이클입니다.
//
// ## 두 가지를 맞춰 줍니다
//
//  1. **끝 표시** — 예전 판은 `C`(=Lk)를 받아 스스로 셌지만 새 코어는 `in_last`
//     로 받습니다. 여기서 `C-1` 번째 열에 실어 줍니다.
//
//  2. **출력 포맷** — 새 코어는 signed Q1.14 를 내고, `attn·V` 의 A 피연산자는
//     **uint8 코드**(골든의 `attn_int = round(attn / softmax_scale)`)입니다.
//     softmax_scale = 16513·2^-21 이므로
//
//         code = round( y_q14 · 2^-14 / softmax_scale )
//              = (y_q14 · SM_MULT + 2^(SM_SH-1)) >> SM_SH
//
//     기본값 SM_MULT=16253, SM_SH=21 이 그 비율입니다 (상대오차 3e-6).
//     y 는 [0,1] 이라 부호가 안 서고, y=1 이어도 code=127 이라 포화도 없습니다.
// -----------------------------------------------------------------------------
module softmax_top #(
    parameter N        = 32,        // 레인 = 쿼리
    parameter CMAX     = 128,       // 최대 클래스 수 = Lk 상한
    parameter SM_MULT  = 16253,     // Q1.14 → uint8 (softmax_scale 의 역수)
    parameter SM_SH    = 21
)(
    input  wire              clk,
    input  wire              rst,

    input  wire              start,      // C 를 잡고 입력 수집 시작
    input  wire [7:0]        C,          // 클래스 수 = Lk
    output reg               done,

    // ---- 입력: 클래스 하나 = 32레인 Q6.9 ----
    input  wire              in_valid,
    input  wire [N*16-1:0]   in_data,

    // ---- 출력: 클래스 하나 = 32레인 uint8 ----
    output reg               out_valid,
    output reg  [7:0]        out_c,
    output reg  [N*8-1:0]    out_data
);
    // =========================================================================
    // 입력 : C 를 세어 마지막 열에 in_last
    //
    // `C` 를 레지스터로 옮겨 담지 않습니다. 엔진이 step 내내 `q_NOUT` 을 잡고
    // 있으므로 그대로 쓰면 됩니다. 옮겨 담으려다 **첫 열을 흘리는** 실수를
    // 했었는데, 그러면 그 타일이 영영 안 끝나고 출력이 **한 타일씩 밀립니다**
    // (다음 타일의 첫 열이 들어와야 앞 타일이 완성됩니다).
    // =========================================================================
    wire       core_iready;
    wire [7:0] Ceff  = (C == 8'd0) ? 8'd1 : C;
    reg  [7:0] icnt;
    wire       take    = start && (icnt < Ceff);      // 아직 받을 열이 남았나
    wire       in_last = (icnt == Ceff - 8'd1);

    always @(posedge clk) begin
        if (rst || !start) icnt <= 8'd0;
        else if (in_valid && core_iready && take) icnt <= icnt + 8'd1;
    end

    // =========================================================================
    // 코어
    // =========================================================================
    wire              core_ovalid, core_olast;
    wire [N*16-1:0]   core_ocol;

    softmax_unit #(.Tile_M(N), .TMAX(CMAX), .DW(16), .IF(9),
                  .OW(16), .OF(14), .EW(17), .EF(16), .RW(18), .RF(17),
                  .BRAM_LAT(1)) u_core (
        .clk(clk), .rst_n(~rst),
        .in_valid(in_valid && take), .in_ready(core_iready),
        .in_col(in_data), .in_last(in_last),
        .out_valid(core_ovalid), .out_col(core_ocol), .out_last(core_olast));

    // =========================================================================
    // 출력 : Q1.14 → uint8, 열 번호를 세어 붙임
    // =========================================================================
    // 변환은 **조합**으로 하고 레지스터에는 비블로킹으로만 담습니다.
    // (클럭 블록 안에서 출력 레지스터에 블로킹 대입을 하면 같은 엣지에 읽는
    //  쪽과 경쟁해 값이 갈립니다 — attention 결과가 통째로 깨졌습니다.)
    wire [N*8-1:0] u8_bus;
    genvar q;
    generate
        for (q = 0; q < N; q = q + 1) begin : Q2U8
            wire signed [15:0] yq = core_ocol[q*16 +: 16];
            wire signed [47:0] pr = ($signed({{32{yq[15]}}, yq}) * SM_MULT
                                     + (48'sd1 <<< (SM_SH - 1))) >>> SM_SH;
            assign u8_bus[q*8 +: 8] = (pr < 0)   ? 8'd0
                                    : (pr > 255) ? 8'd255 : pr[7:0];
        end
    endgenerate

    // `out_c` 는 **지금 나가는** 열 번호여야 합니다. 데이터와 같은 엣지에
    // 증가시키면 첫 열이 1번 자리에 써져 전체가 한 칸씩 밀립니다.
    reg [7:0] ocnt;
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; out_c <= 8'd0; out_data <= {(N*8){1'b0}};
            done <= 1'b0; ocnt <= 8'd0;
        end else begin
            out_valid <= core_ovalid;
            if (core_ovalid) begin
                out_data <= u8_bus;
                out_c    <= ocnt;
                ocnt     <= core_olast ? 8'd0 : (ocnt + 8'd1);
            end
            // 마지막 열을 내보낸 뒤 완료. `start` 가 내려가면 해제합니다.
            if (core_ovalid && core_olast) done <= 1'b1;
            else if (!start)               done <= 1'b0;
        end
    end
endmodule
