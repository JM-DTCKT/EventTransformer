// ============================================================================
//  Q411_To_Fix.v  --  Q4.11 정수 코드 -> signed 고정소수점 Q(IW-1-IF).IF (조합)
// ----------------------------------------------------------------------------
//  입력 code 는 2의 보수 16비트이고 값은 code * 2^-QF 입니다 (QF=11 -> Q4.11).
//  목표값이 x = value * 2^IF * 2^xsh 이므로
//
//     x = code * 2^sh ,   sh = IF - QF + xsh
//
//  즉 **코드를 sh 만큼 시프트**하는 것이 전부입니다.  `bf16_to_fix` 와 달리
//  정규화(선행 0 카운트)가 없습니다 — 코드가 이미 고정소수점이기 때문입니다.
//
//  [왜 만들었나]
//  예전에는 `Int32_To_Bf16` 으로 bf16 을 거쳐 `bf16_to_fix` 에 넣었는데,
//  **정규화했다가 곧바로 역정규화**하는 셈이라 배럴 시프터 두 개가 한 사이클에
//  직렬로 놓였습니다 (ZU9EG -2, A_Mem BRAM -> x_p 실측 8.475 ns).  직행하면
//  시프터 하나로 끝나고, bf16 가수 8비트로 떨어뜨리던 손실도 없어집니다.
//  264 샘플 정확도는 양쪽 동일했습니다 (오답 샘플 번호까지 같음).
//
//  [규약]  `bf16_to_fix` 와 맞춥니다 :
//     - 반올림은 round-half-up (부호는 마지막에 적용하므로 0 기준 대칭)
//     - 포화값은 +-(2^(IW-1)-1) 로 **대칭** 이라 -2^(IW-1) 은 절대 나오지 않는다
//       -> 이후 X*X 가 2^(2IW-2) 를 넘지 않음이 보장된다
//
//  [경계]  IW=24, IF=15, QF=11 기준 (sh = 4 + xsh)
//     sh >= IW-1 (=23)  -> 포화 (code != 0 이면 반드시 넘침),  ovf=1
//     sh <  -16         -> |code| <= 2^15 이므로 반올림해도 0
//     그 외             -> 시프트 후 결과 상위비트로 포화 판정 (정확)
//     실제 스케줄은 세 ln_2 step 모두 xsh=+2 -> sh=+6 (무손실 좌시프트)
// ============================================================================
`timescale 1ns/1ps

module Q411_To_Fix #(
    parameter integer IW  = 24,   // 출력 폭 (signed Q(IW-1-IF).IF)
    parameter integer IF  = 15,   // 출력 소수부 비트수
    parameter integer QF  = 11,   // 입력 소수부 비트수 (Q4.11)
    parameter integer XSW = 6     // xsh 폭 (signed)
)(
    input  wire signed [15:0]     code,   // Q4.11 정수 코드
    input  wire signed [XSW-1:0]  xsh,    // 추가 지수 시프트 (value * 2^xsh)
    output wire signed [IW-1:0]   x,      // signed Q(IW-1-IF).IF
    output wire                   ovf     // 포화 발생
);
    localparam integer          SHW    = 12;                // sh 폭 (signed)
    localparam signed [SHW-1:0] SH_OFS = IF - QF;           // = 4
    localparam signed [SHW-1:0] SH_HI  = IW - 1;            // = 23, 이상이면 포화
    localparam signed [SHW-1:0] SH_LO  = -16;               // 미만이면 0
    localparam        [IW-2:0]  MAXMAG = {(IW-1){1'b1}};    // 대칭 포화값
    localparam integer          WL     = 16 + (IW - 1);     // 좌시프트 중간폭 (=39)

    wire        s   = code[15];
    wire [15:0] mag = s ? (~code + 16'd1) : code;           // |code| (0x8000 은 그대로)

    wire signed [SHW-1:0] xsh_s = {{(SHW-XSW){xsh[XSW-1]}}, xsh};
    wire signed [SHW-1:0] sh    = SH_OFS + xsh_s;

    wire is_zero = (mag == 16'd0);
    wire neg_sh  = sh[SHW-1];
    wire hi      = (sh >= SH_HI) & ~is_zero;                // 무조건 포화
    wire lo      = (sh <  SH_LO);                           // 무조건 0

    // ---- 좌시프트 (sh >= 0) : 무손실 ----
    wire [4:0]    shl = (neg_sh | hi | lo) ? 5'd0 : sh[4:0];
    wire [WL-1:0] wl  = ({{(WL-16){1'b0}}, mag} << shl);

    // ---- 우시프트 (sh < 0) : round-half-up ----
    wire signed [SHW-1:0] nsh_s = -sh;
    wire [4:0]    shr = (neg_sh & ~lo) ? nsh_s[4:0] : 5'd0;
    wire [16:0]   rnd = (shr == 5'd0) ? 17'd0 : (17'd1 << (shr - 5'd1));
    wire [16:0]   wr  = ({1'b0, mag} + rnd) >> shr;

    wire [WL-1:0] sel = neg_sh ? {{(WL-17){1'b0}}, wr} : wl;

    wire          sat  = hi | (|sel[WL-1:IW-1]);
    wire [IW-2:0] magf = sat ? MAXMAG
                       : lo  ? {(IW-1){1'b0}}
                             : sel[IW-2:0];

    assign x   = s ? -$signed({1'b0, magf}) : $signed({1'b0, magf});
    assign ovf = sat;

    // ---- 포맷 정합성 체크 (합성 무관) --------------------------------------
    initial begin
        if (IW < 18 || IW > 31)
            $display("ERROR: Q411_To_Fix IW(%0d) out of supported range [18,31]", IW);
    end
endmodule
