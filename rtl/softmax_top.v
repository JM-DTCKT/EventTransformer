// ============================================================================
//  softmax_top.v    --  32행 Tile 단위 T-축 softmax  (Q6.9 in / signed Q1.14 out)
// ----------------------------------------------------------------------------
//  attention 의 QK^T 결과 (Tile_M x T) 에 대해 **T 축(키 축)** 으로 softmax 한다.
//  Tensor Core 출력을 **열 단위(32원소)** 로 받아 32 x T Tile 을 모으고,
//  32개 행의 softmax 를 한꺼번에 처리한다.
//
//     for r in 0..Tile_M-1:  y[r][0..T-1] = softmax( x[r][0..T-1] )
//
//  [입력]  매 clk "한 열" = 32행 각각의 원소 1개  (Tile_M x DW = 512-bit)
//    · 열 하나가 32행에 1원소씩이므로 **부분 beat 가 없다** -> keep 마스크 불필요
//    · T 는 런타임 가변(<= TMAX). 마지막 열에 in_last 를 실어준다.
//
//  [핵심 1] 수신하면서 max 를 같이 구한다
//    데이터가 어차피 버스를 지나가므로 **비교기 Tile_M개**만 얹으면 추가 사이클/추가
//    메모리 읽기 없이 행별 max 가 확정된다.  -> 이후 단계엔 max 로직이 아예 없다.
//
//      rmax[r] <= (t==0) ? x[r] : max(rmax[r], x[r])     매 clk Tile_M개 동시
//
//  [핵심 2] 3개의 독립 FSM 을 Tile 단위로 파이프라인
//    한 Tile 안에서는 max 가 전역 의존성이라 RECV 와 EXP 를 겹칠 수 없다.
//    대신 **서로 다른 Tile** 을 세 단계가 동시에 처리한다.
//
//      Tile k+2 :  [S1 RECV  T=324 clk ]
//      Tile k+1 :               [S2 EXP+RCP  1+(T+L+4)+(Tile_M+3) = 365 clk ]
//      Tile k   :                            [S3 MUL  1+(T+L) = 326 clk ]
//                 주기 = max(324, 365, 326) = 365 clk    (L = BRAM_LAT = 1)
//
//    S1 RECV    : in_col -> xbuf[뱅크] + rmax 갱신                    T clk
//    S2 EXP+RCP : z = x - rmax[r] -> exp2_unit xTile_M -> ebuf[뱅크] T+L+4 clk
//                 (BRAM 읽기 L단 + exp 파이프 4단)
//                 행별 sum Tile_M개 누산, 이어서 행별 1/S 를 recip 1개로  Tile_M+3 clk
//    S3 MUL     : y = e * R[r] >> (p[r]+3) -> out_col              T+L clk
//
//  [버퍼를 몇 벌 두는가]  "생산자와 소비자가 서로 다른 Tile 을 동시에 다루는가"
//    xbuf : S1(Tile k+2 수신) ‖ S2(Tile k+1 소비) -> 다른 Tile -> **2 뱅크**
//           뱅크 하나 = TMAX word x (Tile_M*DW) = 32 x T 원소 전부 = 166 kbit
//    ebuf : S2(Tile k+1 생산) ‖ S3(Tile k 소비)   -> 다른 Tile -> **2 뱅크**
//           뱅크 하나 = TMAX word x (Tile_M*EW)                     = 176 kbit
//    rr/pp/etlen 도 Tile 별 상태라 2벌 (작음)
//    합계 : xbuf 332 kbit + ebuf 352 kbit = 684 kbit
//
//    ebuf 를 2벌로 둔 이유는 **S2 와 S3 가 서로 다른 연산기를 쓰는데 1벌이면
//    직렬화되기 때문**이다.  1벌일 때는 EXP 동안 곱셈기 32개가, MUL 동안 exp
//    유닛 32개가 놀아서 가동률이 ~48% 였다.  2벌이면 둘 다 ~90% 로 올라간다.
//
//  [메모리 추론]  xbuf + ebuf = 684 kbit 라 FF 로는 불가능하다 (68만 개).
//    둘 다 **등록형 read** 로 작성해 BRAM(simple dual-port) 으로 추론되게 했다.
//    조합 read 로 두면 distributed RAM 이 되어 LUT 을 수천 개 먹는다.
//    읽기 지연은 파라미터 BRAM_LAT 로 조절한다 :
//      1 = BRAM 코어만        -> 주기 365, e2e(3) = 1,745 clk   (기본)
//      2 = 출력 레지스터까지  -> 주기 366, e2e(3) = 1,749 clk
//    차이가 4 clk (0.2%) 뿐이므로 **사이클이 아니라 타이밍 클로저로 정한다**.
//    ~400MHz 까지는 1 로 충분하고, 500MHz 에서 WNS 가 음수면 2 로 올리면 된다
//    (BRAM 출력 레지스터 사용).  파라미터 한 줄이라 합성 후 뒤집어도 된다.
//
//  [고정소수점 포맷]
//    x (in_col)   signed   Q6.9    16b   범위 [-64, 64),  LSB 2^-9 = 1.95e-3
//    z (내부)     signed   Q7.9    17b   = x - max,  범위 (-128, 0]
//    e (내부)     unsigned UQ1.16  17b   exp 결과 (0,1],  1.0 = 2^16
//    S (내부)     unsigned UQ10.16 26b   sum(e), 범위 [1, TMAX]
//    R (내부)     unsigned UQ1.17  18b   1/m  (m = S/2^p, m in [1,2))
//    y (out_col)  **signed** Q1.14 16b   범위 [0,1],      1.0 = 2^14 = 16'h4000
//                 (부호 1 + 정수 1 + 소수 14).  softmax 출력은 항상 >= 0 이고
//                 y <= 1.0 = 0x4000 < 0x8000 이므로 **부호비트가 절대 서지 않는다**
//                 -> 포화(saturation) 로직 불필요.  TB 에서 전 원소 검증한다.
//
//    y = e/S = e*R*2^-p  이므로  y_out = (e*R) >> (p + RF - OF) = >> (p+3)
//    * e(17b) x R(18b) = 18s x 19s -> DSP48 (25x18 / 27x18) 1개에 정확히 들어간다.
//    * 시프트량 p 는 **행마다 다르므로** lane 별 배럴 시프터가 필요하다
//      (p in [16,24] -> sh in [19,27], 9가지뿐이라 9:1 mux 규모).
// ============================================================================
`timescale 1ns/1ps

module softmax_top #(
    parameter integer Tile_M = 32,   // Tile 행 수 (= Tensor Core 타일 행 수)
    parameter integer TMAX   = 324,  // 최대 T (정규화 축 길이)
    parameter integer DW     = 16,   // 입력 원소 폭 (signed Q6.9)
    parameter integer IF     = 9,    // 입력 소수부 비트수
    parameter integer OW     = 16,   // 출력 폭 (signed Q1.14 : 부호1+정수1+소수14)
    parameter integer OF     = 14,   // 출력 소수부 비트수 (signed Q1.14)
    parameter integer EW     = 17,   // 내부 exp 폭 (unsigned UQ1.16, = EF+1)
    parameter integer EF     = 16,   // 내부 exp 소수부 비트수
    parameter integer RW     = 18,   // 역수 폭 (unsigned UQ1.17, = RF+1)
    parameter integer RF     = 17,   // 역수 소수부 비트수
    // BRAM read latency : 1 = 코어만,  2 = 출력 레지스터까지 (고Fmax 권장)
    parameter integer BRAM_LAT = 1
)(
    input                        clk,
    input                        rst_n,
    // ---- 입력 : 매 clk 한 열(Tile_M원소), T 회 ----
    input                        in_valid,
    output                       in_ready,
    input       [Tile_M*DW-1:0]       in_col,    // Q6.9 x Tile_M (행 r = in_col[r*DW +: DW])
    input                        in_last,   // 이 Tile 의 마지막 열
    // ---- 출력 : 매 clk 한 열(Tile_M원소), T 회 ----
    output reg                   out_valid,
    output reg  [Tile_M*OW-1:0]       out_col,   // signed Q1.14 x Tile_M (원소 r = [r*OW +: OW])
    output reg                   out_last
);
    // ---- width 파라미터 ----------------------------------------------------
    localparam integer TW     = $clog2(TMAX);          // = 9,  열 인덱스 폭
    localparam integer TCW    = TW + 1;                // = 10, 열 카운터 폭
    localparam integer MW     = $clog2(Tile_M);        // = 5,  행 인덱스 폭
    localparam integer MCW    = MW + 1;                // = 6,  행 카운터 폭
    localparam integer ZW     = DW + 1;                // = 17, z = x - max 폭
    localparam integer SW     = EF + TW + 1;           // = 26, S = sum(e) 폭
    localparam integer PW     = 5;                     // p (S 의 MSB 위치) 폭
    localparam integer SH_OFF = RF - OF;               // = 3,  최종 시프트 오프셋
    localparam integer PDW    = EW + RW;               // = 35, e*R 곱 폭
    localparam signed [DW-1:0] XMIN = {1'b1, {(DW-1){1'b0}}};   // -64.0

    genvar  gr;
    integer i;

    // ======================================================================
    //  뱅크 full 플래그  (단일 드라이버 — 각 FSM 은 done 펄스만 낸다)
    // ======================================================================
    reg  [1:0] xfull, efull;      // xbuf / ebuf 뱅크가 소비 대기중인가
    reg        wbank;             // S1 이 쓰는 xbuf 뱅크
    reg        xrb, ewb;          // S2 가 읽는 xbuf / 쓰는 ebuf 뱅크
    reg        erb;               // S3 가 읽는 ebuf 뱅크

    wire recv_hs   = in_valid & in_ready;
    wire recv_done;               // S1 : Tile 수신 완료
    wire exp_done;                // S2 : exp+recip 완료
    wire mul_done;                // S3 : 출력 완료

    assign in_ready = ~xfull[wbank];
    assign recv_done = recv_hs & in_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            xfull <= 2'b00; efull <= 2'b00;
        end else begin
            if (recv_done) xfull[wbank] <= 1'b1;   // S1 -> S2 로 넘김
            if (exp_done)  xfull[xrb]   <= 1'b0;   // S2 가 xbuf 반납
            if (exp_done)  efull[ewb]   <= 1'b1;   // S2 -> S3 로 넘김
            if (mul_done)  efull[erb]   <= 1'b0;   // S3 가 ebuf 반납
        end
    end

    // ======================================================================
    //  S1 : 수신 + running max   (Tensor Core 와 독립적으로 굴러감)
    // ======================================================================
    reg  [DW-1:0]   rmax [0:1][0:Tile_M-1];      // 행별 max (수신 중 갱신)
    reg  [TCW-1:0]  tlen [0:1];             // 뱅크별 유효 열 수 T
    reg  [TCW-1:0]  wptr;                   // 수신 열 포인터

    wire signed [DW-1:0] xcol     [0:Tile_M-1];
    wire signed [DW-1:0] rmax_nxt [0:Tile_M-1];
    generate
        for (gr = 0; gr < Tile_M; gr = gr + 1) begin : g_rmax
            assign xcol[gr] = in_col[gr*DW +: DW];
            // 첫 열이면 그대로, 아니면 기존 rmax 와 비교 (비교기 Tile_M개)
            assign rmax_nxt[gr] = (wptr == {TCW{1'b0}}) ? xcol[gr]
                                : (($signed(xcol[gr]) > $signed(rmax[wbank][gr]))
                                   ? xcol[gr] : rmax[wbank][gr]);
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wbank <= 1'b0;
            wptr  <= {TCW{1'b0}};
            for (i = 0; i < Tile_M; i = i + 1) begin
                rmax[0][i] <= XMIN; rmax[1][i] <= XMIN;
            end
            tlen[0] <= {TCW{1'b0}};  tlen[1] <= {TCW{1'b0}};
        end else if (recv_hs) begin
            for (i = 0; i < Tile_M; i = i + 1)
                rmax[wbank][i] <= rmax_nxt[i];               // 비교기 Tile_M개
            wptr <= wptr + 1'b1;
            if (in_last) begin                                // Tile 완성
                tlen[wbank] <= wptr + 1'b1;
                wbank       <= ~wbank;
                wptr        <= {TCW{1'b0}};
            end
        end
    end

    // ======================================================================
    //  S2 : exp + 행별 sum + 행별 역수      (xbuf[xrb] -> ebuf[ewb])
    // ======================================================================
    localparam [1:0] ES_IDLE = 2'd0, ES_EXP = 2'd1, ES_RCP = 2'd2;

    reg  [1:0]      estate;
    reg  [TCW-1:0]  ei, ec, etc_;           // 투입/수집 열 포인터, 이번 Tile 의 T
    reg  [MCW-1:0]  qi, qc;                 // 역수 투입/수집 행 포인터
    reg  [SW-1:0]   sum  [0:Tile_M-1];           // 행별 분모 (S2 안에서만 사용)

    reg  [RW-1:0]   rr    [0:1][0:Tile_M-1];     // 행별 1/m   (Tile 별 상태)
    reg  [PW-1:0]   pp    [0:1][0:Tile_M-1];     // 행별 p     (Tile 별 상태)
    reg  [TCW-1:0]  etlen [0:1];            // 뱅크별 T

    wire [TW-1:0] e_rd = (ei < etc_) ? ei[TW-1:0] : {TW{1'b0}};
    wire          e_iv = (estate == ES_EXP) && (ei < etc_);

    // ==================================================================
    //  xbuf : simple dual-port BRAM   (write = S1,  등록형 read = S2)
    // ------------------------------------------------------------------
    //  BRAM 은 0-latency 조합 읽기가 물리적으로 불가능하므로 **읽기를
    //  레지스터 뒤로** 뺀다.  아래가 Vivado 의 표준 SDP 추론 템플릿이다.
    //     cycle k          : 주소 e_rd 제시
    //     cycle k+BRAM_LAT : xrd 에 데이터 도착 -> z 계산 -> exp 투입
    //  BRAM_LAT 은 **총 읽기 지연 단수**다 (0 은 불가능) :
    //    1 = BRAM 코어 출력 레지스터만        (RAMB 의 필수 1단)
    //    2 = + 선택적 출력 레지스터 (DO_REG)  (고Fmax 용)
    //  UltraScale+ 에서 500MHz 를 노리면 출력 레지스터(=2)가 사실상 필수다.
    //  조합 읽기로 두면 332 kbit 가 distributed RAM 으로 깔려 LUT 을 먹는다.
    // ==================================================================
    // **뱅크 차원을 주소로 펴 둡니다.** `[0:1][0:TMAX-1]` 처럼 2차원으로 두면
    // Vivado 가 3D-RAM 으로 보고 BRAM 추론을 포기해 **FF 132,096개**로 깔립니다
    // (Synth 8-11357). 그러면 배선이 혼잡도 6 으로 실패합니다.
    // `layernorm_top` 의 `xbuf [0:NB*D-1]` 과 같은 형태입니다.
    // 주소를 `{뱅크, 열}` 로 이으므로 깊이는 2^(TW+1) 입니다 (TMAX 가 2의
    // 거듭제곱이 아니면 남는 자리가 생기지만 BRAM 한 벌 안이라 무해합니다).
    reg [Tile_M*DW-1:0] xbuf [0:(2<<TW)-1];     // 입력 Tile
    reg [Tile_M*DW-1:0] xrd_c;   // BRAM 코어 출력 레지스터 (= 읽기 1단째, 필수)
    reg                 xrv_c;

    always @(posedge clk) begin
        if (recv_hs) xbuf[{wbank, wptr[TW-1:0]}] <= in_col;  // write 포트
        xrd_c <= xbuf[{xrb, e_rd}];  // read 포트 : 주소 제시 -> 다음 clk 에 데이터
                                     //   BRAM 은 0-cycle 조합 읽기가 불가능하므로
                                     //   이 1단은 BRAM_LAT 값과 무관하게 항상 있다
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) xrv_c <= 1'b0;
        else        xrv_c <= e_iv;
    end

    wire [Tile_M*DW-1:0] xrd;
    wire                 xrv;
    generate
        if (BRAM_LAT >= 2) begin : g_xoreg  // 선택적 출력 레지스터 추가 -> 총 2단
            reg [Tile_M*DW-1:0] xrd_r;
            reg                 xrv_r;
            always @(posedge clk) xrd_r <= xrd_c;
            always @(posedge clk or negedge rst_n)
                if (!rst_n) xrv_r <= 1'b0; else xrv_r <= xrv_c;
            assign xrd = xrd_r;  assign xrv = xrv_r;
        end else begin : g_xoreg            // 출력 레지스터 생략 -> 코어 1단만
            assign xrd = xrd_c;  assign xrv = xrv_c;
        end
    endgenerate

    wire [Tile_M-1:0]    exp_ov_v;
    wire [EW-1:0]   exp_e_v [0:Tile_M-1];
    wire [Tile_M*EW-1:0] e_pack;
    generate
        for (gr = 0; gr < Tile_M; gr = gr + 1) begin : g_lane
            wire signed [DW-1:0] xr = xrd[gr*DW +: DW];   // BRAM 출력 (1 clk 지연)
            wire signed [DW-1:0] mr = rmax[xrb][gr];      // ES_EXP 동안 상수
            // 17비트 부호확장 후 뺄셈 : z <= 0 보장
            wire signed [ZW-1:0] zr = {xr[DW-1], xr} - {mr[DW-1], mr};

            exp2_unit #(.ZW(ZW), .ZF(IF), .EW(EW), .EF(EF)) u_exp (
                .clk(clk), .rst_n(rst_n),
                .in_valid(xrv), .z(zr),
                .out_valid(exp_ov_v[gr]), .e(exp_e_v[gr])
            );
            assign e_pack[gr*EW +: EW] = exp_e_v[gr];
        end
    endgenerate
    wire e_ov = exp_ov_v[0];                 // 전 lane lockstep

    // 역수 유닛 1개를 Tile_M회 파이프라인
    wire          rcp_iv = (estate == ES_RCP) && (qi < Tile_M);
    wire [SW-1:0] rcp_s  = sum[qi[MW-1:0]];
    wire          rcp_ov;
    wire [RW-1:0] rcp_r;
    wire [PW-1:0] rcp_p;
    recip_unit #(.SW(SW), .RW(RW), .RF(RF), .PW(PW)) u_rcp (
        .clk(clk), .rst_n(rst_n),
        .in_valid(rcp_iv), .s(rcp_s),
        .out_valid(rcp_ov), .r(rcp_r), .p(rcp_p)
    );

    assign exp_done = (estate == ES_RCP) && rcp_ov && (qc == Tile_M-1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            estate <= ES_IDLE;
            xrb <= 1'b0; ewb <= 1'b0;
            ei <= 0; ec <= 0; etc_ <= 0; qi <= 0; qc <= 0;
            for (i = 0; i < Tile_M; i = i + 1) sum[i] <= {SW{1'b0}};
            etlen[0] <= 0; etlen[1] <= 0;
        end else begin
            case (estate)
            // ---- xbuf 뱅크가 차고 ebuf 뱅크가 비면 시작
            ES_IDLE: if (xfull[xrb] && !efull[ewb]) begin
                etc_ <= tlen[xrb];
                ei <= 0; ec <= 0;
                for (i = 0; i < Tile_M; i = i + 1) sum[i] <= {SW{1'b0}};
                estate <= ES_EXP;
            end
            // ---- exp + 행별 sum 누산
            ES_EXP: begin
                if (ei < etc_) ei <= ei + 1'b1;              // 매 clk 한 열 투입
                if (e_ov) begin                               // 5 clk 뒤부터 수집
                    for (i = 0; i < Tile_M; i = i + 1)             // 행별 독립 누산
                        sum[i] <= sum[i] + {{(SW-EW){1'b0}}, exp_e_v[i]};
                    ec <= ec + 1'b1;
                    if (ec == etc_ - 1'b1) begin
                        estate <= ES_RCP; qi <= 0; qc <= 0;
                    end
                end
            end
            // ---- 행별 1/S (Tile_M개 파이프라인)
            ES_RCP: begin
                if (qi < Tile_M) qi <= qi + 1'b1;
                if (rcp_ov) begin
                    rr[ewb][qc[MW-1:0]] <= rcp_r;
                    pp[ewb][qc[MW-1:0]] <= rcp_p;
                    qc <= qc + 1'b1;
                    if (qc == Tile_M-1) begin                      // exp_done 펄스
                        etlen[ewb] <= etc_;
                        xrb    <= ~xrb;                       // xbuf 뱅크 넘김
                        ewb    <= ~ewb;                       // ebuf 뱅크 넘김
                        estate <= ES_IDLE;
                    end
                end
            end
            default: estate <= ES_IDLE;
            endcase
        end
    end

    // ======================================================================
    //  S3 : y = e * R  ->  출력            (ebuf[erb] -> out_col)
    // ======================================================================
    localparam MS_IDLE = 1'b0, MS_MUL = 1'b1;

    reg             mstate;
    reg  [TCW-1:0]  mi, mtc;

    wire [TW-1:0] m_rd  = (mi < mtc) ? mi[TW-1:0] : {TW{1'b0}};
    wire          m_iss = (mstate == MS_MUL) && (mi < mtc);

    // ==================================================================
    //  ebuf : simple dual-port BRAM   (write = S2,  등록형 read = S3)
    // ==================================================================
    reg [Tile_M*EW-1:0] ebuf [0:(2<<TW)-1];     // exp 결과
    reg [Tile_M*EW-1:0] ea_c;    // BRAM 코어 출력 레지스터 (= 읽기 1단째, 필수)
    always @(posedge clk) begin
        if (e_ov) ebuf[{ewb, ec[TW-1:0]}] <= e_pack; // write 포트
        ea_c <= ebuf[{erb, m_rd}];   // read 포트 : 주소 제시 -> 다음 clk 에 데이터
    end

    // 주소 제시 시점의 valid/last 를 BRAM_LAT 만큼 지연시켜 데이터와 정렬
    reg  va_c, la_c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin va_c <= 1'b0; la_c <= 1'b0; end
        else begin
            va_c <= m_iss;
            la_c <= m_iss && (mi == mtc - 1'b1);
        end
    end

    wire [Tile_M*EW-1:0] ea;
    wire                 va, la;
    generate
        if (BRAM_LAT >= 2) begin : g_eoreg  // 선택적 출력 레지스터 추가 -> 총 2단
            reg [Tile_M*EW-1:0] ea_r;
            reg                 va_r, la_r;
            always @(posedge clk) ea_r <= ea_c;
            always @(posedge clk or negedge rst_n)
                if (!rst_n) begin va_r <= 1'b0; la_r <= 1'b0; end
                else        begin va_r <= va_c;  la_r <= la_c;  end
            assign ea = ea_r;  assign va = va_r;  assign la = la_r;
        end else begin : g_eoreg            // 출력 레지스터 생략 -> 코어 1단만
            assign ea = ea_c;  assign va = va_c;  assign la = la_c;
        end
    endgenerate

    wire [Tile_M*OW-1:0] y_pack;
    generate
        for (gr = 0; gr < Tile_M; gr = gr + 1) begin : g_omul
            wire [EW-1:0]  e_r = ea[gr*EW +: EW];
            // e(17b) x R(18b) -> DSP48 1개.  R 과 p 는 행마다 다름.
            wire [PDW-1:0] prd = e_r * rr[erb][gr];
            wire [5:0]     shr = pp[erb][gr] + SH_OFF;
            wire [PDW-1:0] hlf = {{(PDW-1){1'b0}}, 1'b1} << (shr - 1);
            wire [PDW-1:0] ysh = (prd + hlf) >> shr;          // Q1.14 반올림
            assign y_pack[gr*OW +: OW] = ysh[OW-1:0];
        end
    endgenerate

    assign mul_done = (mstate == MS_MUL) && va && la;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mstate    <= MS_IDLE;
            erb       <= 1'b0;
            mi <= 0; mtc <= 0;
            out_valid <= 1'b0;
            out_col   <= {(Tile_M*OW){1'b0}};
            out_last  <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            out_last  <= 1'b0;

            case (mstate)
            MS_IDLE: if (efull[erb]) begin
                mtc    <= etlen[erb];
                mi     <= 0;
                mstate <= MS_MUL;
            end
            MS_MUL: begin
                if (m_iss) mi <= mi + 1'b1;                   // Stage A : ebuf 주소 제시
                if (va) begin                                 // Stage B : 곱셈 출력
                    out_valid <= 1'b1;
                    out_col   <= y_pack;
                    out_last  <= la;
                    if (la) begin                             // mul_done 펄스
                        erb    <= ~erb;
                        mstate <= MS_IDLE;
                    end
                end
            end
            endcase
        end
    end

    // ---- 파라미터 정합성 체크 (합성 무관) ---------------------------------
    initial begin
        if (SH_OFF < 1)   $display("ERROR: softmax_top RF(%0d) must be > OF(%0d)", RF, OF);
        if (SW < EF + TW) $display("ERROR: softmax_top SW too small");
        if (EW != EF+1)   $display("ERROR: softmax_top EW must be EF+1");
        if (RW != RF+1)   $display("ERROR: softmax_top RW must be RF+1");
    end
endmodule
