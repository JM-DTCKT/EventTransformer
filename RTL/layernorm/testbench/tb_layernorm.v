// ============================================================================
//  tb_layernorm.v  --  32행 Tile D축 LayerNorm 검증 TB (BF16 in / signed Q4.11 out)
// ----------------------------------------------------------------------------
//  골든 레퍼런스를 외부 파일 없이 TB 안에서 real 연산으로 직접 계산한다.
//  BF16 입력은 $shortrealtobits 로 만들고, 골든은 $bitstoshortreal 로 되읽은
//  **정확한 BF16 값**에 대해 double 정밀도로 LayerNorm 을 돌린 값이다.
//  따라서 측정되는 오차 = 고정소수점 파이프라인이 만든 오차 전부.
//
//   PART A  bf16_to_fix 단독 : 지수/가수 전수 스윕 + 포화/언더플로/부호 대칭 + xsh
//   PART B  rsqrt_unit  단독 : v = eps..2^60 로그스윕 + 랜덤, 짝/홀 지수 양쪽
//   PART C  지정 Tile : 상수행 / ramp / one-hot / 큰평균+작은분산 / 대스케일 / 부호혼합
//   PART D  랜덤 Tile : (평균, 표준편차) 조합 6종
//   PART E  프로토콜  : in_valid gap 비트일치, 동작중 리셋, 3 Tile 중첩(슬롯 독립성)
//   PART F  타이밍    : Tile 처리주기 실측 + 96x128 end-to-end 실측
//   PART G  스케일 민감도 : sigma 를 바꿔가며 오차 측정, in_shift 로 창 이동 효과
//
//  Tile 마다 32행 전부에 대해 확인
//   - 원소별 절대/RMS 오차 vs 이상적 LayerNorm
//   - 행별 sum(y) ~= 0, sum(y^2)/D ~= 1  (정규화가 실제로 됐는가)
//   - |y| <= sqrt(D-1) = 11.27 이므로 **Q4.11(+-16) 은 절대 포화하지 않는다**
//   - 결정성(같은 입력 -> 같은 출력) / 단조성(같은 행 안에서 x 순서 = y 순서)
//   - out_last 위치
// ============================================================================
`timescale 1ns/1ps

module tb_layernorm;

    // ---------------- 파라미터 (DUT 와 일치) ----------------
    localparam integer LANE = 32;
    localparam integer D    = 128;
    localparam integer DLOG = 7;
    localparam integer IW   = 24;
    localparam integer IF   = 15;
    localparam integer OW   = 16;
    localparam integer OF   = 11;
    localparam integer RW   = 18;
    localparam integer RF   = 17;
    localparam integer DSW  = 25;
    localparam integer DSF  = 18;
    localparam integer XSW  = 6;
    localparam integer EWD  = 6;
    localparam integer QW   = 6;
    localparam integer SRAM_LAT = 1;
    localparam integer NB   = 3;
    localparam integer EPS_INT  = 175921860;
    localparam integer MAXSHL   = 5;

    localparam integer SXW = IW + DLOG;             // 31
    localparam integer SQW = 2*(IW-1) + DLOG + 1;   // 54
    localparam integer VW  = SQW + DLOG;            // 61
    localparam integer VF  = 2*IF + 2*DLOG;         // 44

    localparam real LSB_IN  = 2.0 ** (-IF);         // 3.052e-5
    localparam real LSB_OUT = 2.0 ** (-OF);         // 4.883e-4
    localparam real EPS_R   = EPS_INT / (2.0 ** VF);
    localparam real YMAX_TH = 11.2694;              // sqrt(D-1), LN 출력의 이론 상한

    // 기대 사이클 (RTL 구조식) — PART F 에서 실측과 대조
    localparam integer S1_CLK = D;                       // 수신 (열 D개)
    localparam integer S2_CLK = 1 + 2 + LANE + 3;        // idle1 + var전단2 + 행32 + rsqrt3
    localparam integer S3_CLK = D + SRAM_LAT + 3;        // 열 D개 + SRAM + B1/B2/C
    localparam integer EXP_PER = (S1_CLK > S2_CLK)
                               ? ((S1_CLK > S3_CLK) ? S1_CLK : S3_CLK)
                               : ((S2_CLK > S3_CLK) ? S2_CLK : S3_CLK);
    // Tile n개 end-to-end : 첫 in_valid -> 마지막 out_last **경과 clk**
    // (softmax 유닛과 같은 규약 : t_last_out - t_first_in, +1 안 함)
    localparam integer E2E_3  = S1_CLK + S2_CLK + S3_CLK + 2*EXP_PER;

    // 오차 허용치
    localparam real TOL_ELEM = 7.0e-4;    // 1.43 LSB of Q4.11
    localparam real TOL_RMS  = 2.5e-4;    // 0.51 LSB
    localparam real TOL_MEAN = 2.0e-3;    // |mean(y)|
    localparam real TOL_STD  = 3.0e-3;    // |rms(y) - 1|
    localparam real TOL_CVT  = 0.51;      // bf16_to_fix : 0.5 LSB + 여유
    localparam real TOL_RSQ  = 4.0e-5;    // rsqrt_unit 상대오차

    reg clk = 1'b0;
    reg rst_n;
    always #1 clk = ~clk;            // 2ns (500MHz)

    integer errors = 0, checks = 0;

    task pass_fail;
        input string item;
        input        ok;
        begin
            checks = checks + 1;
            if (!ok) errors = errors + 1;
            $display("   %-52s : %s", item, ok ? "PASS" : "*** FAIL ***");
        end
    endtask

    // ======================================================================
    //  BF16 <-> real 변환 헬퍼
    // ======================================================================
    function [15:0] r2bf;                 // real -> BF16 (round-to-nearest-even)
        input real v;
        shortreal sv;
        reg [31:0] f;
        begin
            sv  = v;
            f   = $shortrealtobits(sv);
            if ((f[15:0] > 16'h8000) || ((f[15:0] == 16'h8000) && f[16]))
                r2bf = f[31:16] + 16'd1;
            else
                r2bf = f[31:16];
        end
    endfunction

    function real bf2r;                   // BF16 -> real (무손실)
        input [15:0] b;
        begin bf2r = $bitstoshortreal({b, 16'h0000}); end
    endfunction

    function real u2r;                    // 넓은 unsigned -> real
        input [VW-1:0] v;
        integer k; real acc;
        begin
            acc = 0.0;
            for (k = VW-1; k >= 0; k = k - 1) acc = acc*2.0 + (v[k] ? 1.0 : 0.0);
            u2r = acc;
        end
    endfunction

    // ================= DUT #1 : bf16_to_fix =================
    reg  [15:0]           cv_bf;
    reg  signed [XSW-1:0] cv_sh;
    wire signed [IW-1:0]  cv_x;
    wire                  cv_ovf;
    bf16_to_fix #(.IW(IW), .IF(IF), .XSW(XSW)) u_cvt_dut (
        .bf(cv_bf), .xsh(cv_sh), .x(cv_x), .ovf(cv_ovf));

    // ================= DUT #2 : rsqrt_unit =================
    reg                   rq_iv;
    reg  [VW-1:0]         rq_v;
    wire                  rq_ov;
    wire [RW-1:0]         rq_r;
    wire signed [EWD-1:0] rq_e;
    rsqrt_unit #(.VW(VW), .VF(VF), .RW(RW), .RF(RF), .QW(QW), .EW(EWD)) u_rsq_dut (
        .clk(clk), .rst_n(rst_n), .in_valid(rq_iv), .v(rq_v),
        .out_valid(rq_ov), .r(rq_r), .e(rq_e));

    // ================= DUT #3 : layernorm_top (메인) =================
    reg                   ln_iv;
    reg  [LANE*16-1:0]    ln_col;
    reg  signed [XSW-1:0] ln_sh;
    wire                  ln_ir, ln_ov, ln_ol, ln_ovf;
    wire [LANE*OW-1:0]    ln_oc;
    layernorm_top #(.LANE(LANE), .D(D), .DLOG(DLOG), .IW(IW), .IF(IF),
                    .OW(OW), .OF(OF), .RW(RW), .RF(RF), .DSW(DSW), .DSF(DSF),
                    .XSW(XSW), .EPS_INT(EPS_INT), .MAXSHL(MAXSHL),
                    .SRAM_LAT(SRAM_LAT), .NB(NB)) u_ln (
        .clk(clk), .rst_n(rst_n),
        .in_valid(ln_iv), .in_ready(ln_ir), .in_col(ln_col), .in_shift(ln_sh),
        .out_valid(ln_ov), .out_col(ln_oc), .out_last(ln_ol), .ovf(ln_ovf));

    localparam [LANE*16-1:0] GARBAGE = {(LANE*16/32){32'hDEAD_BEEF}};

    // ---------------- 데이터 / 골든 (Tile 3개분) ----------------
    reg  [15:0]   xb [0:2][0:LANE-1][0:D-1];    // 입력 BF16
    real          xv [0:2][0:LANE-1][0:D-1];    // 그 BF16 의 정확한 실수값
    reg  [OW-1:0] ym [0:2][0:LANE-1][0:D-1];    // DUT 출력
    reg  [OW-1:0] yr [0:2][0:LANE-1][0:D-1];    // 기준 출력 (gap 비교용)
    real          gm [0:2][0:LANE-1][0:D-1];    // 이상적 float LayerNorm
    real          gvar [0:2][0:LANE-1];         // 행별 참 분산 (eps 지배 판정용)

    integer ocol, ocap, last_cnt, last_at, oovf;
    integer cyc, t_lastprev, per_min, per_max, per_cnt;
    integer t_first_in, t_last_out;
    reg     e2e_en;
    integer g_nelem, ir_low;
    real    g_maxerr, g_sqsum;

    initial begin
        ocol = 0; ocap = 0; last_cnt = 0; last_at = -1; oovf = 0; cyc = 0;
        t_lastprev = -1; per_min = 1000000; per_max = -1; per_cnt = 0;
        t_first_in = -1; t_last_out = -1; e2e_en = 1'b0;
        g_nelem = 0; g_maxerr = 0.0; g_sqsum = 0.0;
    end

    // ---------------- 출력 캡처 ----------------
    integer cr;
    always @(posedge clk) begin
        if (rst_n) begin
            cyc <= cyc + 1;
            if (!ln_ir) ir_low = ir_low + 1;
            if (e2e_en && ln_iv && ln_ir && t_first_in < 0) t_first_in <= cyc;
            if (e2e_en && ln_ov && ln_ol)                   t_last_out <= cyc;
            if (ln_ov) begin
                if (ocol < D)
                    for (cr = 0; cr < LANE; cr = cr + 1)
                        ym[ocap][cr][ocol] = ln_oc[cr*OW +: OW];
                else oovf = oovf + 1;
                ocol = ocol + 1;
                if (ln_ol) begin
                    last_cnt = last_cnt + 1; last_at = ocol - 1;
                    if (t_lastprev >= 0) begin
                        if (cyc - t_lastprev < per_min) per_min = cyc - t_lastprev;
                        if (cyc - t_lastprev > per_max) per_max = cyc - t_lastprev;
                        per_cnt = per_cnt + 1;
                    end
                    t_lastprev = cyc;
                    ocol = 0;
                    ocap = (ocap == 2) ? 0 : ocap + 1;
                end
            end
        end
    end

    // ---------------- 골든 : 행별 float LayerNorm ----------------
    //  eps_r 을 인자로 받는다 — in_shift 로 창을 2^k 옮기면 내부 var 가 2^2k 배가
    //  되므로 **실효 eps 는 2^-2k 배**가 된다 (eps 는 고정소수점 상수라 같이 안 커진다).
    real eps_use;
    task calc_gold;
        input integer ti;
        integer r, d; real mu, s2, dv;
        begin
            for (r = 0; r < LANE; r = r + 1) begin
                mu = 0.0;
                for (d = 0; d < D; d = d + 1) mu = mu + xv[ti][r][d];
                mu = mu / D;
                s2 = 0.0;                            // 2-pass : 수치적으로 안전
                for (d = 0; d < D; d = d + 1) begin
                    dv = xv[ti][r][d] - mu; s2 = s2 + dv*dv;
                end
                s2 = s2 / D;
                gvar[ti][r] = s2;
                for (d = 0; d < D; d = d + 1)
                    gm[ti][r][d] = (xv[ti][r][d] - mu) / $sqrt(s2 + eps_use);
            end
        end
    endtask

    // ---------------- 데이터 셋업 헬퍼 ----------------
    task set_elem;                       // 실수값 -> BF16 저장 + 정확값 되읽기
        input integer ti, r, d;
        input real    v;
        begin
            xb[ti][r][d] = r2bf(v);
            xv[ti][r][d] = bf2r(xb[ti][r][d]);
        end
    endtask

    // ---------------- Tile 인가 ----------------
    //  gseed != 0 이면 열 사이에 랜덤 gap + 무효구간 쓰레기값
    task drive_tile;
        input integer ti;
        input signed [XSW-1:0] sh;
        inout integer gseed;
        integer d, r, g;
        begin
            @(negedge clk);
            while (!ln_ir) @(negedge clk);
            for (d = 0; d < D; d = d + 1) begin
                if (gseed != 0) begin
                    g = $random(gseed); if (g < 0) g = -g;
                    repeat (g % 3) begin
                        ln_iv = 1'b0; ln_col = GARBAGE; ln_sh = $random(gseed);
                        @(negedge clk);
                    end
                end
                for (r = 0; r < LANE; r = r + 1) ln_col[r*16 +: 16] = xb[ti][r][d];
                ln_iv = 1'b1;
                ln_sh = sh;
                @(negedge clk);
                while (!ln_ir) begin              // 슬롯이 없으면 대기
                    ln_iv = 1'b0; @(negedge clk);
                end
            end
            ln_iv = 1'b0; ln_col = GARBAGE;
        end
    endtask

    // ---------------- Tile 검증 (32행 전부) ----------------
    task check_tile;
        input string  tname;
        input integer ti;
        input real    tol_elem;
        input         verbose;
        integer r, d, u, mono_bad, det_bad, sat_bad, n_pair, seedl;
        real    yd, ye, mx, rms, sy, sy2, mean_worst, std_worst, ymax;
        begin
            calc_gold(ti);
            mx = 0.0; rms = 0.0; mono_bad = 0; det_bad = 0; sat_bad = 0;
            mean_worst = 0.0; std_worst = 0.0; ymax = 0.0; seedl = 12345;

            for (r = 0; r < LANE; r = r + 1) begin
                sy = 0.0; sy2 = 0.0;
                for (d = 0; d < D; d = d + 1) begin
                    yd = $signed(ym[ti][r][d]) * LSB_OUT;
                    ye = yd - gm[ti][r][d]; if (ye < 0.0) ye = -ye;
                    if (ye > mx) mx = ye;
                    rms = rms + ye*ye;
                    sy  = sy + yd;  sy2 = sy2 + yd*yd;
                    if (yd > ymax)  ymax = yd;
                    if (-yd > ymax) ymax = -yd;
                    // LN 출력의 이론 상한 sqrt(D-1) -> Q4.11 포화가 있으면 안 된다
                    if (yd >= 15.9 || yd <= -15.9) sat_bad = sat_bad + 1;
                    if (ye > g_maxerr) g_maxerr = ye;
                    g_sqsum = g_sqsum + ye*ye;
                    g_nelem = g_nelem + 1;
                end
                sy = sy / D;  sy2 = $sqrt(sy2 / D);
                if (sy  > mean_worst) mean_worst = sy;
                if (-sy > mean_worst) mean_worst = -sy;
                // rms(y)=1 은 eps 를 무시할 수 있을 때만 성립한다
                // (var <~ eps 인 행은 출력이 의도적으로 눌린다 — 상수행이면 정확히 0)
                if (gvar[ti][r] > 100.0*eps_use) begin
                    if (sy2 - 1.0 >  std_worst) std_worst = sy2 - 1.0;
                    if (1.0 - sy2 >  std_worst) std_worst = 1.0 - sy2;
                end
                // 단조성 / 결정성 : 같은 행 안에서 x 순서가 y 순서로 보존돼야 한다
                for (n_pair = 0; n_pair < 400; n_pair = n_pair + 1) begin
                    d = $random(seedl); if (d < 0) d = -d; d = d % D;
                    u = $random(seedl); if (u < 0) u = -u; u = u % D;
                    if ((xv[ti][r][d] > xv[ti][r][u]) &&
                        ($signed(ym[ti][r][d]) < $signed(ym[ti][r][u])))
                        mono_bad = mono_bad + 1;
                    if ((xb[ti][r][d] === xb[ti][r][u]) &&
                        (ym[ti][r][d] !== ym[ti][r][u]))
                        det_bad = det_bad + 1;
                end
            end
            rms = $sqrt(rms / (LANE * D));

            if (verbose || mx > tol_elem || rms > TOL_RMS || mono_bad != 0
                        || det_bad != 0 || sat_bad != 0 || last_at != D-1 || oovf != 0)
                $display("  [%-22s] maxerr=%.3e (%5.2f LSB) rms=%.3e |mean|=%.1e |rms-1|=%.1e max|y|=%5.2f%s%s%s",
                         tname, mx, mx/LSB_OUT, rms, mean_worst, std_worst, ymax,
                         (mono_bad != 0) ? "  MONO-VIOL" : "",
                         (det_bad  != 0) ? "  DET-VIOL"  : "",
                         (sat_bad  != 0) ? "  SAT-VIOL"  : "");

            if (mx > tol_elem) begin
                $display("   *** FAIL %0s : max err %.3e > %.3e", tname, mx, tol_elem);
                errors = errors + 1; end
            if (rms > TOL_RMS) begin
                $display("   *** FAIL %0s : rms %.3e > %.3e", tname, rms, TOL_RMS);
                errors = errors + 1; end
            if (mean_worst > TOL_MEAN) begin
                $display("   *** FAIL %0s : |mean(y)| = %.3e", tname, mean_worst);
                errors = errors + 1; end
            if (std_worst > TOL_STD) begin
                $display("   *** FAIL %0s : |rms(y)-1| = %.3e", tname, std_worst);
                errors = errors + 1; end
            if (mono_bad != 0) begin
                $display("   *** FAIL %0s : monotonicity %0d", tname, mono_bad);
                errors = errors + 1; end
            if (det_bad != 0) begin
                $display("   *** FAIL %0s : determinism %0d", tname, det_bad);
                errors = errors + 1; end
            if (sat_bad != 0) begin
                $display("   *** FAIL %0s : Q4.11 포화 %0d개 (이론상 |y|<=%.2f)",
                         tname, sat_bad, YMAX_TH);
                errors = errors + 1; end
            if (last_at != D-1) begin
                $display("   *** FAIL %0s : out_last at %0d (expect %0d)",
                         tname, last_at, D-1);
                errors = errors + 1; end
            if (oovf != 0) begin
                $display("   *** FAIL %0s : 초과 출력 %0d", tname, oovf);
                errors = errors + 1; oovf = 0; end
        end
    endtask

    integer zs;
    task run_tile;                       // 인가 -> 출력 대기 -> 검증
        input string  tname;
        input real    tol_elem;
        input         verbose;
        integer lc0;
        begin
            zs = 0; ocap = 0; lc0 = last_cnt;
            drive_tile(0, 6'sd0, zs);
            wait (last_cnt == lc0 + 1);
            @(negedge clk);
            check_tile(tname, 0, tol_elem, verbose);
        end
    endtask

    // ---------------- 랜덤 가우시안 (Box-Muller) ----------------
    integer gseed_g;
    function real randn;
        input integer dummy;
        real u1, u2;
        begin
            u1 = ($random(gseed_g) & 32'h7FFFFFFF) / 2147483647.0;
            u2 = ($random(gseed_g) & 32'h7FFFFFFF) / 2147483647.0;
            if (u1 < 1.0e-12) u1 = 1.0e-12;
            randn = $sqrt(-2.0*$ln(u1)) * $cos(6.283185307179586*u2);
        end
    endfunction

    task fill_random;                    // 행마다 다른 (평균, 표준편차)
        input integer ti;
        input real    mu_sc, sd_sc;
        integer r, d; real mu, sd;
        begin
            for (r = 0; r < LANE; r = r + 1) begin
                mu = mu_sc * randn(0);
                sd = sd_sc * (0.3 + 1.4*(($random(gseed_g)&32'h7FFFFFFF)/2147483647.0));
                for (d = 0; d < D; d = d + 1)
                    set_elem(ti, r, d, mu + sd*randn(0));
            end
        end
    endtask

    // ======================================================================
    //  비트정확 모델 — DUT 와 **독립적으로** 의도한 정수연산을 재구현한다.
    //  float 골든(허용오차 비교)이 못 잡는 폭/반올림/포화/파이프라인 버그를
    //  비트 단위로 잡기 위한 것.  BF16 변환은 RTL 의 시프트식이 아니라
    //  IEEE 실수연산 -> 반올림 경로로 따로 구현해 알고리즘까지 교차검증한다.
    // ======================================================================
    localparam integer SEGB  = 7;
    localparam integer FRB   = 12;
    localparam integer MUF_  = IF + DLOG;                    // 22
    localparam integer DDW_  = SXW + 2;                      // 33
    localparam integer DQW_  = DDW_ + MAXSHL;                // 38
    localparam integer PDW_  = DSW + RW + 1;                 // 44
    localparam integer YSH_  = DSF + RF - OF;                // 24
    localparam integer MAXU_ = ((VW-1-VF)/2) + (MUF_ - DSF) + MAXSHL;   // 17
    localparam integer XMAXM = (1 << (IW-1)) - 1;            // 2^23-1
    localparam integer DSMAX = (1 << (DSW-1)) - 1;           // 2^24-1
    localparam integer YMAXI = (1 << (OW-1)) - 1;            // 32767

    `include "rsqrt_lut.vh"

    reg [OW-1:0] mY [0:LANE-1][0:D-1];      // 모델 출력
    real         mvar [0:LANE-1];           // 모델이 본 var (정보 표시용)

    // BF16 -> Q8.15 : RTL 과 다른 경로(실수연산 + 반올림 + 포화)로 구현
    function signed [IW-1:0] m_cvt;
        input [15:0] b;
        input integer xs;
        real v, sc2; integer mag;
        begin
            if (b[14:7] == 8'd255) mag = XMAXM;              // Inf / NaN
            else begin
                v = bf2r(b); if (v < 0.0) v = -v;
                sc2 = v * (2.0 ** IF) * (2.0 ** xs);
                if (sc2 >= 8388608.0) mag = XMAXM;           // |x| >= 2^8 -> 포화
                else                  mag = $rtoi(sc2 + 0.5);// round-half-up
            end
            m_cvt = b[15] ? -mag : mag;
        end
    endfunction

    integer m_umin, m_umax, m_qmin, m_qmax;
    reg     dbg_model = 1'b0;

    task model_tile;
        input integer ti;
        input integer xs;
        integer r, d, k, q, par, ev, uu, idx, seg, frac;
        integer basei, mulr, interp;
        reg signed [IW-1:0]   X   [0:D-1];
        reg signed [SXW-1:0]  SXm;
        reg [SQW-1:0]         SQm;
        reg [VW-1:0]          SXsq, Vm, MQ;
        reg signed [RF+2:0]   Rs;
        reg signed [DDW_-1:0] dd;
        reg signed [DQW_-1:0] dq, half, dsr;
        reg signed [DSW-1:0]  dsv;
        reg signed [PDW_-1:0] prd, yv;
        reg [SQW-1:0]         xsq_t;
        reg [VW-1:0]          sq7m;
        begin
            for (r = 0; r < LANE; r = r + 1) begin
                // ---- S1 : 변환 + 누산 (정확) ----
                SXm = 0; SQm = 0;
                for (d = 0; d < D; d = d + 1) begin
                    X[d] = m_cvt(xb[ti][r][d], xs);
                    SXm  = SXm + X[d];
                    // X*X 는 **반드시 따로** 낸다 — unsigned 누산기와 한 식에 섞으면
                    // Verilog 가 피연산자를 전부 unsigned 로 바꿔 음수 X 를 망친다
                    xsq_t = X[d]*X[d];
                    SQm   = SQm + xsq_t;
                end
                // ---- S2 : V = (SQ<<DLOG) - SX^2 + eps ----
                //  SX^2 은 **부호곱을 따로** 낸 뒤 unsigned 로 넘긴다
                //  (unsigned 식에 signed 를 섞으면 Verilog 가 zero-extend 해버린다)
                SXsq = SXm * SXm;
                sq7m = {SQm, {DLOG{1'b0}}};          // SQ << DLOG (명시적 61b)
                Vm   = sq7m - SXsq + EPS_INT;
                mvar[r] = u2r(Vm) * (2.0 ** (-VF));
                // 정수 분산식이 실제 분산과 같은지 (상쇄오차가 정말 0인지) 직접 확인
                if (dbg_model && r < 3)
                    $display("      정수 var = %.9e   float var = %.9e   비 = %.9f",
                             mvar[r], gvar[ti][r]+eps_use, mvar[r]/(gvar[ti][r]+eps_use));
                q = 0;
                for (k = 0; k < VW; k = k + 1) if (Vm[k]) q = k;
                if (q < m_qmin) m_qmin = q;
                if (q > m_qmax) m_qmax = q;
                MQ   = Vm << ((VW-1) - q);
                seg  = MQ[VW-2 -: SEGB];
                frac = MQ[VW-2-SEGB -: FRB];
                par  = q & 1;
                ev   = (q >= VF) ? ((q - VF)/2) : -((VF - q + 1)/2);   // floor(k/2)
                idx  = par*(1<<SEGB) + seg;
                // ** Verilog 부호 전파 함정 **
                //   base(unsigned) 를 같은 식에 넣으면 식 전체가 unsigned 로 바뀌어
                //   delta(음수)가 zero-extend 되고 >>> 도 논리시프트가 된다.
                //   단계마다 signed integer 로 끊어서 계산한다 (RTL 도 같은 이유로
                //   mul_r / interp / r_s 를 별도 signed wire 로 나눠 두었다).
                basei  = rsq_base_rom[idx];
                mulr   = rsq_delta_rom[idx] * frac;
                interp = (mulr + (1 << (FRB-1))) >>> FRB;
                Rs     = basei + interp;
                // ---- S3 : d -> ds -> y ----
                uu = ev + (MUF_ - DSF) + MAXSHL;
                if (uu < 0)      uu = 0;
                if (uu > MAXU_)  uu = MAXU_;
                if (uu < m_umin) m_umin = uu;
                if (uu > m_umax) m_umax = uu;
                for (d = 0; d < D; d = d + 1) begin
                    dd = {{(DDW_-IW-DLOG){X[d][IW-1]}}, X[d], {DLOG{1'b0}}}
                       - {{(DDW_-SXW){SXm[SXW-1]}}, SXm};
                    dq = dd <<< MAXSHL;
                    if (uu == 0) half = 0; else half = (1 <<< (uu-1));
                    dsr = (dq + half) >>> uu;
                    if      (dsr >  DSMAX) dsv =  DSMAX;
                    else if (dsr < -DSMAX) dsv = -DSMAX;
                    else                   dsv =  dsr;
                    prd = dsv * $signed({1'b0, Rs[RW-1:0]});
                    yv  = (prd + (1 <<< (YSH_-1))) >>> YSH_;
                    if      (yv >  YMAXI)     mY[r][d] =  YMAXI;
                    else if (yv < -YMAXI-1)   mY[r][d] = -YMAXI-1;
                    else                      mY[r][d] =  yv[OW-1:0];
                end
            end
        end
    endtask

    integer mism;
    task check_model;                       // DUT vs 비트정확 모델 (비트 단위)
        input string  tname;
        input integer ti;
        input integer xs;
        integer r, d;
        begin
            model_tile(ti, xs);
            mism = 0;
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1)
                if (ym[ti][r][d] !== mY[r][d]) begin
                    if (mism < 4)
                        $display("   *** MISMATCH %0s r=%0d d=%0d dut=%04h model=%04h",
                                 tname, r, d, ym[ti][r][d], mY[r][d]);
                    mism = mism + 1;
                end
            pass_fail({"H3 비트정확 모델 일치 : ", tname}, mism == 0);
        end
    endtask

    // ======================================================================
    //  내부 감시 — 시프트량 u 가 설계상 범위 [0, MAXU] 를 벗어나지 않는지,
    //  포화 경로가 실제로 서는지 계층참조로 직접 본다 (설계 가정의 직접 검증).
    // ======================================================================
    integer h_umin, h_umax, h_esat, h_ysat, h_uclamp;
    initial begin h_umin = 999; h_umax = -999; h_esat = 0; h_ysat = 0; h_uclamp = 0; end

    genvar mg;
    generate
        for (mg = 0; mg < LANE; mg = mg + 1) begin : g_mon
            integer uh, sh_h;
            always @(posedge clk) if (rst_n && u_ln.norm_v2) begin
                uh   = u_ln.g_norm[mg].u;              // unsigned wire -> signed integer
                sh_h = $signed(u_ln.g_norm[mg].u_raw); // 클램프 전 원래 시프트량
                if (uh < h_umin) h_umin = uh;
                if (uh > h_umax) h_umax = uh;
                if (sh_h < 0 || sh_h > MAXU_) h_uclamp = h_uclamp + 1;
                if (u_ln.sat_ds[mg]) h_esat = h_esat + 1;
                if (u_ln.sat_y[mg])  h_ysat = h_ysat + 1;
            end
        end
    endgenerate

    // ======================================================================
    //  2슬롯 자원(sum_x2 / rstd_man,rstd_exp) 충돌 감시 + 여유 마진 실측
    //    같은 물리 슬롯을 S2 가 쓰기 시작하는 시점과 이전 Tile 이 마지막으로
    //    읽은 시점의 간격을 재서, 2벌로 줄인 게 얼마나 여유 있는지 본다.
    // ======================================================================
    integer sq_clash, re_clash, sq_marg, re_marg;
    integer tiles_in, stall_first_tile, marg_at;   // 수신 완료 Tile 수 / 첫 stall Tile
    integer sq_lastrd [0:1];
    integer re_lastrd [0:1];
    initial begin
        sq_clash = 0; re_clash = 0; sq_marg = 999999; re_marg = 999999;
        tiles_in = 0; stall_first_tile = -1; marg_at = -1;
        sq_lastrd[0] = -1; sq_lastrd[1] = -1;
        re_lastrd[0] = -1; re_lastrd[1] = -1;
    end

    always @(posedge clk) if (rst_n) begin
        if (u_ln.recv_done) tiles_in = tiles_in + 1;
        if (!ln_ir && stall_first_tile < 0) stall_first_tile = tiles_in;
        // ---- sum_x2 : S1 이 쓰고(x2_wslot) S2 가 읽는다(x2_rslot) ----
        if (u_ln.stat_issue) sq_lastrd[u_ln.x2_rslot] = cyc;
        if (u_ln.recv_hs && u_ln.recv_first) begin           // 그 슬롯을 새로 덮어쓰는 순간
            if (sq_lastrd[u_ln.x2_wslot] >= 0 &&
                (cyc - sq_lastrd[u_ln.x2_wslot]) < sq_marg) sq_marg = cyc - sq_lastrd[u_ln.x2_wslot];
        end
        if (u_ln.recv_hs && u_ln.stat_issue && (u_ln.x2_wslot == u_ln.x2_rslot))
            sq_clash = sq_clash + 1;

        // ---- rstd_man/exp : S2 가 쓰고(rstd_wslot) S3 가 읽는다(rstd_rslot) ----
        if (u_ln.norm_v2) re_lastrd[u_ln.rstd_rslot] = cyc;
        if (u_ln.rsq_vld && (u_ln.stat_row_c == 0)) begin              // 그 슬롯의 첫 쓰기
            if (re_lastrd[u_ln.rstd_wslot] >= 0 &&
                (cyc - re_lastrd[u_ln.rstd_wslot]) < re_marg) begin
                re_marg = cyc - re_lastrd[u_ln.rstd_wslot];
                marg_at = tiles_in;
            end
        end
        if (u_ln.rsq_vld && u_ln.norm_v2 && (u_ln.rstd_wslot == u_ln.rstd_rslot))
            re_clash = re_clash + 1;
    end

    // ========================================================================
    //  메인
    // ========================================================================
    integer i, j, k, r, d, ti, ei, mi_, n_pts, p_bad, bad, lc0, fd_i, fd_o;
    real    v_ref, v_hw, dif, cmax, cvt_max, rmax_e, rel, sc;
    reg [15:0] bb;
    reg [VW-1:0] vv;

    initial begin
        gseed_g = 32'h1234_5678;
        eps_use = EPS_R;
        m_umin = 999; m_umax = -999; m_qmin = 999; m_qmax = -999;
        ln_iv = 0; ln_col = GARBAGE; ln_sh = 0;
        cv_bf = 0; cv_sh = 0; rq_iv = 0; rq_v = 0;
        rst_n = 0; repeat (8) @(negedge clk);
        rst_n = 1; repeat (4) @(negedge clk);

        $display("");
        $display("############################################################");
        $display("#  32-row Tile  D-axis LayerNorm   (BF16 in / signed Q4.11 out)");
        $display("#  LANE=%0d  D=%0d   입출력 = 열 단위 %0d원소 (%0db / %0db)",
                 LANE, D, LANE, LANE*16, LANE*OW);
        $display("#  내부 : X=Q%0d.%0d(%0db)  SX=%0db  SQ=%0db  V=%0db  r=UQ1.%0d(%0db)",
                 IW-1-IF, IF, IW, SXW, SQW, VW, RF, RW);
        $display("#  출력 : signed Q%0d.%0d (%0db, 1.0=16'h%0h, LSB=%.3e)",
                 OW-1-OF, OF, OW, (1<<OF), LSB_OUT);
        $display("#  eps  = %.4e (= %0d * 2^-%0d)", EPS_R, EPS_INT, VF);
        $display("#  SRAM read latency = %0d clk", SRAM_LAT);
        $display("############################################################");

        // ================= PART A : bf16_to_fix =================
        $display("");
        $display("== PART A : bf16_to_fix  (BF16 -> signed Q%0d.%0d) ==", IW-1-IF, IF);
        cmax = 0.0; n_pts = 0; bad = 0;
        // 지수 전수(1..254) x 가수 8종, 부호 양쪽
        for (ei = 1; ei <= 254; ei = ei + 1)
        for (i = 0; i < 8; i = i + 1)
        for (j = 0; j < 2; j = j + 1) begin
            bb = {j[0], ei[7:0], (i*16)+i};
            cv_bf = bb; cv_sh = 0; #0.1;
            v_ref = bf2r(bb) * (2.0 ** IF);
            if (v_ref >  8388607.0) v_ref =  8388607.0;    // 대칭 포화
            if (v_ref < -8388607.0) v_ref = -8388607.0;
            v_hw = $itor($signed(cv_x));
            dif  = v_hw - v_ref; if (dif < 0.0) dif = -dif;
            if (dif > cmax) cmax = dif;
            n_pts = n_pts + 1;
        end
        cvt_max = cmax;
        $display("   %0d점 스윕(지수 1..254 x 가수 8 x 부호 2), max err = %.3f LSB",
                 n_pts, cmax);
        pass_fail("bf16_to_fix 전 지수영역 반올림 오차 <= 0.5 LSB", cmax <= TOL_CVT);

        // 경계값
        cv_bf = 16'h0000; cv_sh = 0; #0.1;
        pass_fail("bf16_to_fix  +0.0 -> 0", (cv_x === 0) && (cv_ovf === 1'b0));
        cv_bf = 16'h8000; #0.1;
        pass_fail("bf16_to_fix  -0.0 -> 0", (cv_x === 0));
        cv_bf = r2bf(1.0); #0.1;
        pass_fail("bf16_to_fix   1.0 -> 2^15 정확", cv_x === (1 << IF));
        cv_bf = r2bf(-1.0); #0.1;
        pass_fail("bf16_to_fix  -1.0 -> -2^15 정확", cv_x === -(1 << IF));
        cv_bf = r2bf(255.0); #0.1;
        pass_fail("bf16_to_fix 255.0 포화 아님", (cv_ovf === 1'b0) &&
                  (cv_x === $signed(255*(1<<IF))));
        cv_bf = r2bf(256.0); #0.1;
        pass_fail("bf16_to_fix 256.0 -> 포화 + ovf", (cv_ovf === 1'b1) &&
                  (cv_x === $signed((1<<(IW-1))-1)));
        cv_bf = r2bf(-256.0); #0.1;
        pass_fail("bf16_to_fix -256.0 -> 대칭 포화", (cv_ovf === 1'b1) &&
                  (cv_x === -$signed((1<<(IW-1))-1)));
        cv_bf = 16'h7F80; #0.1;
        pass_fail("bf16_to_fix  +Inf -> 포화 + ovf", (cv_ovf === 1'b1));
        cv_bf = r2bf(2.0 ** (-16)); #0.1;      // 정확히 0.5 LSB -> round-half-up
        pass_fail("bf16_to_fix  2^-16 (0.5 LSB) -> 1", cv_x === 1);
        cv_bf = r2bf(2.0 ** (-17)); #0.1;
        pass_fail("bf16_to_fix  2^-17 -> 0 (언더플로)", cv_x === 0);
        // xsh 로 창 이동 : 값을 2^k 배 하고 xsh=-k 면 결과가 같아야 한다
        bad = 0;
        for (k = -8; k <= 8; k = k + 1) begin
            cv_bf = r2bf(1.375 * (2.0 ** k)); cv_sh = -k; #0.1;
            if (cv_x !== $rtoi(1.375 * (1 << IF))) bad = bad + 1;
        end
        cv_sh = 0;
        pass_fail("bf16_to_fix xsh 창 이동 (2^k 배 <-> xsh=-k)", bad == 0);

        // ================= PART B : rsqrt_unit =================
        $display("");
        $display("== PART B : rsqrt_unit  (v = 1 .. 2^%0d, VF=%0d) ==", VW-1, VF);
        rmax_e = 0.0; n_pts = 0; p_bad = 0;
        @(negedge clk);                            // PART A 의 #0.1 로 어긋난 위상 복구
        for (k = 20; k < VW; k = k + 1)            // 지수 전 구간 (짝/홀 모두)
        for (i = 0; i < 24; i = i + 1) begin
            vv = ({{(VW-1){1'b0}}, 1'b1} << k);
            for (j = 0; j < k; j = j + 1)
                if ($random(gseed_g) & 1) vv[j] = 1'b1;
            rq_v = vv; rq_iv = 1; @(negedge clk); rq_iv = 0;
            while (!rq_ov) @(negedge clk);
            v_ref = 1.0 / $sqrt(u2r(vv) * (2.0 ** (-VF)));
            v_hw  = (rq_r / (2.0 ** RF)) * (2.0 ** (-$signed(rq_e)));
            rel   = (v_hw - v_ref) / v_ref; if (rel < 0.0) rel = -rel;
            if (rel > rmax_e) rmax_e = rel;
            if (rq_r < (1 << (RF-1)) || rq_r > (1 << RF)) p_bad = p_bad + 1;
            n_pts = n_pts + 1;
        end
        $display("   %0d점 스윕, max 상대오차 = %.4e", n_pts, rmax_e);
        pass_fail("rsqrt_unit 상대오차", rmax_e <= TOL_RSQ);
        pass_fail("rsqrt_unit r 정규화 범위 (0.5, 1.0]", p_bad == 0);

        // ================= PART C : 지정 Tile =================
        $display("");
        $display("== PART C : 지정 Tile ==");

        // C1 : 행마다 다른 상수 (var = 0) -> y = 0
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) set_elem(0, r, d, (r - 16) * 3.5);
        run_tile("C1 상수행 (var=0)", TOL_ELEM, 1);
        bad = 0;
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) if (ym[0][r][d] !== 0) bad = bad + 1;
        pass_fail("C1 상수행 -> y 가 전부 정확히 0", bad == 0);

        // C2 : 행마다 다른 기울기의 ramp
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1)
            set_elem(0, r, d, (d - 63.5) * (0.01 * (r+1)) + (r - 16));
        run_tile("C2 ramp (행별 기울기)", TOL_ELEM, 1);

        // C3 : one-hot -> |y| 가 이론 최대 sqrt(D-1) 에 근접
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1)
            set_elem(0, r, d, (d == r*4) ? 10.0 : 1.0);
        run_tile("C3 one-hot (|y| 최대)", TOL_ELEM, 1);

        // C4 : 큰 평균 + 작은 분산  <- E[x^2]-mu^2 상쇄오차 저격
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1)
            set_elem(0, r, d, 100.0 + (r+1)*0.05*randn(0));
        run_tile("C4 mu=100, sigma~0.05..1.6", TOL_ELEM, 1);

        // C5 : 대스케일 (포화 직전)
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1)
            set_elem(0, r, d, 200.0*randn(0)*0.25);
        run_tile("C5 대스케일 (|x| ~ 200)", TOL_ELEM, 1);

        // C6 : 톱니 + 부호 혼합
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1)
            set_elem(0, r, d, ((d % (r+2)) - (r+2)/2.0) * (1.0 + r*0.1));
        run_tile("C6 톱니 (부호 혼합)", TOL_ELEM, 1);

        // C7 : 스케일 불변성 — 입력을 2^k 배 하고 in_shift=-k 면 비트 동일
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) set_elem(0, r, d, 2.0*randn(0) + r*0.5);
        run_tile("C7 기준 (in_shift=0)", TOL_ELEM, 0);
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) yr[0][r][d] = ym[0][r][d];
        bad = 0;
        for (k = -6; k <= 6; k = k + 3) begin
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1)
                set_elem(0, r, d, bf2r(xb[0][r][d]) * (2.0 ** k));
            zs = 0; ocap = 0; lc0 = last_cnt;
            drive_tile(0, -k, zs);
            wait (last_cnt == lc0 + 1); @(negedge clk);
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1)
                if (ym[0][r][d] !== yr[0][r][d]) bad = bad + 1;
            // 다음 루프를 위해 원본 복구
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1)
                set_elem(0, r, d, bf2r(xb[0][r][d]) * (2.0 ** (-k)));
        end
        pass_fail("C7 스케일 불변성 (2^k 배 + in_shift=-k -> 비트 동일)", bad == 0);
        pass_fail("C1~C7 구간 포화 플래그 미발생", ln_ovf === 1'b0);

        // ================= PART D : 랜덤 Tile =================
        $display("");
        $display("== PART D : 랜덤 Tile ==");
        fill_random(0, 0.0,  1.0);   run_tile("D1 mu=0    sd~1",    TOL_ELEM, 1);
        fill_random(0, 5.0,  1.0);   run_tile("D2 mu~5    sd~1",    TOL_ELEM, 1);
        fill_random(0, 0.0,  0.2);   run_tile("D3 mu=0    sd~0.2",  TOL_ELEM, 1);
        fill_random(0, 50.0, 2.0);   run_tile("D4 mu~50   sd~2",    TOL_ELEM, 1);
        fill_random(0, 0.0, 20.0);   run_tile("D5 mu=0    sd~20",   TOL_ELEM, 1);
        fill_random(0, 20.0, 0.5);   run_tile("D6 mu~20   sd~0.5",  TOL_ELEM, 1);

        // ================= PART E : 프로토콜 =================
        $display("");
        $display("== PART E : 프로토콜 ==");
        // E1 : in_valid gap 을 넣어도 결과가 비트 단위로 동일한가
        fill_random(0, 2.0, 1.5);
        run_tile("E1 기준 (gap 없음)", TOL_ELEM, 0);
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) yr[0][r][d] = ym[0][r][d];
        zs = 32'h5EED_0001; ocap = 0; lc0 = last_cnt;
        drive_tile(0, 6'sd0, zs);
        wait (last_cnt == lc0 + 1); @(negedge clk);
        bad = 0;
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) if (ym[0][r][d] !== yr[0][r][d]) bad = bad + 1;
        pass_fail("E1 in_valid gap 삽입시 비트 단위 동일", bad == 0);

        // E2 : 동작 중 리셋 -> 복구
        zs = 0; ocap = 0;
        @(negedge clk);
        ln_iv = 1'b1; ln_col = GARBAGE;
        repeat (40) @(negedge clk);
        ln_iv = 1'b0;
        rst_n = 1'b0; repeat (6) @(negedge clk);
        rst_n = 1'b1; repeat (6) @(negedge clk);
        ocol = 0; ocap = 0; last_cnt = 0; last_at = -1; oovf = 0;
        t_lastprev = -1;
        fill_random(0, 1.0, 1.0);
        run_tile("E2 리셋 후 복구", TOL_ELEM, 1);
        pass_fail("E2 동작중 리셋 후 정상 복구", 1'b1);

        // E3 : 3 Tile 을 끊김없이 밀어넣고(슬롯 3개 동시 사용) 전부 검증
        for (ti = 0; ti < 3; ti = ti + 1) fill_random(ti, 3.0*(ti+1), 1.0+ti);
        zs = 0; ocap = 0; lc0 = last_cnt;
        fork
            begin
                for (ti = 0; ti < 3; ti = ti + 1) drive_tile(ti, 6'sd0, zs);
            end
        join
        wait (last_cnt == lc0 + 3);
        @(negedge clk);
        for (ti = 0; ti < 3; ti = ti + 1) begin
            $write("  E3 slot%0d ", ti);
            check_tile("E3 3-Tile 중첩", ti, TOL_ELEM, 1);
        end

        // ================= PART F : 타이밍 =================
        $display("");
        $display("== PART F : 타이밍 ==");
        per_min = 1000000; per_max = -1; per_cnt = 0; t_lastprev = -1;
        for (ti = 0; ti < 3; ti = ti + 1) fill_random(ti, 0.0, 1.0);
        zs = 0; ocap = 0; lc0 = last_cnt;
        e2e_en = 1'b1; t_first_in = -1; t_last_out = -1;
        for (i = 0; i < 2; i = i + 1)               // 6 Tile 연속 (정상상태 확인)
        for (ti = 0; ti < 3; ti = ti + 1) drive_tile(ti, 6'sd0, zs);
        wait (last_cnt == lc0 + 6);
        @(negedge clk);
        e2e_en = 1'b0;
        $display("   Tile 처리주기 실측 : min=%0d  max=%0d clk  (%0d개 구간)",
                 per_min, per_max, per_cnt);
        $display("   구조식 기대값       : max(S1=%0d, S2=%0d, S3=%0d) = %0d clk",
                 S1_CLK, S2_CLK, S3_CLK, EXP_PER);
        pass_fail("Tile 처리주기 = 구조식 기대값 (버블 없음)",
                  (per_min == EXP_PER) && (per_max == EXP_PER));
        $display("   throughput = %.2f elem/clk  (%0d 원소 / %0d clk)",
                 (LANE*D*1.0)/EXP_PER, LANE*D, EXP_PER);

        // 96 x 128 end-to-end (Tile 3개) — 파이프라인을 비우고 새로 측정
        repeat (400) @(negedge clk);
        per_min = 1000000; per_max = -1; per_cnt = 0; t_lastprev = -1;
        for (ti = 0; ti < 3; ti = ti + 1) fill_random(ti, 0.0, 1.0);
        zs = 0; ocap = 0; lc0 = last_cnt;
        t_first_in = -1; t_last_out = -1; e2e_en = 1'b1;
        for (ti = 0; ti < 3; ti = ti + 1) drive_tile(ti, 6'sd0, zs);
        wait (last_cnt == lc0 + 3);
        @(negedge clk); e2e_en = 1'b0;
        $display("   M(96) x D(128) 첫 in_valid -> 마지막 out_last = %0d clk 경과 = %.3f us @500MHz",
                 t_last_out - t_first_in, (t_last_out - t_first_in) * 2.0e-3);
        $display("   구조식 : S1+S2+S3+(n-1)*P = %0d+%0d+%0d+2*%0d = %0d clk",
                 S1_CLK, S2_CLK, S3_CLK, EXP_PER, E2E_3);
        pass_fail("96x128 end-to-end = 구조식 기대값",
                  (t_last_out - t_first_in) == E2E_3);
        for (ti = 0; ti < 3; ti = ti + 1) check_tile("F 검증", ti, TOL_ELEM, 0);

        // ---- F2 : 30 Tile 연속 — in_ready 가 내려가는 **정상상태 throttle** 구간까지
        //           밀어 넣어야 슬롯 2벌짜리(acc_sq / rr / ee)의 최악 조건이 나온다.
        //           데이터는 Tile 0~2 를 순환시켜 마지막 3 Tile 을 그대로 검증한다.
        $display("");
        $display("   [F2] 200 Tile 연속 인가 (정상상태 throttle 구간 진입 확인)");
        for (ti = 0; ti < 3; ti = ti + 1) fill_random(ti, 4.0*(ti+1), 1.0+ti);
        ir_low = 0; per_min = 1000000; per_max = -1; per_cnt = 0; t_lastprev = -1;
        tiles_in = 0; stall_first_tile = -1; marg_at = -1;      // F2 기준으로 다시 셈
        sq_marg = 999999; re_marg = 999999; sq_clash = 0; re_clash = 0;
        re_lastrd[0] = -1; re_lastrd[1] = -1; sq_lastrd[0] = -1; sq_lastrd[1] = -1;
        zs = 0; ocap = 0; lc0 = last_cnt;
        for (i = 0; i < 200; i = i + 1) drive_tile(i % 3, 6'sd0, zs);
        wait (last_cnt == lc0 + 200);
        @(negedge clk);
        $display("        in_ready 하강 %0d clk,  처음 내려간 시점 = %0d번째 Tile 수신 후",
                 ir_low, stall_first_tile);
        $display("        Tile 처리주기 min=%0d max=%0d clk", per_min, per_max);
        pass_fail("F2 throttle 구간 도달 (in_ready 하강 발생)", ir_low > 0);
        pass_fail("F2 200 Tile 연속에서도 주기 일정", (per_min == EXP_PER) && (per_max == EXP_PER));
        $display("        2슬롯 자원 여유 마진 : sum_x2 %0d clk,  rstd %0d clk  (최소값 발생 = %0d번째 Tile)",
                 sq_marg, re_marg, marg_at);
        pass_fail("F2 sum_x2 슬롯 충돌 없음 (읽기/쓰기 겹침 0)", sq_clash == 0);
        pass_fail("F2 rstd   슬롯 충돌 없음 (읽기/쓰기 겹침 0)", re_clash == 0);
        for (ti = 0; ti < 3; ti = ti + 1)
            check_tile("F2 200-Tile 연속", ti, TOL_ELEM, 0);

        // ================= PART G : 스케일 민감도 =================
        //  X 는 Q8.15 (LSB 3.05e-5) 이므로 y 오차 ~ (LSB/2)/sqrt(var+eps) 이다.
        //  sigma 가 작아지면 이 항이 커진다 — 창(window)이 데이터에 비해 성긴 것이다.
        //  in_shift 로 창을 2^k 옮기면 해소되는데, 이때 **실효 eps 도 2^-2k 배**가
        //  되므로 골든도 같은 eps 로 계산해야 공정한 비교가 된다.
        $display("");
        $display("== PART G : 입력 스케일 민감도 (고정소수점 창 Q%0d.%0d, LSB=%.2e) ==",
                 IW-1-IF, IF, LSB_IN);
        $display("   sigma      in_shift  실효eps      max err        LSB    판정");
        for (i = 0; i < 6; i = i + 1) begin
            sc = 10.0 ** (-i);                    // sigma = 1, 0.1, ... 1e-5
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1) set_elem(0, r, d, sc*randn(0));
            zs = 0; ocap = 0; lc0 = last_cnt;
            drive_tile(0, 6'sd0, zs);
            wait (last_cnt == lc0 + 1); @(negedge clk);
            eps_use = EPS_R; calc_gold(0);
            dif = 0.0;
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1) begin
                rel = $signed(ym[0][r][d])*LSB_OUT - gm[0][r][d];
                if (rel < 0.0) rel = -rel;
                if (rel > dif) dif = rel;
            end
            $display("   %-9.1e  %4d     %.2e   %.4e  %8.2f   %s", sc, 0, EPS_R,
                     dif, dif/LSB_OUT, (dif <= TOL_ELEM) ? "ok" : "격하");
            if (i == 0) pass_fail("G sigma=1 에서 오차 허용치 이내", dif <= TOL_ELEM);
            if (i == 1) pass_fail("G sigma=0.1 에서 오차 허용치 이내", dif <= TOL_ELEM);
        end
        // 창을 옮기면 작은 sigma 도 복구된다 (실효 eps 도 2^-30 배가 된다)
        sc = 1.0e-5;
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) set_elem(0, r, d, sc*randn(0));
        zs = 0; ocap = 0; lc0 = last_cnt;
        drive_tile(0, 6'sd15, zs);                 // 2^15 배 -> 내부 sigma ~ 0.33
        wait (last_cnt == lc0 + 1); @(negedge clk);
        eps_use = EPS_R * (2.0 ** (-30)); calc_gold(0);
        dif = 0.0;
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) begin
            rel = $signed(ym[0][r][d])*LSB_OUT - gm[0][r][d];
            if (rel < 0.0) rel = -rel;
            if (rel > dif) dif = rel;
        end
        $display("   %-9.1e  %4d     %.2e   %.4e  %8.2f   %s", sc, 15, eps_use,
                 dif, dif/LSB_OUT, (dif <= TOL_ELEM) ? "ok" : "격하");
        pass_fail("G in_shift=15 로 sigma=1e-5 복구", dif <= TOL_ELEM);
        eps_use = EPS_R;

        // ================= PART H : eps 직접 검증 + 코너 + 비트정확 모델 =========
        $display("");
        $display("== PART H : eps 정확성 / 코너 케이스 / 비트정확 모델 ==");
        $display("   EPS_INT = %0d,  eps = %.10e  (목표 1e-5, 상대오차 %.2e)",
                 EPS_INT, EPS_R, (EPS_R-1.0e-5)/1.0e-5 > 0 ?
                 (EPS_R-1.0e-5)/1.0e-5 : (1.0e-5-EPS_R)/1.0e-5);
        pass_fail("H0 eps 가 1e-5 (상대오차 < 1e-6)",
                  ((EPS_R > 1.0e-5*(1.0-1.0e-6)) && (EPS_R < 1.0e-5*(1.0+1.0e-6))));

        // ---- H1 : eps 를 정면으로 겨냥한 테스트 -------------------------------
        //  행을 +-a (a = 2^-k) 교대로 채우면
        //     X = 2^(15-k) 정확,  SX = 0 정확,  var = a^2 정확  -> 양자화 오차 0
        //  이 되어 y = a/sqrt(a^2 + eps) 가 **오직 eps 에만** 의존한다.
        //  a 가 sqrt(eps) 근처면 eps 유무로 y 가 배 이상 달라지므로 eps 값이
        //  1e-5 인지가 비트 수준으로 드러난다.
        $display("");
        $display("   H1 : x = +-2^-k 교대 (X 정확, var=a^2 정확) -> y = a/sqrt(a^2+eps)");
        $display("        k    a          y(eps=1e-5)   y(eps=0)   DUT y        오차(LSB)");
        bad = 0;
        for (k = 4; k <= 14; k = k + 2) begin
            sc = 2.0 ** (-k);
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1) set_elem(0, r, d, (d[0] ? -sc : sc));
            zs = 0; ocap = 0; lc0 = last_cnt;
            drive_tile(0, 6'sd0, zs);
            wait (last_cnt == lc0 + 1); @(negedge clk);
            v_ref = sc / $sqrt(sc*sc + EPS_R);          // eps 반영 기준값
            v_hw  = sc / $sqrt(sc*sc);                  // eps=0 이었다면 (= 1.0)
            dif = 0.0;
            for (r = 0; r < LANE; r = r + 1)
            for (d = 0; d < D; d = d + 1) begin
                rel = $signed(ym[0][r][d])*LSB_OUT - (d[0] ? -v_ref : v_ref);
                if (rel < 0.0) rel = -rel;
                if (rel > dif) dif = rel;
            end
            $display("       %2d   %.3e   %.7f     %.7f  %.7f    %6.2f  %s",
                     k, sc, v_ref, v_hw, $signed(ym[0][0][0])*LSB_OUT,
                     dif/LSB_OUT, (dif <= TOL_ELEM) ? "ok" : "*** FAIL");
            if (dif > TOL_ELEM) bad = bad + 1;
        end
        pass_fail("H1 eps=1e-5 반영 확인 (2^-4 ~ 2^-14 전 구간)", bad == 0);

        // ---- H2 : 코너 케이스 -------------------------------------------------
        $display("");
        // H2a : var ~ 0 이지만 d != 0  -> u = 0 (좌시프트 최대) 경계
        //       127개는 2^-8 (X=128), 1개만 129*2^-15 (X=129) — 둘 다 BF16 정확
        for (r = 0; r < LANE; r = r + 1) begin
            for (d = 0; d < D; d = d + 1) set_elem(0, r, d, 128.0/32768.0);
            set_elem(0, r, r*4, 129.0/32768.0);
        end
        run_tile("H2a var~0, d!=0 (u=0 경계)", TOL_ELEM, 1);
        check_model("H2a", 0, 0);

        // H2b : var 최대 (|x| ~ 255 교대) -> u 최대 경계
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) set_elem(0, r, d, (d[0] ? -255.0 : 255.0));
        run_tile("H2b var 최대 (u 최대 경계)", TOL_ELEM, 1);
        check_model("H2b", 0, 0);

        // H2c : 포화 입력 (|x| > 256) + BF16 특수값 혼합
        //       ovf 가 서야 하고, 포화된 X 기준으로는 여전히 정상 동작해야 한다
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) begin
            if      (d % 16 == 0) set_elem(0, r, d, 1000.0);      // 포화
            else if (d % 16 == 1) set_elem(0, r, d, -4000.0);     // 포화
            else if (d % 16 == 2) xb[0][r][d] = 16'h0000;         // +0
            else if (d % 16 == 3) xb[0][r][d] = 16'h8000;         // -0
            else if (d % 16 == 4) xb[0][r][d] = 16'h0001;         // subnormal
            else                  set_elem(0, r, d, 3.0*randn(0));
            if (d % 16 >= 2 && d % 16 <= 4) xv[0][r][d] = bf2r(xb[0][r][d]);
        end
        zs = 0; ocap = 0; lc0 = last_cnt;
        drive_tile(0, 6'sd0, zs);
        wait (last_cnt == lc0 + 1); @(negedge clk);
        pass_fail("H2c |x|>256 입력에서 ovf 플래그 발생", ln_ovf === 1'b1);
        check_model("H2c 포화+특수값", 0, 0);   // 포화 포함 비트 단위 일치

        // ---- H3 : 비트정확 모델 대조 (일반 데이터) ---------------------------
        $display("");
        m_umin = 999; m_umax = -999; m_qmin = 999; m_qmax = -999;
        fill_random(0, 0.0,  1.0);  run_tile("H3-1 mu=0 sd~1", TOL_ELEM, 0);
        check_model("H3-1 mu=0 sd~1", 0, 0);
        fill_random(0, 80.0, 0.3);  run_tile("H3-2 mu~80 sd~0.3", TOL_ELEM, 0);
        // mu>>sigma 라 E[x^2]-mu^2 상쇄가 가장 심한 조건 — 정수식이 정말 정확한지 표시
        $display("   H3-2 정수식 분산 vs float 분산 (mu~80, sigma~0.3 : 상쇄 최악 조건)");
        dbg_model = 1'b1; check_model("H3-2 mu~80 sd~0.3", 0, 0); dbg_model = 1'b0;
        fill_random(0, 0.0, 30.0);  run_tile("H3-3 mu=0 sd~30", TOL_ELEM, 0);
        check_model("H3-3 mu=0 sd~30", 0, 0);
        for (r = 0; r < LANE; r = r + 1)                    // in_shift 를 건 경우도
        for (d = 0; d < D; d = d + 1) set_elem(0, r, d, 1.0e-3*randn(0));
        zs = 0; ocap = 0; lc0 = last_cnt;
        drive_tile(0, 6'sd10, zs);
        wait (last_cnt == lc0 + 1); @(negedge clk);
        check_model("H3-4 in_shift=10", 0, 10);
        $display("   모델이 본 범위 : q = %0d..%0d,  u = %0d..%0d   (설계 가정 q>=%0d, u in [0,%0d])",
                 m_qmin, m_qmax, m_umin, m_umax, VF-17, MAXU_);

        // ---- H4 : 내부 계층참조 감시 결과 -------------------------------------
        $display("");
        $display("   H4 : DUT 내부 시프트량 u 실측 범위 = %0d .. %0d  (설계 범위 [0, %0d])",
                 h_umin, h_umax, MAXU_);
        pass_fail("H4 u 가 설계 범위 [0, MAXU] 안에 있음",
                  (h_umin >= 0) && (h_umax <= MAXU_));
        pass_fail("H4 u 클램프가 한 번도 동작하지 않음 (MAXSHL 유도가 맞음)",
                  h_uclamp == 0);
        $display("   H4 : ds 포화 %0d회, y 포화 %0d회 (정상 데이터에서 0 이어야 함)",
                 h_esat, h_ysat);
        pass_fail("H4 y(Q4.11) 포화가 한 번도 없음", h_ysat == 0);

        // ================= PART I : 96x128 덤프 (torch 교차검증용) =============
        $display("");
        $display("== PART I : 96x128 전체 행렬 덤프 (verify_torch.py 교차검증용) ==");
        for (ti = 0; ti < 3; ti = ti + 1) fill_random(ti, 2.0, 1.0);
        zs = 0; ocap = 0; lc0 = last_cnt;
        for (ti = 0; ti < 3; ti = ti + 1) drive_tile(ti, 6'sd0, zs);
        wait (last_cnt == lc0 + 3);
        @(negedge clk);
        fd_i = $fopen("data/tb_in_bf16.hex",  "w");
        fd_o = $fopen("data/tb_out_q411.hex", "w");
        for (ti = 0; ti < 3; ti = ti + 1)
        for (r = 0; r < LANE; r = r + 1)
        for (d = 0; d < D; d = d + 1) begin
            $fwrite(fd_i, "%04x\n", xb[ti][r][d]);
            $fwrite(fd_o, "%04x\n", ym[ti][r][d]);
        end
        $fclose(fd_i); $fclose(fd_o);
        for (ti = 0; ti < 3; ti = ti + 1) check_model("I 96x128", ti, 0);
        $display("   data/tb_in_bf16.hex, data/tb_out_q411.hex 기록 (96 x 128 행 우선)");

        // ================= 요약 =================
        $display("");
        $display("############################################################");
        $display("  출력 : signed Q%0d.%0d (%0db, LSB=%.4e)", OW-1-OF, OF, OW, LSB_OUT);
        $display("  내부 : X=Q%0d.%0d(%0db)  V=%0db(정확)  r=UQ1.%0d(%0db)  ds=Q%0d.%0d(%0db)",
                 IW-1-IF, IF, IW, VW, RF, RW, DSW-1-DSF, DSF, DSW);
        $display("");
        $display("  검증 원소 수        : %0d elements", g_nelem);
        $display("  원소 최대 절대오차  : %.4e  (%.2f LSB of Q%0d.%0d)   [기준 %.1e]",
                 g_maxerr, g_maxerr/LSB_OUT, OW-1-OF, OF, TOL_ELEM);
        $display("  원소 전체 RMS 오차  : %.4e  (%.2f LSB)               [기준 %.1e]",
                 $sqrt(g_sqsum/g_nelem), $sqrt(g_sqsum/g_nelem)/LSB_OUT, TOL_RMS);
        $display("  bf16_to_fix 최대오차: %.3f LSB", cvt_max);
        $display("  rsqrt_unit  최대오차: %.4e (rel)", rmax_e);
        $display("  체크 항목           : %0d", checks);
        $display("");
        $display("  Tile 처리주기 = %0d clk,  throughput = %.2f elem/clk",
                 EXP_PER, (LANE*D*1.0)/EXP_PER);
        $display("  96x128 end-to-end = %0d clk 경과 = %.3f us @500MHz",
                 E2E_3, E2E_3*2.0e-3);
        $display("");
        if (errors == 0) $display("  RESULT : *** ALL PASS ***");
        else             $display("  RESULT : *** %0d FAIL ***", errors);
        $display("############################################################");
        $display("");
        $finish;
    end

    initial begin
        #1500000;
        $display("*** TIMEOUT ***");
        $finish;
    end
endmodule
