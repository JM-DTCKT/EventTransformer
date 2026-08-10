// ============================================================================
//  tb_softmax.v  --  32행 Tile T-축 softmax 검증 TB  (Q6.9 in / signed Q1.14 out)
// ----------------------------------------------------------------------------
//  골든 레퍼런스를 외부 파일 없이 TB 안에서 real 연산($exp)으로 직접 계산한다.
//
//  DUT 는 QK^T 결과의 32행 Tile 을 **열 단위(32원소)** 로 T 회 받아,
//  수신하면서 행별 max 를 갱신하고, Tile 이 다 차면 exp/역수/곱셈을 수행해
//  다시 열 단위로 32행분 y 를 T 회 내보낸다.
//
//   PART A  exp2_unit 단독  : z = 0 ~ -16.0 전 구간(8193점) vs $exp
//   PART B  recip_unit 단독 : S = 1 ~ TMAX 스윕 + 랜덤, p 정확성
//   PART C  지정 Tile : uniform / one-hot / 동점 / ramp / 톱니 + T 경계값
//   PART D  랜덤 Tile : 분포 스케일 4종, T 랜덤
//   PART E  프로토콜 : in_valid gap, 동작중 리셋, 행 독립성
//   PART F  타이밍 : Tile 처리주기 실측 + ping-pong 중첩 확인
//
//  Tile 마다 32행 전부에 대해 확인
//   - 원소별 절대/RMS 오차 vs 이상적 float softmax
//   - sum(y) ~= 1.0 (행별)
//   - argmax 보존 (양자화 동점 허용)
//   - 단조성 / 결정성
//   - out_last 위치
// ============================================================================
`timescale 1ns/1ps

module tb_softmax;

    // ---------------- 파라미터 ----------------
    localparam integer Tile_M = 32;  // Tile 행 수 (= Tensor Core 타일 행 수)
    localparam integer TMAX = 324;   // 최대 T
    localparam integer OW   = 16;    // 출력 폭
    localparam integer OF   = 14;    // 출력 소수부 (signed Q1.OF) — 여기만 바꾸면 됨
    localparam integer EF   = 16;    // e 소수부 -> e 는 UQ1.EF
    localparam integer RF   = 17;    // R 소수부 -> R 은 UQ1.RF
    localparam integer EW   = EF + 1;
    localparam integer RW   = RF + 1;
    localparam integer BRAM_LAT = 1; // BRAM read latency (1=코어, 2=출력레지스터)
    localparam integer TWB  = 9;     // $clog2(TMAX)
    localparam integer SWS  = EF + TWB + 1;   // = 26, sum 폭

    localparam real LSB_IN  = 1.0 / 512.0;
    localparam real LSB_OUT = 2.0 ** (-OF);
    localparam real ONE_E   = 2.0 ** EF;
    localparam real ONE_R   = 2.0 ** RF;
    localparam integer S_ONE = 1 << EF;
    localparam integer S_MAX = TMAX << EF;

    localparam signed [15:0] XMIN = 16'sh8000;   // -64.0
    localparam signed [15:0] XMAX = 16'sh7FFF;   // +63.998

    localparam real TOL_ELEM = 1.5e-4;
    localparam real TOL_RMS  = 5.0e-5;
    localparam real TOL_SUM  = 1.0e-2;
    localparam real TOL_EXP  = 1.0e-5 + 4.0 * (2.0 ** (-EF-1));
    localparam real TOL_RCP  = 8.0e-5;

    // 3-stage Tile 파이프라인의 기대 주기 = max(S1, S2, S3)
    localparam integer S1_CLK  = TMAX;                 // 수신
    localparam integer S2_CLK  = 1 + (TMAX+BRAM_LAT+4) + (Tile_M+3); // exp + recip
    localparam integer S3_CLK  = 1 + (TMAX+BRAM_LAT);                // mul
    localparam integer EXP_PER = (S1_CLK > S2_CLK)
                               ? ((S1_CLK > S3_CLK) ? S1_CLK : S3_CLK)
                               : ((S2_CLK > S3_CLK) ? S2_CLK : S3_CLK);

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
            $display("   %-46s : %s", item, ok ? "PASS" : "*** FAIL ***");
        end
    endtask

    // ================= DUT #1 : exp2_unit =================
    reg                  exp_iv;
    reg  signed [16:0]   exp_z;
    wire                 exp_ov;
    wire        [EW-1:0] exp_e;
    exp2_unit #(.ZW(17), .ZF(9), .EW(EW), .EF(EF)) u_exp_dut (
        .clk(clk), .rst_n(rst_n), .in_valid(exp_iv), .z(exp_z),
        .out_valid(exp_ov), .e(exp_e));

    // ================= DUT #2 : recip_unit =================
    reg  [SWS-1:0] rcp_s;
    reg            rcp_iv;
    wire           rcp_ov;
    wire [RW-1:0]  rcp_r;
    wire [4:0]     rcp_p;
    recip_unit #(.SW(SWS), .RW(RW), .RF(RF), .PW(5)) u_rcp_dut (
        .clk(clk), .rst_n(rst_n), .in_valid(rcp_iv), .s(rcp_s),
        .out_valid(rcp_ov), .r(rcp_r), .p(rcp_p));

    // ================= DUT #3 : softmax_top (메인) =================
    reg              sm_iv, sm_ilast;
    reg  [Tile_M*16-1:0]  sm_col;
    wire             sm_ir, sm_ov, sm_ol;
    wire [Tile_M*16-1:0]  sm_oc;
    softmax_top #(.Tile_M(Tile_M), .TMAX(TMAX), .OW(OW), .OF(OF),
                  .EW(EW), .EF(EF), .RW(RW), .RF(RF), .BRAM_LAT(BRAM_LAT)) u_sm (
        .clk(clk), .rst_n(rst_n),
        .in_valid(sm_iv), .in_ready(sm_ir), .in_col(sm_col), .in_last(sm_ilast),
        .out_valid(sm_ov), .out_col(sm_oc), .out_last(sm_ol));

    localparam [Tile_M*16-1:0] GARBAGE = {(Tile_M*16/32){32'hDEAD_BEEF}};

    // ---------------- 데이터 / 골든 ----------------
    reg signed [15:0] xm   [0:Tile_M-1][0:TMAX-1];   // 입력 Tile
    reg        [15:0] ym   [0:Tile_M-1][0:TMAX-1];   // DUT 출력
    reg        [15:0] ymr  [0:Tile_M-1][0:TMAX-1];   // 기준 출력 (gap 비교용)
    real              gm   [0:Tile_M-1][0:TMAX-1];   // 이상적 float softmax

    integer cur_T;
    integer ocol, last_cnt, last_at, ovf;
    integer cyc, t_lastprev, per_min, per_max, per_cnt;
    integer t_first_in, t_last_out;   // M x T 전체 end-to-end 실측
    reg     e2e_en;
    integer g_nelem, g_ntile;
    real    g_maxerr, g_sqsum;

    initial begin
        ocol = 0; last_cnt = 0; last_at = -1; ovf = 0; cyc = 0;
        t_lastprev = -1; per_min = 1000000; per_max = -1; per_cnt = 0;
        t_first_in = -1; t_last_out = -1; e2e_en = 1'b0;
        g_nelem = 0; g_ntile = 0; g_maxerr = 0.0; g_sqsum = 0.0; cur_T = 0;
    end

    // ---------------- 출력 캡처 ----------------
    integer cr;
    always @(posedge clk) begin
        if (rst_n) begin
            cyc <= cyc + 1;
            if (e2e_en && sm_iv && sm_ir && t_first_in < 0) t_first_in <= cyc;
            if (e2e_en && sm_ov && sm_ol)                   t_last_out <= cyc;
            if (sm_ov) begin
                if (ocol < cur_T)
                    for (cr = 0; cr < Tile_M; cr = cr + 1)
                        ym[cr][ocol] = sm_oc[cr*16 +: 16];
                else ovf = ovf + 1;
                ocol = ocol + 1;
                if (sm_ol) begin
                    last_cnt = last_cnt + 1; last_at = ocol - 1;
                    if (t_lastprev >= 0) begin
                        if (cyc - t_lastprev < per_min) per_min = cyc - t_lastprev;
                        if (cyc - t_lastprev > per_max) per_max = cyc - t_lastprev;
                        per_cnt = per_cnt + 1;
                    end
                    t_lastprev = cyc;
                    ocol = 0;                       // 다음 Tile 준비
                end
            end
        end
    end

    // ---------------- 골든 : 행별 float softmax ----------------
    task calc_gold;
        integer r, t, ti; real mx, s;
        begin
            for (r = 0; r < Tile_M; r = r + 1) begin
                mx = -1.0e30;
                for (t = 0; t < cur_T; t = t + 1) begin
                    ti = xm[r][t];
                    if (ti * LSB_IN > mx) mx = ti * LSB_IN;
                end
                s = 0.0;
                for (t = 0; t < cur_T; t = t + 1) begin
                    ti = xm[r][t];
                    gm[r][t] = $exp(ti * LSB_IN - mx);
                    s = s + gm[r][t];
                end
                for (t = 0; t < cur_T; t = t + 1) gm[r][t] = gm[r][t] / s;
            end
        end
    endtask

    // ---------------- Tile 인가 (열 단위 T회) ----------------
    //  gseed != 0 이면 열 사이에 랜덤 gap + 무효구간 쓰레기값
    task drive_tile;
        input integer T;
        inout integer gseed;
        integer t, r, g, lc0;
        begin
            cur_T = T; lc0 = last_cnt;
            @(negedge clk);
            while (!sm_ir) @(negedge clk);
            for (t = 0; t < T; t = t + 1) begin
                if (gseed != 0) begin
                    g = $random(gseed); if (g < 0) g = -g;
                    repeat (g % 3) begin
                        sm_iv = 1'b0; sm_col = GARBAGE; sm_ilast = 1'b0;
                        @(negedge clk);
                    end
                end
                for (r = 0; r < Tile_M; r = r + 1) sm_col[r*16 +: 16] = xm[r][t];
                sm_iv    = 1'b1;
                sm_ilast = (t == T-1);
                @(negedge clk);
                while (!sm_ir) begin              // 뱅크가 full 이면 대기
                    sm_iv = 1'b0; sm_ilast = 1'b0; @(negedge clk);
                end
            end
            sm_iv = 1'b0; sm_ilast = 1'b0; sm_col = GARBAGE;
            wait (last_cnt == lc0 + 1);
            @(negedge clk);
        end
    endtask

    // ---------------- Tile 검증 (32행 전부) ----------------
    task check_tile;
        input string tname;
        input        verbose;
        integer r, t, u, ti, am_d, am_g, mono_bad, det_bad, n_pair, seedl, sign_bad;
        real    yd, e, mx, rms, sy, s_worst;
        begin
            calc_gold;
            mx = 0.0; rms = 0.0; mono_bad = 0; det_bad = 0; s_worst = 0.0; sign_bad = 0;
            am_d = 0; am_g = 0;
            seedl = 12345;

            for (r = 0; r < Tile_M; r = r + 1) begin
                sy = 0.0; am_d = 0; am_g = 0;
                for (t = 0; t < cur_T; t = t + 1) begin
                    // signed Q1.14 해석. 출력은 항상 >=0 이라 부호비트는 0 이어야 한다.
                    if (ym[r][t][OW-1] !== 1'b0) sign_bad = sign_bad + 1;
                    ti = $signed(ym[r][t]);
                    yd = ti * LSB_OUT;
                    e  = yd - gm[r][t]; if (e < 0.0) e = -e;
                    if (e > mx) mx = e;
                    rms = rms + e*e;
                    sy  = sy + yd;
                    if (ym[r][t] > ym[r][am_d]) am_d = t;
                    if (gm[r][t] > gm[r][am_g]) am_g = t;
                    if (e > g_maxerr) g_maxerr = e;
                    g_sqsum = g_sqsum + e*e;
                    g_nelem = g_nelem + 1;
                end
                if (sy - 1.0 > s_worst)  s_worst = sy - 1.0;
                if (1.0 - sy > s_worst)  s_worst = 1.0 - sy;
                // argmax : 양자화 동점 허용 (순위 역전만 잡음)
                if (ym[r][am_g] !== ym[r][am_d]) begin
                    $display("   *** FAIL %0s row%0d : argmax 역전 dut=%0d gold=%0d",
                             tname, r, am_d, am_g);
                    errors = errors + 1;
                end
                // 단조성/결정성 : T 가 작으면 전수, 크면 랜덤 샘플
                if (cur_T <= 64) begin
                    for (t = 0; t < cur_T; t = t + 1)
                        for (u = 0; u < cur_T; u = u + 1) begin
                            if ((xm[r][t] >  xm[r][u]) && (ym[r][t] <  ym[r][u]))
                                mono_bad = mono_bad + 1;
                            if ((xm[r][t] == xm[r][u]) && (ym[r][t] != ym[r][u]))
                                det_bad = det_bad + 1;
                        end
                end else begin
                    for (n_pair = 0; n_pair < 300; n_pair = n_pair + 1) begin
                        t = $random(seedl); if (t < 0) t = -t; t = t % cur_T;
                        u = $random(seedl); if (u < 0) u = -u; u = u % cur_T;
                        if ((xm[r][t] >  xm[r][u]) && (ym[r][t] <  ym[r][u]))
                            mono_bad = mono_bad + 1;
                        if ((xm[r][t] == xm[r][u]) && (ym[r][t] != ym[r][u]))
                            det_bad = det_bad + 1;
                    end
                end
            end
            rms = $sqrt(rms / (Tile_M * cur_T));
            g_ntile = g_ntile + 1;

            if (verbose || mx > TOL_ELEM || rms > TOL_RMS || s_worst > TOL_SUM
                        || mono_bad != 0 || det_bad != 0
                        || last_at != cur_T-1 || ovf != 0)
                $display("  [%-20s T=%3d] maxerr=%.3e (%5.2f LSB) rms=%.3e |sum-1|max=%.2e%s%s%s",
                         tname, cur_T, mx, mx/LSB_OUT, rms, s_worst,
                         (mono_bad != 0) ? "  MONO-VIOL" : "",
                         (det_bad  != 0) ? "  DET-VIOL"  : "",
                         (last_at != cur_T-1) ? "  LAST-VIOL" : "");

            if (mx > TOL_ELEM) begin
                $display("   *** FAIL %0s : max err %.3e > %.3e", tname, mx, TOL_ELEM);
                errors = errors + 1; end
            if (rms > TOL_RMS) begin
                $display("   *** FAIL %0s : rms %.3e > %.3e", tname, rms, TOL_RMS);
                errors = errors + 1; end
            if (s_worst > TOL_SUM) begin
                $display("   *** FAIL %0s : |sum-1| = %.4e", tname, s_worst);
                errors = errors + 1; end
            if (mono_bad != 0) begin
                $display("   *** FAIL %0s : monotonicity %0d", tname, mono_bad);
                errors = errors + 1; end
            if (det_bad != 0) begin
                $display("   *** FAIL %0s : determinism %0d", tname, det_bad);
                errors = errors + 1; end
            if (sign_bad != 0) begin
                $display("   *** FAIL %0s : 부호비트가 선 원소 %0d개 (signed Q1.%0d 위반)",
                         tname, sign_bad, OF);
                errors = errors + 1; end
            if (last_at != cur_T-1) begin
                $display("   *** FAIL %0s : out_last at %0d (expect %0d)",
                         tname, last_at, cur_T-1);
                errors = errors + 1; end
            if (ovf != 0) begin
                $display("   *** FAIL %0s : 초과 출력 %0d", tname, ovf);
                errors = errors + 1; ovf = 0; end
        end
    endtask

    // ---------------- 출력 대기 없이 연속 인가 (ping-pong 중첩 측정용) ----
    task drive_tile_nowait;
        input integer T;
        integer t, r;
        begin
            cur_T = T;
            @(negedge clk);
            for (t = 0; t < T; t = t + 1) begin
                while (!sm_ir) begin sm_iv = 1'b0; sm_ilast = 1'b0; @(negedge clk); end
                for (r = 0; r < Tile_M; r = r + 1) sm_col[r*16 +: 16] = xm[r][t];
                sm_iv    = 1'b1;
                sm_ilast = (t == T-1);
                @(negedge clk);
            end
            sm_iv = 1'b0; sm_ilast = 1'b0;
        end
    endtask

    integer zs;
    task run_tile;
        input string  tname;
        input integer T;
        input         verbose;
        begin
            zs = 0;
            drive_tile(T, zs);
            check_tile(tname, verbose);
        end
    endtask

    // ========================================================================
    //  메인
    // ========================================================================
    integer i, j, k, r, t, ti, seed, err_mark, p_bad, Tv, n_pts;
    real    exp_max, exp_hw, exp_ref, rcp_max, rcp_hw, rcp_ref, dif;
    integer rnd;  reg [31:0] rndu;

    initial begin
        seed = 32'h1234_5678;
        sm_iv = 0; sm_ilast = 0; sm_col = GARBAGE;
        exp_iv = 0; exp_z = 0; rcp_iv = 0; rcp_s = 0;
        rst_n = 0; repeat (8) @(negedge clk);
        rst_n = 1; repeat (4) @(negedge clk);

        $display("");
        $display("############################################################");
        $display("#  32-row Tile  T-axis Softmax   (Q6.9 in / UQ1.15 out)");
        $display("#  Tile_M=%0d  TMAX=%0d  입력/출력 = 열 단위 %0d원소 (%0db)",
                 Tile_M, TMAX, Tile_M, Tile_M*16);
        $display("#  출력 : signed Q1.%0d (%0db, 1.0=16'h%0h, LSB=%.3e)",
                 OF, OW, (1<<OF), LSB_OUT);
        $display("#  내부 : e=UQ1.%0d(%0db)  R=UQ1.%0d(%0db)  S=%0db", EF,EW,RF,RW,SWS);
        $display("#  BRAM read latency = %0d clk", BRAM_LAT);
        $display("############################################################");

        // ---------------- PART A ----------------
        $display("");
        $display("== PART A : exp2_unit ==");
        exp_max = 0.0; n_pts = 0;
        for (i = 0; i <= 8192; i = i + 1) begin
            exp_z = -i; exp_iv = 1; @(negedge clk); exp_iv = 0;
            while (!exp_ov) @(negedge clk);
            ti = exp_e; exp_hw = ti / ONE_E; exp_ref = $exp(-i * LSB_IN);
            dif = exp_hw - exp_ref; if (dif < 0.0) dif = -dif;
            if (dif > exp_max) exp_max = dif;
            n_pts = n_pts + 1;
        end
        $display("   %0d점 스윕, max abs error = %.4e (%.2f LSB of UQ1.%0d)",
                 n_pts, exp_max, exp_max*ONE_E, EF);
        pass_fail("exp2_unit z-sweep accuracy", exp_max <= TOL_EXP);
        exp_z = 0; exp_iv = 1; @(negedge clk); exp_iv = 0;
        while (!exp_ov) @(negedge clk);
        pass_fail("exp2_unit exp(0) == exactly 1.0", exp_e === S_ONE[EW-1:0]);

        // ---------------- PART B ----------------
        $display("");
        $display("== PART B : recip_unit  (S = 1 .. %0d, SW=%0db) ==", TMAX, SWS);
        rcp_max = 0.0; p_bad = 0;
        for (i = 0; i < 3000; i = i + 1) begin
            if (i < 1500) rcp_s = S_ONE + i * ((S_MAX - S_ONE)/1500);
            else begin
                rnd = $random(seed); if (rnd < 0) rnd = -rnd;
                rcp_s = S_ONE + (rnd % (S_MAX - S_ONE));
            end
            rcp_iv = 1; @(negedge clk); rcp_iv = 0;
            while (!rcp_ov) @(negedge clk);
            ti = rcp_r; rcp_hw = (ti / ONE_R) * $pow(2.0, EF*1.0 - rcp_p);
            ti = rcp_s; rcp_ref = 1.0 / (ti / ONE_E);
            dif = (rcp_hw - rcp_ref)/rcp_ref; if (dif < 0.0) dif = -dif;
            if (dif > rcp_max) rcp_max = dif;
            if ((rcp_s >> rcp_p) != 1) p_bad = p_bad + 1;
        end
        $display("   3000점, max rel error = %.4e", rcp_max);
        pass_fail("recip_unit sweep accuracy", rcp_max <= TOL_RCP);
        pass_fail("recip_unit leading-one correct", p_bad == 0);

        // ---------------- PART C ----------------
        $display("");
        $display("== PART C : 지정 Tile + T 경계값 ==");

        for (r = 0; r < Tile_M; r = r + 1) for (t = 0; t < TMAX; t = t + 1) xm[r][t] = 0;
        run_tile("C1 all-zero", TMAX, 1);
        ti = ym[0][0];
        $display("       y = %0d (%.8f), ideal 1/%0d = %.8f",
                 ti, ti*LSB_OUT, TMAX, 1.0/TMAX);
        pass_fail("C1 전 원소 동일", (ym[0][0]==ym[31][323]) && (ym[0][0]==ym[7][100]));

        // 행마다 다른 위치에 one-hot -> 행 독립성 확인
        for (r = 0; r < Tile_M; r = r + 1) begin
            for (t = 0; t < TMAX; t = t + 1) xm[r][t] = XMIN;
            xm[r][r*10] = XMAX;
        end
        run_tile("C2 행별 one-hot", TMAX, 1);
        p_bad = 0;
        for (r = 0; r < Tile_M; r = r + 1)
            if (ym[r][r*10] !== (16'd1 << OF)) p_bad = p_bad + 1;
        pass_fail("C2 행별 y[max] == 1.0 (32행 전부)", p_bad == 0);

        // 행마다 다른 스케일 -> 행별 max/sum 이 섞이지 않는지
        for (r = 0; r < Tile_M; r = r + 1)
            for (t = 0; t < TMAX; t = t + 1)
                xm[r][t] = (t - 162) * (r + 1) * 16'sd3;
        run_tile("C3 행별 다른 기울기", TMAX, 1);

        for (r = 0; r < Tile_M; r = r + 1) begin
            for (t = 0; t < TMAX; t = t + 1) xm[r][t] = -16'sd20000;
            xm[r][10] = 16'sd16000; xm[r][300] = 16'sd16000;
        end
        run_tile("C4 동점 max", TMAX, 1);
        pass_fail("C4 동점 : y[10]==y[300] (전 행)",
                  (ym[0][10]==ym[0][300]) && (ym[31][10]==ym[31][300]));

        for (r = 0; r < Tile_M; r = r + 1)
            for (t = 0; t < TMAX; t = t + 1) xm[r][t] = ((t+r) % 97)*16'sd37 - 16'sd1800;
        run_tile("C5 T=1",    1,   1);
        run_tile("C6 T=2",    2,   1);
        run_tile("C7 T=63",   63,  1);
        run_tile("C8 T=96",   96,  1);
        run_tile("C9 T=323",  323, 1);
        run_tile("C10 T=324", 324, 1);

        for (r = 0; r < Tile_M; r = r + 1)
            for (t = 0; t < TMAX; t = t + 1) xm[r][t] = (t % 2) ? XMAX : XMIN;
        run_tile("C11 saw max/min", TMAX, 1);
        for (r = 0; r < Tile_M; r = r + 1)
            for (t = 0; t < TMAX; t = t + 1) xm[r][t] = XMIN;
        run_tile("C12 all-XMIN", TMAX, 1);

        // ---------------- PART D ----------------
        $display("");
        $display("== PART D : 랜덤 Tile (스케일 4종, T 랜덤) ==");
        for (j = 0; j < 4; j = j + 1) begin
            for (k = 0; k < 3; k = k + 1) begin
                rnd = $random(seed); if (rnd < 0) rnd = -rnd;
                Tv = 1 + (rnd % TMAX);
                for (r = 0; r < Tile_M; r = r + 1)
                    for (t = 0; t < Tv; t = t + 1) begin
                        rnd = $random(seed); rndu = rnd;
                        case (j)
                        0: xm[r][t] = rnd % 64;
                        1: xm[r][t] = rnd % 1024;
                        2: xm[r][t] = rnd % 8192;
                        default: xm[r][t] = rndu[15:0];
                        endcase
                    end
                run_tile("D random", Tv, 0);
            end
            $display("   scale %0d : 3 Tiles OK  (누적 max err %.3e = %.2f LSB)",
                     j, g_maxerr, g_maxerr/LSB_OUT);
        end

        // ---------------- PART E ----------------
        $display("");
        $display("== PART E : 프로토콜 ==");
        $display("   [E1] in_valid gap (무효구간 쓰레기값) vs gap 없음");
        err_mark = errors; p_bad = 0;
        for (k = 0; k < 2; k = k + 1) begin
            rnd = $random(seed); if (rnd < 0) rnd = -rnd;
            Tv = 1 + (rnd % 200);
            for (r = 0; r < Tile_M; r = r + 1)
                for (t = 0; t < Tv; t = t + 1) begin
                    rnd = $random(seed); xm[r][t] = rnd % 6000; end
            run_tile("E1 no-gap ref", Tv, 0);
            for (r = 0; r < Tile_M; r = r + 1)
                for (t = 0; t < Tv; t = t + 1) ymr[r][t] = ym[r][t];
            drive_tile(Tv, seed);
            check_tile("E1 gapped", 0);
            for (r = 0; r < Tile_M; r = r + 1)
                for (t = 0; t < Tv; t = t + 1)
                    if (ym[r][t] !== ymr[r][t]) p_bad = p_bad + 1;
        end
        pass_fail("E1 gap 유무 출력 비트단위 일치", p_bad == 0);
        pass_fail("E1 gapped Tile 정확도 정상",    errors == err_mark);

        $display("   [E2] 연산 도중 리셋 후 복구");
        for (r = 0; r < Tile_M; r = r + 1)
            for (t = 0; t < TMAX; t = t + 1) xm[r][t] = (t+r) * 16'sd7;
        cur_T = TMAX; ocol = 0;
        @(negedge clk);
        while (!sm_ir) @(negedge clk);
        for (t = 0; t < 100; t = t + 1) begin        // 일부만 인가하고 리셋
            for (r = 0; r < Tile_M; r = r + 1) sm_col[r*16 +: 16] = xm[r][t];
            sm_iv = 1; sm_ilast = 0; @(negedge clk);
        end
        sm_iv = 0;
        rst_n = 0; repeat (5) @(negedge clk);
        rst_n = 1; repeat (5) @(negedge clk);
        ovf = 0; last_cnt = 0; ocol = 0;
        pass_fail("E2 리셋 직후 in_ready 복귀", sm_ir === 1'b1);
        pass_fail("E2 리셋 직후 out_valid 없음", sm_ov === 1'b0);
        err_mark = errors;
        for (r = 0; r < Tile_M; r = r + 1)
            for (t = 0; t < 150; t = t + 1) xm[r][t] = (t - 75) * (r+1) * 16'sd5;
        run_tile("E2 reset 후 재실행", 150, 1);
        pass_fail("E2 리셋 후 결과 정상", errors == err_mark);

        // ---------------- PART F ----------------
        $display("");
        $display("== PART F : Tile 처리주기 실측 ==");
        for (r = 0; r < Tile_M; r = r + 1)
            for (t = 0; t < TMAX; t = t + 1) xm[r][t] = (t - 162) * 16'sd50;
        err_mark = errors;
        run_tile("F 단독 (정확도 확인)", TMAX, 0);
        pass_fail("F 단독 Tile 정확도 정상", errors == err_mark);

        // ping-pong 중첩 : 출력 완료를 기다리지 않고 연속 인가
        per_min = 1000000; per_max = -1; per_cnt = 0; t_lastprev = -1;
        p_bad = last_cnt;  ovf = 0;
        for (k = 0; k < 5; k = k + 1) drive_tile_nowait(TMAX);
        wait (last_cnt >= p_bad + 5);
        repeat (3) @(negedge clk);
        ovf = 0;
        $display("   T=%0d 연속 5 Tile (중첩) : out_last 간격 %0d ~ %0d clk",
                 TMAX, per_min, per_max);
        $display("   -> Tile 처리주기 = %0d clk   (32행 x %0d = %0d 원소)",
                 per_min, TMAX, Tile_M*TMAX);
        $display("   -> throughput = %0d / %0d = %.2f elem/clk",
                 Tile_M*TMAX, per_min, Tile_M*TMAX*1.0/per_min);
        $display("   -> 설계값 : max( S1 수신 %0d , S2 exp+rcp %0d , S3 mul %0d ) = %0d",
                 TMAX, S2_CLK, S3_CLK, EXP_PER);
        $display("   -> 정상상태 처리율 기준 96행 = 3 x %0d clk (실제 end-to-end 는 F2)",
                 per_min);
        pass_fail("F 처리주기 == max(S1, S2, S3)", per_min == EXP_PER);
        pass_fail("F 처리주기 일정 (버블 없음)", per_min == per_max);

        // ---- F2 : M(96) x T(324) 전체 = Tile 3개 end-to-end ----
        $display("");
        $display("   [F2] M=96 x T=%0d 전체 (Tile 3개) end-to-end 실측", TMAX);
        rst_n = 1'b0; repeat (5) @(negedge clk);      // 파이프라인 비우고 시작
        rst_n = 1'b1; repeat (5) @(negedge clk);
        last_cnt = 0; ocol = 0;
        t_first_in = -1; t_last_out = -1; e2e_en = 1'b1;
        for (k = 0; k < 3; k = k + 1) drive_tile_nowait(TMAX);
        wait (last_cnt >= 3);
        repeat (2) @(negedge clk);
        e2e_en = 1'b0;
        $display("   첫 in_valid -> 3번째 out_last : %0d clk 경과", t_last_out - t_first_in);
        $display("   내역 = S1(%0d) + S2(%0d) + S3(%0d) + 2 x 주기(%0d) = %0d",
                 S1_CLK, S2_CLK, S3_CLK, EXP_PER,
                 S1_CLK + S2_CLK + S3_CLK + 2*EXP_PER);
        $display("   -> @500MHz = %.3f us   (정상상태 처리율만 보면 3x%0d = %0d clk)",
                 (t_last_out - t_first_in) * 2.0 / 1000.0, EXP_PER, 3*EXP_PER);
        pass_fail("F2 end-to-end == S1+S2+S3 + 2x주기",
                  (t_last_out - t_first_in) == S1_CLK + S2_CLK + S3_CLK + 2*EXP_PER);

        // ---------------- 종합 ----------------
        $display("");
        $display("############################################################");
        $display("#  종합 결과");
        $display("############################################################");
        $display("  검증 Tile 수       : %0d Tiles (%0d elements)", g_ntile, g_nelem);
        $display("  원소 최대 절대오차  : %.4e  (%.2f LSB of Q1.%0d)  [기준 %.1e]",
                 g_maxerr, g_maxerr/LSB_OUT, OF, TOL_ELEM);
        $display("  원소 전체 RMS 오차  : %.4e  (%.2f LSB of Q1.%0d)  [기준 %.1e]",
                 $sqrt(g_sqsum/g_nelem), $sqrt(g_sqsum/g_nelem)/LSB_OUT, OF, TOL_RMS);
        $display("  exp2_unit  최대오차 : %.4e (abs)", exp_max);
        $display("  recip_unit 최대오차 : %.4e (rel)", rcp_max);
        $display("  체크 항목           : %0d", checks);
        $display("------------------------------------------------------------");
        if (errors == 0) $display("  RESULT : *** ALL PASS ***");
        else             $display("  RESULT : *** %0d FAIL ***", errors);
        $display("############################################################");
        $display("");
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("ERROR: simulation timeout");
        $finish;
    end
endmodule
