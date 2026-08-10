// ============================================================================
//  tb_softmax.v  --  96-D softmax unit 기능 검증 테스트벤치
// ----------------------------------------------------------------------------
//  골든 레퍼런스를 외부 파일로 두지 않고 TB 안에서 real 연산($exp)으로 직접
//  계산해 비교한다.  따라서 데이터 파일/스크립트 의존성이 전혀 없다.
//
//  검증 구성
//   PART A  exp2_unit 단독  : z = 0 ~ -16.0 전 구간(8193점) 스윕 vs $exp
//   PART B  recip_unit 단독 : S = 1.0 ~ 96.0 스윕 + 랜덤 vs 1/S, p 정확성
//   PART C  softmax_top 지정 벡터 : uniform / one-hot / ramp / 경계 / 동점 등
//   PART D  softmax_top 랜덤 벡터 : 분포 스케일 4종 x 각 12개
//   PART E  프로토콜 : 출력 개수, out_last 위치, latency, back-to-back, 리셋복구
//   * 출력은 매 clk P(=8)원소씩 N/P beat 로 나온다 (out_data = P x UQ1.15)
//
//  벡터마다 확인하는 것
//   - 원소별 절대오차 (max / RMS)          vs 이상적 float softmax
//   - sum(y) ~= 1.0                        (확률분포 성질)
//   - argmax 보존                          (분류 결과 불변)
//   - 단조성 : x_i > x_j  ->  y_i >= y_j   (순서 보존)
//   - 결정성 : x_i == x_j -> y_i == y_j
// ============================================================================
`timescale 1ns/1ps

module tb_softmax;

    // ---------------- 공통 파라미터 / 포맷 상수 ----------------
    localparam integer N       = 96;
    localparam integer PLANE   = 8;                // exp lane 수
    // ---- 내부 정밀도 (DSP 폭에 맞춰 조정하는 노브) ----
    localparam integer EF      = 16;               // e 소수부  -> e 는 UQ1.EF, EW=EF+1
    localparam integer RF      = 17;               // R 소수부  -> R 은 UQ1.RF, RW=RF+1
    localparam integer EW      = EF + 1;
    localparam integer RW      = RF + 1;
    localparam integer SWS     = EF + 7 + 1;       // recip 입력(sum) 폭
    localparam integer S_ONE   = 1 << EF;          // S = 1.0
    localparam integer S_MAX   = N << EF;          // S = 96.0

    localparam real    LSB_IN  = 1.0 / 512.0;      // Q6.9   입력 LSB
    localparam real    LSB_OUT = 1.0 / 32768.0;    // UQ1.15 출력 LSB
    localparam real    ONE_E   = 2.0 ** EF;        // e 의 1.0
    localparam real    ONE_R   = 2.0 ** RF;        // R 의 1.0

    localparam signed [15:0] XMIN = 16'sh8000;     // -64.0    (Q6.9 최소)
    localparam signed [15:0] XMAX = 16'sh7FFF;     // +63.998  (Q6.9 최대)

    // ---------------- 합격 기준 (오차 예산 분석 기반) ----------------
    //   exp PWL 6.4e-6 + e 양자화 2^-18 + recip 상대 9.0e-6 + 최종 반올림 0.5LSB
    //   => 원소당 절대오차 ~5e-5 (1.6 LSB) 예상.  여유 3배로 잡는다.
    localparam real TOL_ELEM = 1.5e-4;   // 원소별 최대 절대오차
    localparam real TOL_RMS  = 5.0e-5;   // 원소별 RMS 오차
    localparam real TOL_SUM  = 2.5e-3;   // |sum(y)-1| : 96 x 0.5LSB = 1.46e-3
    //  exp  : PWL(세그먼트수 고정) + e 양자화 2^-(EF+1)  -> EF 에 연동
    //  recip: PWL 이 지배적이라 RF 와 거의 무관 -> 고정값
    localparam real TOL_EXP  = 1.0e-5 + 4.0 * (2.0 ** (-EF-1));
    localparam real TOL_RCP  = 8.0e-5;

    // ---------------- 클럭 / 리셋 ----------------
    reg clk = 1'b0;
    reg rst_n;
    always #1 clk = ~clk;                // 2ns period (500MHz)

    integer errors = 0;                  // 전체 FAIL 카운트
    integer checks = 0;                  // 전체 체크 항목 수

    task pass_fail;                      // 공통 판정 출력
        input string item;
        input        ok;
        begin
            checks = checks + 1;
            if (!ok) errors = errors + 1;
            $display("   %-42s : %s", item, ok ? "PASS" : "*** FAIL ***");
        end
    endtask

    // ========================================================================
    //  DUT #1 : exp2_unit 단독
    // ========================================================================
    reg                exp_iv;
    reg  signed [16:0] exp_z;
    wire               exp_ov;
    wire        [EW-1:0] exp_e;
    exp2_unit #(.ZW(17), .ZF(9), .EW(EW), .EF(EF)) u_exp_dut (
        .clk(clk), .rst_n(rst_n), .in_valid(exp_iv), .z(exp_z),
        .out_valid(exp_ov), .e(exp_e)
    );

    // ========================================================================
    //  DUT #2 : recip_unit 단독
    // ========================================================================
    reg         rcp_iv;
    reg  [SWS-1:0] rcp_s;
    wire           rcp_ov;
    wire [RW-1:0]  rcp_r;
    wire [4:0]     rcp_p;
    recip_unit #(.SW(SWS), .RW(RW), .RF(RF), .PW(5)) u_rcp_dut (
        .clk(clk), .rst_n(rst_n), .in_valid(rcp_iv), .s(rcp_s),
        .out_valid(rcp_ov), .r(rcp_r), .p(rcp_p)
    );

    // ========================================================================
    //  DUT #3 : softmax_top (메인)
    // ========================================================================
    reg                sm_iv;
    reg  [N*16-1:0]    sm_vec;          // 벡터 전체를 한 번에 (원소 i = [i*16 +: 16])
    wire               sm_ir;
    wire               sm_ov, sm_ol;
    wire  [PLANE*16-1:0] sm_od;      // 매 beat 8원소
    softmax_top #(.N(N), .P(PLANE), .EW(EW), .EF(EF), .RW(RW), .RF(RF)) u_sm (
        .clk(clk), .rst_n(rst_n),
        .in_valid(sm_iv), .in_ready(sm_ir), .in_vec(sm_vec),
        .out_valid(sm_ov), .out_data(sm_od), .out_last(sm_ol)
    );

    localparam [N*16-1:0] GARBAGE = {(N*16/32){32'hDEAD_BEEF}};   // 무효구간 구동값

    // ---------------- 출력 캡처 & 프로토콜 모니터 ----------------
    reg  [15:0] yv     [0:N-1];  // DUT 출력 수집
    reg  [15:0] yv_ref [0:N-1];  // gap 없는 경우의 기준 출력 (E3 비교용)
    integer     ocnt;            // 현재 벡터의 출력 개수
    integer     ocnt_tot;        // 전체 out_valid 누적
    integer     last_cnt;        // 현재 벡터의 out_last 개수
    integer     last_idx;        // out_last 가 뜬 위치
    integer     ovf_cnt;         // N개 초과 출력 (프로토콜 위반)
    integer     cyc;             // 사이클 카운터
    integer     ci;              // beat 내 원소 인덱스
    integer     t_in0, t_out0, t_outL;
    reg         meas_en;
    // 벡터 처리주기(throughput) 실측용 : out_last 와 out_last 사이 간격
    integer     t_lastprev, vec_per_min, vec_per_max, vec_per_cnt;

    initial begin
        ocnt = 0; ocnt_tot = 0; last_cnt = 0; last_idx = -1; ovf_cnt = 0;
        cyc  = 0; t_in0 = -1; t_out0 = -1; t_outL = -1; meas_en = 1'b0;
        t_lastprev = -1; vec_per_min = 1000000; vec_per_max = -1; vec_per_cnt = 0;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            cyc <= cyc + 1;
            if (meas_en && sm_iv && sm_ir && t_in0 < 0) t_in0 <= cyc;
            if (sm_ov) begin
                for (ci = 0; ci < PLANE; ci = ci + 1) begin   // beat 하나 = P원소
                    if (ocnt < N) yv[ocnt] = sm_od[ci*16 +: 16];
                    else          ovf_cnt = ovf_cnt + 1;
                    ocnt     = ocnt + 1;
                    ocnt_tot = ocnt_tot + 1;
                end
                if (sm_ol) begin
                    last_cnt = last_cnt + 1; last_idx = ocnt - 1;   // 마지막 원소 인덱스
                    if (t_lastprev >= 0) begin      // 벡터 처리주기 실측
                        if (cyc - t_lastprev < vec_per_min) vec_per_min = cyc - t_lastprev;
                        if (cyc - t_lastprev > vec_per_max) vec_per_max = cyc - t_lastprev;
                        vec_per_cnt = vec_per_cnt + 1;
                    end
                    t_lastprev = cyc;
                end
                if (meas_en) begin
                    if (t_out0 < 0) t_out0 <= cyc;
                    t_outL <= cyc;
                end
            end
        end
    end

    // ---------------- 테스트 벡터 / 골든 ----------------
    reg signed [15:0] xv   [0:N-1];      // 입력 (Q6.9)
    real              xr   [0:N-1];      // 입력 실수값
    real              gold [0:N-1];      // 이상적 float softmax
    real              er   [0:N-1];

    real    g_maxerr, g_sqsum;           // 전역 통계
    integer g_nelem, g_nvec;

    // xv[] 를 입력 버스로 패킹
    task pack_vec;
        integer i;
        begin
            for (i = 0; i < N; i = i + 1) sm_vec[i*16 +: 16] = xv[i];
        end
    endtask

    // ---------------- 골든 : 이상적 float softmax ----------------
    task calc_gold;
        integer i, ti; real m, s;
        begin
            m = -1.0e30;
            for (i = 0; i < N; i = i + 1) begin
                ti    = xv[i];                    // signed reg -> integer (부호확장)
                xr[i] = ti * LSB_IN;
                if (xr[i] > m) m = xr[i];
            end
            s = 0.0;
            for (i = 0; i < N; i = i + 1) begin
                er[i] = $exp(xr[i] - m);          // max 를 빼고 exp (DUT 와 동일 전략)
                s     = s + er[i];
            end
            for (i = 0; i < N; i = i + 1) gold[i] = er[i] / s;
        end
    endtask

    // ---------------- 한 벡터 인가 & 출력 수집 ----------------
    //  벡터 전체를 1 clk handshake 로 넣는다.  handshake 직후 입력버스를 X 로
    //  덮어써서, DUT 가 벡터를 확실히 래치했고 이후 버스에 의존하지 않음을 확인한다.
    task drive_vector;
        begin
            ocnt = 0; last_cnt = 0; last_idx = -1;
            pack_vec;
            @(negedge clk);
            while (!sm_ir) @(negedge clk);        // idle(ST_LOAD) 대기
            sm_iv = 1'b1;
            @(negedge clk);                       // 직전 posedge 에서 handshake
            sm_iv  = 1'b0;
            sm_vec = {N*16{1'bx}};                // 입력버스 무효화
            wait (ocnt >= N);                     // 96개 출력 완료 대기
            @(negedge clk);
        end
    endtask

    // ---------------- 한 벡터 결과 검증 ----------------
    task check_vector;
        input string tname;
        input        verbose;
        integer i, j, ti, am_d, am_g, mono_bad, det_bad;
        real    ydut, e, mx, rms, sy;
        begin
            calc_gold;
            mx = 0.0; rms = 0.0; sy = 0.0; am_d = 0; am_g = 0;
            mono_bad = 0; det_bad = 0;

            for (i = 0; i < N; i = i + 1) begin
                ti   = yv[i];                       // unsigned 16b
                ydut = ti * LSB_OUT;
                e    = ydut - gold[i]; if (e < 0.0) e = -e;
                if (e > mx) mx = e;
                rms = rms + e * e;
                sy  = sy + ydut;
                if (yv[i]   > yv[am_d])   am_d = i;   // DUT argmax
                if (gold[i] > gold[am_g]) am_g = i;   // 이상값 argmax
                if (e > g_maxerr) g_maxerr = e;
                g_sqsum = g_sqsum + e * e;
                g_nelem = g_nelem + 1;
            end
            rms = $sqrt(rms / N);

            // 단조성 & 결정성 (O(N^2) 전수 비교)
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    if ((xv[i] >  xv[j]) && (yv[i] <  yv[j])) mono_bad = mono_bad + 1;
                    if ((xv[i] == xv[j]) && (yv[i] != yv[j])) det_bad  = det_bad  + 1;
                end

            g_nvec = g_nvec + 1;
            if (verbose || mx > TOL_ELEM || (sy-1.0 > TOL_SUM) || (1.0-sy > TOL_SUM)
                        || am_d != am_g || mono_bad != 0 || det_bad != 0
                        || last_cnt != 1 || last_idx != N-1 || ovf_cnt != 0)
                $display("  [%-24s] maxerr=%.3e (%5.2f LSB)  rms=%.3e  sum=%.7f  argmax d/g=%0d/%0d%s%s%s",
                         tname, mx, mx/LSB_OUT, rms, sy, am_d, am_g,
                         (mono_bad != 0) ? "  MONO-VIOL" : "",
                         (det_bad  != 0) ? "  DET-VIOL"  : "",
                         (last_cnt != 1 || last_idx != N-1) ? "  LAST-VIOL" : "");

            if (mx > TOL_ELEM) begin
                $display("   *** FAIL %0s : element error %.3e > %.3e", tname, mx, TOL_ELEM);
                errors = errors + 1;
                for (i = 0; i < N; i = i + 1) begin
                    ti = yv[i]; ydut = ti * LSB_OUT;
                    e  = ydut - gold[i]; if (e < 0.0) e = -e;
                    if (e > TOL_ELEM)
                        $display("       i=%2d x=%9.4f dut=%.8f gold=%.8f err=%.3e",
                                 i, xr[i], ydut, gold[i], e);
                end
            end
            if (rms > TOL_RMS) begin
                $display("   *** FAIL %0s : rms %.3e > %.3e", tname, rms, TOL_RMS);
                errors = errors + 1;
            end
            if ((sy-1.0 > TOL_SUM) || (1.0-sy > TOL_SUM)) begin
                $display("   *** FAIL %0s : sum(y) = %.7f", tname, sy);
                errors = errors + 1;
            end
            if (am_d != am_g) begin
                $display("   *** FAIL %0s : argmax dut=%0d gold=%0d", tname, am_d, am_g);
                errors = errors + 1;
            end
            if (mono_bad != 0) begin
                $display("   *** FAIL %0s : monotonicity violated in %0d pairs", tname, mono_bad);
                errors = errors + 1;
            end
            if (det_bad != 0) begin
                $display("   *** FAIL %0s : determinism violated in %0d pairs", tname, det_bad);
                errors = errors + 1;
            end
            if (last_cnt != 1 || last_idx != N-1 || ovf_cnt != 0) begin
                $display("   *** FAIL %0s : out_last cnt=%0d idx=%0d, overflow=%0d",
                         tname, last_cnt, last_idx, ovf_cnt);
                errors = errors + 1;
            end
        end
    endtask

    // ---------------- in_valid 를 늦춰서 인가 (handshake 게이팅 검증) ------
    //  in_ready 가 high 인데도 in_valid 를 안 올린 채 입력버스에 쓰레기값을
    //  흘린다.  in_valid 게이팅이 제대로 되면 DUT 는 이를 무시해야 하고,
    //  결과가 지연 없는 경우와 비트 단위로 완전히 같아야 한다.
    task drive_vector_gapped;
        inout integer sd;
        integer r;
        begin
            ocnt = 0; last_cnt = 0; last_idx = -1;
            @(negedge clk);
            while (!sm_ir) @(negedge clk);
            r = $random(sd); if (r < 0) r = -r;
            repeat (r % 7) begin                  // 0~6 cycle 동안 쓰레기값 구동
                sm_iv  = 1'b0;
                sm_vec = GARBAGE;
                @(negedge clk);
            end
            pack_vec;                             // 진짜 값을 실은 뒤에만 valid
            sm_iv = 1'b1;
            @(negedge clk);
            sm_iv  = 1'b0;
            sm_vec = {N*16{1'bx}};
            wait (ocnt >= N);
            @(negedge clk);
        end
    endtask

    // ---------------- 최대 throughput 스트리밍 구동 -----------------------
    //  in_ready 가 high 인 모든 사이클에 in_valid 를 올려 DUT 가 낼 수 있는
    //  최대 속도로 nvec 개 벡터를 연속 인가한다 (TB 가 만드는 gap 을 제거).
    task drive_stream;
        input integer nvec;
        integer sent;
        begin
            sent = 0;
            pack_vec;
            @(negedge clk);
            while (sent < nvec) begin
                if (sm_ir) begin                  // 다음 posedge 에서 수락됨
                    sm_iv = 1'b1;
                    @(negedge clk);
                    sent = sent + 1;
                end else begin                    // DUT 가 연산 중 -> 대기
                    sm_iv = 1'b0;
                    @(negedge clk);
                end
            end
            sm_iv = 1'b0;
        end
    endtask

    task run_vector;                      // 인가 + 검증
        input string tname;
        input        verbose;
        begin
            drive_vector;
            check_vector(tname, verbose);
        end
    endtask

    // ========================================================================
    //  메인 시나리오
    // ========================================================================
    integer    i, j, k, ti, seed, err_mark, n_pts, p_bad;
    real       exp_max, exp_ref, exp_hw, rcp_max, rcp_ref, rcp_hw, dif;
    // recip 절대오차 : (a) r = 1/m 도메인,  (b) 1/S 도메인
    real       rcp_abs_r, rcp_abs_i, r_hw_v, m_v, dif2;
    integer    rnd;
    reg [31:0] rndu;

    initial begin
        seed = 32'h1234_5678;
        g_maxerr = 0.0; g_sqsum = 0.0; g_nelem = 0; g_nvec = 0;
        sm_iv  = 1'b0; sm_vec = GARBAGE;
        exp_iv = 1'b0; exp_z = 17'sd0;
        rcp_iv = 1'b0; rcp_s = 25'd0;
        rst_n  = 1'b0;
        repeat (8) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        $display("");
        $display("############################################################");
        $display("#  96-D Softmax Unit   (in Q6.9 / out UQ1.15)   검증 시작");
        $display("#  내부 포맷 : e = UQ1.%0d (%0db),  R = UQ1.%0d (%0db),  exp lane = %0d",
                 EF, EW, RF, RW, PLANE);
        $display("############################################################");

        // ================================================================
        //  PART A : exp2_unit 단독 스윕
        // ================================================================
        $display("");
        $display("== PART A : exp2_unit  exp(z) = 2^(z*log2e), 소수부 PWL-LUT ==");
        exp_max = 0.0; n_pts = 0;
        for (i = 0; i <= 8192; i = i + 1) begin           // z = 0 .. -16.0
            exp_z  = -i;
            exp_iv = 1'b1;
            @(negedge clk);
            exp_iv = 1'b0;
            while (!exp_ov) @(negedge clk);
            ti      = exp_e;
            exp_hw  = ti / ONE_E;
            exp_ref = $exp(-i * LSB_IN);
            dif     = exp_hw - exp_ref; if (dif < 0.0) dif = -dif;
            if (dif > exp_max) exp_max = dif;
            n_pts = n_pts + 1;
            if (dif > TOL_EXP)
                $display("   z=%9.4f hw=%.9f ref=%.9f err=%.3e", -i*LSB_IN, exp_hw, exp_ref, dif);
        end
        $display("   sweep points  : %0d   (z = 0.0 .. -16.0, step 2^-9)", n_pts);
        $display("   max abs error : %.4e  (= %.2f LSB of UQ1.%0d)", exp_max, exp_max*ONE_E, EF);
        pass_fail("exp2_unit  z-sweep accuracy", exp_max <= TOL_EXP);

        // z=0 -> 정확히 1.0.  분모 S >= 1.0 을 보장하는 핵심 성질.
        exp_z = 17'sd0; exp_iv = 1'b1; @(negedge clk); exp_iv = 1'b0;
        while (!exp_ov) @(negedge clk);
        pass_fail("exp2_unit  exp(0) == exactly 1.0", exp_e === S_ONE[EW-1:0]);

        // 아주 작은 z -> 0 (clamp 영역, UQ1.17 로 표현 불가한 값)
        exp_z = -17'sd30000; exp_iv = 1'b1; @(negedge clk); exp_iv = 1'b0;
        while (!exp_ov) @(negedge clk);
        pass_fail("exp2_unit  exp(very negative) == 0", exp_e === {EW{1'b0}});

        // ================================================================
        //  PART B : recip_unit 단독 스윕
        // ================================================================
        $display("");
        $display("== PART B : recip_unit  1/S = (1/m) * 2^-p,  m in [1,2) ==");
        rcp_max = 0.0; p_bad = 0; rcp_abs_r = 0.0; rcp_abs_i = 0.0;
        for (i = 0; i < 4000; i = i + 1) begin
            if (i < 2000) begin
                rcp_s = S_ONE + i * ((S_MAX - S_ONE) / 2000);     // 균일 스윕
            end else begin
                rnd = $random(seed); if (rnd < 0) rnd = -rnd;
                rcp_s = S_ONE + (rnd % (S_MAX - S_ONE));          // 랜덤 [1.0, 96.0]
            end
            rcp_iv = 1'b1;
            @(negedge clk);
            rcp_iv = 1'b0;
            while (!rcp_ov) @(negedge clk);
            ti      = rcp_r;
            rcp_hw  = (ti / ONE_R) * $pow(2.0, EF * 1.0 - rcp_p); // 1/S 복원
            ti      = rcp_s;
            rcp_ref = 1.0 / (ti / ONE_E);
            dif     = (rcp_hw - rcp_ref) / rcp_ref; if (dif < 0.0) dif = -dif;
            if (dif > rcp_max) rcp_max = dif;
            // (a) LUT 출력 r = 1/m 자체의 절대오차 (Q0.20 도메인, r in (0.5,1])
            ti     = rcp_r;  r_hw_v = ti / ONE_R;
            ti     = rcp_s;  m_v    = ti / $pow(2.0, 1.0 * rcp_p);   // m = S / 2^p
            dif2   = r_hw_v - 1.0 / m_v; if (dif2 < 0.0) dif2 = -dif2;
            if (dif2 > rcp_abs_r) rcp_abs_r = dif2;
            // (b) 최종적으로 쓰이는 1/S 의 절대오차 (1/S in [1/96, 1])
            dif2   = rcp_hw - rcp_ref; if (dif2 < 0.0) dif2 = -dif2;
            if (dif2 > rcp_abs_i) rcp_abs_i = dif2;
            if ((rcp_s >> rcp_p) != 25'd1) begin                  // 2^p <= S < 2^(p+1)
                $display("   *** leading-one wrong : S=%0d p=%0d", rcp_s, rcp_p);
                p_bad = p_bad + 1;
            end
            if (dif > TOL_RCP)
                $display("   S=%0d hw=%.9f ref=%.9f relerr=%.3e", rcp_s, rcp_hw, rcp_ref, dif);
        end
        $display("   sweep points  : 4000  (S = 1.0 .. 96.0, 균일 2000 + 랜덤 2000)");
        $display("   max rel error         : %.4e", rcp_max);
        $display("   max abs error [r=1/m] : %.4e  (= %.2f LSB of UQ1.%0d, r in (0.5,1])",
                 rcp_abs_r, rcp_abs_r * ONE_R, RF);
        $display("   max abs error [1/S]   : %.4e  (1/S in [1/96, 1])", rcp_abs_i);
        pass_fail("recip_unit sweep accuracy", rcp_max <= TOL_RCP);
        pass_fail("recip_unit leading-one (p) correct", p_bad == 0);

        // ================================================================
        //  PART C : softmax_top 지정 벡터
        // ================================================================
        $display("");
        $display("== PART C : softmax_top 지정(directed) 벡터 ==");
        meas_en = 1'b1;

        // C1 : 전부 0 -> 완전 균등분포 1/96
        for (i = 0; i < N; i = i + 1) xv[i] = 16'sd0;
        run_vector("C1 all-zero(uniform)", 1);
        ti = yv[0];
        $display("       y[*] = %0d (%.8f)   ideal 1/96 = %.8f", ti, ti*LSB_OUT, 1.0/96.0);
        pass_fail("C1 uniform : 96개 원소 모두 동일", (yv[0]==yv[50]) && (yv[0]==yv[95]));

        // C2/C3 : 전부 같은 값 (max 뺄셈이 offset 을 제거하는지 확인)
        for (i = 0; i < N; i = i + 1) xv[i] = 16'sd10240;    // +20.0
        run_vector("C2 all-equal +20.0", 1);
        pass_fail("C2 : C1 과 동일 결과 (shift-invariance)", yv[0] == 16'd341);
        for (i = 0; i < N; i = i + 1) xv[i] = -16'sd25600;   // -50.0
        run_vector("C3 all-equal -50.0", 1);
        pass_fail("C3 : C1 과 동일 결과 (shift-invariance)", yv[0] == 16'd341);

        // C4 : 극단 one-hot -> y[max] = 1.0, 나머지 0
        for (i = 0; i < N; i = i + 1) xv[i] = XMIN;
        xv[37] = XMAX;
        run_vector("C4 one-hot max@37", 1);
        ti = yv[37];
        $display("       y[37] = %0d (%.8f)   1.0 = 32768", ti, ti*LSB_OUT);
        pass_fail("C4 one-hot : y[max] == 1.0 (32768)", yv[37] == 16'd32768);
        pass_fail("C4 one-hot : 나머지 원소 == 0",      (yv[0]==0) && (yv[95]==0));

        // C5 : 동점 최대값 2개 -> 각각 0.5
        for (i = 0; i < N; i = i + 1) xv[i] = -16'sd20000;
        xv[10] = 16'sd16000; xv[70] = 16'sd16000;
        run_vector("C5 tie max @10,@70", 1);
        pass_fail("C5 tie : y[10] == y[70]", yv[10] == yv[70]);

        // C6/C7 : max 위치가 벡터의 처음 / 끝
        for (i = 0; i < N; i = i + 1) xv[i] = -16'sd5000;
        xv[0] = 16'sd5000;
        run_vector("C6 max at index 0", 1);
        for (i = 0; i < N; i = i + 1) xv[i] = -16'sd5000;
        xv[N-1] = 16'sd5000;
        run_vector("C7 max at index 95", 1);

        // C8~C10 : 여러 기울기의 ramp (exp 정수부/소수부 전 영역 사용)
        for (i = 0; i < N; i = i + 1) xv[i] = (i - 48) * 16'sd256;
        run_vector("C8 ramp step 0.5", 1);
        for (i = 0; i < N; i = i + 1) xv[i] = (i - 48) * 16'sd16;
        run_vector("C9 ramp step 1/32", 1);
        for (i = 0; i < N; i = i + 1) xv[i] = (i - 48) * 16'sd680;
        run_vector("C10 ramp step 1.328", 1);

        // C11 : LSB 단위 미세 차이 -> 거의 균등, 단조성 스트레스
        for (i = 0; i < N; i = i + 1) xv[i] = i;
        run_vector("C11 LSB-step ramp", 1);

        // C12 : 톱니 (최대/최소가 교대) -> 48개가 1/48 씩
        for (i = 0; i < N; i = i + 1) xv[i] = (i % 2) ? XMAX : XMIN;
        run_vector("C12 saw max/min", 1);
        pass_fail("C12 saw : 48개 동일값 (1/48)", (yv[1]==yv[95]) && (yv[0]==0));

        // C13 : 전 구간 음수 (max 자체가 음수)
        for (i = 0; i < N; i = i + 1) xv[i] = -16'sd1000 - i * 16'sd100;
        run_vector("C13 all-negative decay", 1);

        // C14 : 값들이 촘촘히 몰린 경우 (exp 소수부 정밀도 스트레스)
        for (i = 0; i < N; i = i + 1) xv[i] = 16'sd1000 - (i * 16'sd3);
        run_vector("C14 near-tie cluster", 1);

        // ================================================================
        //  PART D : softmax_top 랜덤 벡터
        // ================================================================
        $display("");
        $display("== PART D : softmax_top 랜덤 벡터 (분포 스케일 4종 x 12개) ==");
        for (j = 0; j < 4; j = j + 1) begin
            for (k = 0; k < 12; k = k + 1) begin
                for (i = 0; i < N; i = i + 1) begin
                    rnd  = $random(seed);
                    rndu = rnd;
                    case (j)
                    0: xv[i] = rnd % 64;         // +-0.125   거의 균등
                    1: xv[i] = rnd % 1024;       // +-2.0
                    2: xv[i] = rnd % 8192;       // +-16.0
                    default: xv[i] = rndu[15:0]; // [-64, 64) 전 범위
                    endcase
                end
                run_vector("D random", 0);       // 이상 있을 때만 상세 출력
            end
            $display("   scale %0d : 12 vectors OK   (누적 max err %.3e = %.2f LSB)",
                     j, g_maxerr, g_maxerr/LSB_OUT);
        end

        // ================================================================
        //  PART E : 프로토콜 / 타이밍
        // ================================================================
        $display("");
        $display("== PART E : 프로토콜 & 타이밍 ==");
        $display("   벡터 latency (1st in_valid -> 1st out_valid) : %0d cycle", t_out0 - t_in0);
        pass_fail("out_valid 총 개수 == 96 x 벡터수", ocnt_tot == N * g_nvec);
        pass_fail("96개 초과 출력 없음",              ovf_cnt == 0);

        // E1 : 벡터 사이 idle 없이 연속 인가
        $display("   [E1] back-to-back 3 벡터 연속 인가");
        err_mark = errors;
        for (k = 0; k < 3; k = k + 1) begin
            for (i = 0; i < N; i = i + 1) begin
                rnd = $random(seed); xv[i] = rnd % 4096;
            end
            run_vector("E1 back-to-back", 0);
        end
        pass_fail("E1 back-to-back 3 벡터 정상", errors == err_mark);

        // E2 : 동작 중(ST_EXP) 리셋 -> 복구 후 정상 동작
        $display("   [E2] 연산 도중 리셋 인가 후 복구 확인");
        for (i = 0; i < N; i = i + 1) xv[i] = i * 16'sd7;
        ocnt = 0; last_cnt = 0; last_idx = -1;
        pack_vec;
        @(negedge clk);
        while (!sm_ir) @(negedge clk);
        sm_iv = 1'b1; @(negedge clk); sm_iv = 1'b0;
        repeat (30) @(negedge clk);        // exp 파이프라인이 도는 중
        rst_n = 1'b0;
        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (5) @(negedge clk);
        ovf_cnt = 0;                       // 리셋으로 버려진 부분출력 카운트 정리
        pass_fail("E2 리셋 직후 in_ready 복귀", sm_ir  === 1'b1);
        pass_fail("E2 리셋 직후 out_valid 없음", sm_ov  === 1'b0);
        err_mark = errors;
        for (i = 0; i < N; i = i + 1) xv[i] = (i - 40) * 16'sd333;
        run_vector("E2 reset 후 재실행", 1);
        pass_fail("E2 리셋 후 결과 정상", errors == err_mark);

        // E3 : in_valid 를 늦추고 쓰레기값 구동 -> 지연 없는 결과와 비트단위 동일해야
        $display("   [E3] in_valid 게이팅 검증 (무효구간에 쓰레기값 구동)");
        err_mark = errors; p_bad = 0;
        for (k = 0; k < 3; k = k + 1) begin
            for (i = 0; i < N; i = i + 1) begin
                rnd = $random(seed); xv[i] = rnd % 6000;
            end
            drive_vector;                              // (1) gap 없이
            for (i = 0; i < N; i = i + 1) yv_ref[i] = yv[i];
            check_vector("E3 no-gap reference", 0);
            drive_vector_gapped(seed);                 // (2) gap 있게
            check_vector("E3 gapped input", 0);
            for (i = 0; i < N; i = i + 1)
                if (yv[i] !== yv_ref[i]) p_bad = p_bad + 1;
        end
        pass_fail("E3 valid 게이팅 후 출력 비트단위 일치", p_bad == 0);
        pass_fail("E3 지연 인가 벡터 정확도 정상",         errors == err_mark);

        // E4 : in_ready 타이밍 — 연산 중 계속 low, 마지막 출력 사이클에만 복귀
        //      (그 1 clk 복귀 덕분에 back-to-back 인가가 버블 없이 된다)
        $display("   [E4] in_ready 타이밍");
        for (i = 0; i < N; i = i + 1) xv[i] = i * 16'sd11;
        ocnt = 0; last_cnt = 0; last_idx = -1;
        pack_vec;
        @(negedge clk);
        while (!sm_ir) @(negedge clk);
        pass_fail("E4 idle 상태에서 in_ready == 1", sm_ir === 1'b1);
        sm_iv = 1'b1;
        @(negedge clk);                       // handshake 완료
        sm_iv  = 1'b0;
        sm_vec = {N*16{1'bx}};
        pass_fail("E4 handshake 직후 in_ready == 0", sm_ir === 1'b0);
        p_bad = 0;                            // 연산 구간 중 in_ready high 사이클 수
        while (ocnt < N) begin
            if (sm_ir) p_bad = p_bad + 1;
            @(negedge clk);
        end
        $display("       연산 구간 중 in_ready high = %0d clk (마지막 복귀 1 clk 허용)", p_bad);
        pass_fail("E4 연산 중 in_ready low 유지", p_bad <= 1);
        @(negedge clk);
        check_vector("E4 in_ready 확인 벡터", 0);
        pass_fail("E4 완료 후 in_ready 복귀", sm_ir === 1'b1);

        // ================================================================
        //  PART F : throughput 실측 (최대 속도 연속 스트리밍)
        // ================================================================
        $display("");
        $display("== PART F : throughput 실측 (in_ready high 시 항상 인가) ==");
        for (i = 0; i < N; i = i + 1) xv[i] = (i - 48) * 16'sd120;
        ocnt = 0; last_cnt = 0; last_idx = -1; ovf_cnt = 0;
        vec_per_min = 1000000; vec_per_max = -1; vec_per_cnt = 0;
        t_lastprev  = -1;
        drive_stream(5);                       // 5 벡터 연속
        wait (last_cnt >= 5);                  // 5개 벡터의 out_last 모두 확인
        repeat (5) @(negedge clk);
        $display("   연속 5벡터 : out_last 간격 %0d ~ %0d clk (샘플 %0d개)",
                 vec_per_min, vec_per_max, vec_per_cnt);
        $display("   -> 벡터 처리주기 = %0d clk/vector", vec_per_min);
        $display("   -> throughput    = %0d elem / %0d clk = %.4f elem/clk",
                 N, vec_per_min, N * 1.0 / vec_per_min);
        $display("   -> 500MHz 기준     %.2f M vector/s,  %.1f M elem/s",
                 500.0 / vec_per_min, 500.0 * N / vec_per_min);
        // 구조상 벡터당 2(load+max)+16(exp)+4(rcp)+13(mul) = 35 clk  (P=8, 출력도 8-lane)
        pass_fail("F 벡터 처리주기 == 35 clk (설계값)", vec_per_min == 35);
        pass_fail("F 연속 벡터 간 주기 일정 (버블 없음)", vec_per_min == vec_per_max);

        // ================================================================
        //  종합
        // ================================================================
        $display("");
        $display("############################################################");
        $display("#  종합 결과");
        $display("############################################################");
        $display("  검증 벡터 수        : %0d vectors (%0d elements)", g_nvec, g_nelem);
        $display("  원소 최대 절대오차  : %.4e  (%.2f LSB of UQ1.15)  [기준 %.1e]",
                 g_maxerr, g_maxerr / LSB_OUT, TOL_ELEM);
        $display("  원소 전체 RMS 오차  : %.4e  (%.2f LSB of UQ1.15)  [기준 %.1e]",
                 $sqrt(g_sqsum / g_nelem), $sqrt(g_sqsum / g_nelem) / LSB_OUT, TOL_RMS);
        $display("  exp2_unit  최대오차 : %.4e (abs)   [기준 %.1e]", exp_max, TOL_EXP);
        $display("  recip_unit 최대오차 : %.4e (rel)   [기준 %.1e]", rcp_max, TOL_RCP);
        $display("  체크 항목           : %0d", checks);
        $display("------------------------------------------------------------");
        if (errors == 0) $display("  RESULT : *** ALL PASS ***");
        else             $display("  RESULT : *** %0d FAIL ***", errors);
        $display("############################################################");
        $display("");
        $finish;
    end

    // ---------------- 타임아웃 ----------------
    initial begin
        #200_000_000;
        $display("ERROR: simulation timeout");
        $finish;
    end
endmodule
