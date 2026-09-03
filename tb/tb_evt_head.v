// -----------------------------------------------------------------------------
// tb_evt_head : EvT_Engine 통합 검증 — **attention 앞단 명령어 7개**
//
// 엔진의 첫 관문입니다. attention 복잡도를 빼고 GEMM · Format_Cast_Act(GELU/ReLU) ·
// RES · LayerNorm 이 스케줄대로 도는지 봅니다.
//
//   0 GEMM event_projection  144→96   Q4.11→GELU→int8, **출력 stride 160**
//   1 GEMM preproc           160→128  Q4.11→GELU→int8
//   2 GEMM proc_ev.1         128→128  ReLU
//   3 GEMM proc_ev.4         128→128  ReLU
//   4 RES  proc_ev.res       + x_input
//   5 LN   layer_norm_x
//   6 LN   layer_norm_1
//
// 기대값은 골든의 **MAC 피연산자**입니다 (`sw/golden_insts.py`, README §5.5) —
// 어떤 GEMM 의 `a` 가 곧 그 GEMM 이 읽는 A_Mem 영역이라 나눗셈도 반올림도 없고,
// **어떤 명령어 의 출력이 다음 명령어 의 입력으로 검증**됩니다.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_evt_head;
  localparam N=32, AW_A=14, AW_W=14, AW_INST=8, DIM_W=16;
  localparam NTOK = 52, TT = 2;                 // 샘플0 t0 : 토큰 52 → 타일 2
  localparam R_X=0, R_PIDX=576, R_PIN=580, R_PRE=1220,
             R_EV1=1732, R_EV=2244, R_LATV=2756,
             R_Z=3140, R_LNX=3908, R_LN1=4420;
  localparam LAT_WORDS = 384;      // latent 3타일 x 128특징
  localparam W_WORDS = 7904, PB_WORDS = 869, PG_WORDS = 208, S_WORDS = 99;
  localparam POS_ROWS = 441;

  reg clk = 0, rst = 1; always #5 clk = ~clk;

  reg  [AW_INST-1:0] n_body = 8, n_tail = 1;
  reg  [5:0]      n_tstep = 1;
  reg  [31:0]     eps = 32'h3727c5ac;
  reg             start = 0;
  wire            done, busy;
  wire [3:0]      dbg_state;
  wire [AW_INST-1:0] dbg_inst;
  wire [5:0]      tstep_idx;
  reg  [DIM_W-1:0] tok_n = NTOK;
  reg             ld_we = 0;
  reg  [2:0]      ld_sel = 0;
  reg  [AW_W-1:0] ld_addr = 0;
  reg  [N*16-1:0] ld_data = 0;
  wire [3:0]      res_class;
  wire [10*32-1:0] res_logits;
  reg             dbg_rd_en = 0;
  reg  [AW_A-1:0] dbg_rd_addr = 0;
  wire [N*16-1:0] dbg_rd_data;

  EvT_Engine #(.N(N), .AW_A(AW_A), .AW_W(AW_W), .AW_INST(AW_INST)) dut (
    .clk(clk), .rst(rst), .start(start), .done(done), .busy(busy),
    .dbg_state(dbg_state), .dbg_inst(dbg_inst),
    .n_body(n_body), .n_tail(n_tail), .n_tstep(n_tstep), .eps(eps),
    // 입력이 이미 A_Mem 에 있으므로 요청은 즉시 승인합니다
    .tstep_idx(tstep_idx), .tok_n(tok_n),
    .tok_req(), .tok_ack(1'b1),
    .ld_we(ld_we), .ld_sel(ld_sel), .ld_addr(ld_addr), .ld_data(ld_data),
    .res_class(res_class), .res_logits(res_logits),
    .dbg_rd_en(dbg_rd_en), .dbg_rd_addr(dbg_rd_addr), .dbg_rd_data(dbg_rd_data));

  // ---- 메모리 이미지 ----
  reg [N*8-1:0]  wimg  [0:W_WORDS-1];
  reg [N*8-1:0]  rqimg [0:PB_WORDS-1];
  reg [N*8-1:0]  afimg [0:PG_WORDS-1];
  reg [N*8-1:0]  instimg  [0:S_WORDS-1];
  reg [N*16-1:0] ximg  [0:TT*144-1];
  reg [N*16-1:0] pimg  [0:TT*64-1];
  reg [N*16-1:0] qimg  [0:TT-1];          // pos_idx
  reg [63*8:0]   dummy;
  reg [64*8-1:0] posimg [0:POS_ROWS-1];   // PL 표 (441행 x 64B)
  reg [N*16-1:0] limg  [0:LAT_WORDS-1];   // memory_vertical → Z / LATV

  // ---- 골든 ----
  reg [N*16-1:0] g_pin [0:TT*160-1];
  reg [N*16-1:0] g_pre [0:TT*128-1];
  reg [N*16-1:0] g_ev1 [0:TT*128-1];
  reg [N*16-1:0] g_lnx [0:TT*128-1];
  reg [N*16-1:0] g_ln1 [0:3*128-1];
  reg [N*16-1:0] g_ev4 [0:TT*128-1];    // proc_ev.4 출력 int8 (ReLU 뒤)
  reg [N*16-1:0] g_ev  [0:TT*128-1];    // 잔차 뒤 bf16

  integer errors = 0, checks = 0, i, cyc, e0;
  integer n1 = 0, n3 = 0, nbig = 0, nx = 0;    // 틀린 것의 성격 분포
  reg fin;                       // done 으로 끝났는가 (사이클 초과가 아니라)

  task load(input [2:0] sel, input integer nw, input integer which);
    integer w;
    begin
      for (w = 0; w < nw; w = w + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = sel; ld_addr = w;
        case (which)
          0: ld_data = {{(N*8){1'b0}}, wimg[w]};
          1: ld_data = {{(N*8){1'b0}}, rqimg[w]};
          2: ld_data = {{(N*8){1'b0}}, afimg[w]};
          3: ld_data = {{(N*8){1'b0}}, instimg[w]};
        endcase
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  // A_Mem 은 주소를 직접 지정해 넣습니다 (X 와 PIN 뒤쪽)
  task load_a(input integer base, input integer nw, input integer which);
    integer w, mt, d;
    begin
      if (which == 0) begin                       // X : mt*144 + k
        for (w = 0; w < nw; w = w + 1) begin
          @(negedge clk); ld_we = 1; ld_sel = 3'd1;
          ld_addr = base + w; ld_data = ximg[w];
        end
      end else begin                              // POS : mt*160 + 96 + d
        for (mt = 0; mt < TT; mt = mt + 1)
          for (d = 0; d < 64; d = d + 1) begin
            @(negedge clk); ld_we = 1; ld_sel = 3'd1;
            ld_addr = base + mt*160 + 96 + d;
            ld_data = pimg[mt*64 + d];
          end
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  // PL 의 pos enc 표 (로더 sel 5)
  task load_pos_tbl;
    integer w;
    begin
      for (w = 0; w < POS_ROWS; w = w + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd5; ld_addr = w;
        ld_data = posimg[w];              // 둘 다 512비트
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  task load_pidx;
    integer w;
    begin
      for (w = 0; w < TT; w = w + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd1;
        ld_addr = R_PIDX + w; ld_data = qimg[w];
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  task load_lat(input integer base);
    integer w;
    begin
      for (w = 0; w < LAT_WORDS; w = w + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd1;
        ld_addr = base + w; ld_data = limg[w];
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  // `tol` : 허용 오차. 0 이 아닌 자리는 **골든과 하드웨어의 포맷이 다른 곳**이고
  // 그 이유가 README §5 에 적혀 있습니다. 0 인 자리는 1 LSB 도 못 틀립니다.
  task chk(input [255:0] nm, input integer base, input integer nfeat,
           input integer nrow, input integer which, input integer tol);
    integer mt, k, lane, TTx;
    reg signed [15:0] got, exp;
    begin
      e0 = errors; TTx = (nrow + 31) / 32;
      for (mt = 0; mt < TTx; mt = mt + 1)
        for (k = 0; k < nfeat; k = k + 1)
          for (lane = 0; lane < N; lane = lane + 1) begin
            if (mt*32 + lane >= nrow) continue;
            checks = checks + 1;
            got = dut.u_a_mem0.mem[base + mt*nfeat + k][lane*16 +: 16];
            case (which)
              0: exp = g_pin[mt*nfeat + k][lane*16 +: 16];
              1: exp = g_pre[mt*nfeat + k][lane*16 +: 16];
              2: exp = g_ev1[mt*nfeat + k][lane*16 +: 16];
              3: exp = g_lnx[mt*nfeat + k][lane*16 +: 16];
              5: exp = g_ev4[mt*nfeat + k][lane*16 +: 16];
              6: exp = g_ev [mt*nfeat + k][lane*16 +: 16];
              default: exp = g_ln1[mt*nfeat + k][lane*16 +: 16];
            endcase
            if (got !== exp) begin
              if (got === 16'hxxxx) begin nx = nx + 1; errors = errors + 1; end
              else if (got - exp <= tol && exp - got <= tol) n1 = n1 + 1;
              else begin nbig = nbig + 1; errors = errors + 1; end
              if (errors - e0 < 6)
                $display("    [FAIL] %0s mt=%0d k=%0d lane=%0d  got %0d(%04x)  exp %0d(%04x)",
                         nm, mt, k, lane, got, got, exp, exp);
            end
          end
      $display("  %-24s %0d/%0d 일치  (허용 %0d 이내 %0d개, 초과 %0d개, X %0d개)",
               nm, (TTx*nfeat*N) - (errors - e0) - n1, TTx*nfeat*N,
               tol, n1, nbig, nx);
      n1=0; n3=0; nbig=0; nx=0;
    end
  endtask

  // RES 가 R_EV 를 덮어쓰므로 명령어 3 이 끝난 순간을 떠 둡니다
  reg [N*16-1:0] snap_ev [0:TT*128-1];
  reg snapped = 0;
  integer sv;
  always @(posedge clk) if (!rst && dbg_inst == 8'd5 && !snapped) begin
    snapped <= 1'b1;
    for (sv = 0; sv < TT*128; sv = sv + 1)
      snap_ev[sv] = dut.u_a_mem0.mem[R_EV + sv];
  end

  task chk_snap(input [255:0] nm, input integer nfeat, input integer nrow);
    integer mt, k, lane, TTx;
    reg signed [15:0] got, exp;
    begin
      e0 = errors; TTx = (nrow + 31) / 32;
      for (mt = 0; mt < TTx; mt = mt + 1)
        for (k = 0; k < nfeat; k = k + 1)
          for (lane = 0; lane < N; lane = lane + 1) begin
            if (mt*32 + lane >= nrow) continue;
            checks = checks + 1;
            got = snap_ev[mt*nfeat + k][lane*16 +: 16];
            exp = g_ev4 [mt*nfeat + k][lane*16 +: 16];
            if (got !== exp) begin
              errors = errors + 1;
              if (errors - e0 < 4)
                $display("    [FAIL] %0s mt=%0d k=%0d lane=%0d  got %0d  exp %0d",
                         nm, mt, k, lane, got, exp);
            end
          end
      $display("  %-22s %0d/%0d 일치", nm,
               (TTx*nfeat*N) - (errors - e0), TTx*nfeat*N);
    end
  endtask

  // ---- 명령어 별 계수 : 어디서 끊겼는지 바로 짚습니다 (fpga_nl 에서 유효했던 방법)
  integer n_colv=0, n_cpv=0, n_awe=0, n_lnov=0, n_gmd=0;
  reg [7:0] prev_inst = 8'hFF;
  always @(posedge clk) if (!rst && dut.busy) begin
    if (dut.col_valid)   n_colv = n_colv + 1;
    if (dut.fca_valid)    n_cpv  = n_cpv  + 1;
    if (dut.a_we_en) n_awe  = n_awe  + 1;
    if (dut.ln_valid)   n_lnov = n_lnov + 1;
    if (dbg_inst != prev_inst) begin
      if (prev_inst != 8'hFF)
        $display("  inst %0d 종료: col_valid=%0d fca_valid=%0d a_we=%0d ln_valid=%0d",
                 prev_inst, n_colv, n_cpv, n_awe, n_lnov);
      n_colv=0; n_cpv=0; n_awe=0; n_lnov=0;
      prev_inst = dbg_inst;
    end
  end

  // 명령어 진입 시 디코드된 상수 (스텝 워드가 의도대로 풀렸는지)
  always @(posedge clk) if (!rst && dut.state == 4'd3 && dut.const_ph == 2'd2) begin
    $display("  DEC inst=%0d kind=%0d fmt=%0d act=%0d flag=%0d/%0d shift=%0d shift2=%0d",
             dbg_inst, dut.op_kind, dut.op_fmt, dut.op_act, dut.op_flag, dut.op_flag2,
             dut.op_shift, dut.op_shift2);
    $display("      M=%0d K=%0d NOUT=%0d AIN=%0d BIN=%0d AOUT=%0d RQ_BASE=%0d OSTR=%0d gRQ=%0d",
             dut.op_m, dut.op_k, dut.op_nout, dut.op_ain, dut.op_bin, dut.op_aout,
             dut.op_rq_base, dut.op_ostr, dut.rq_idx);
  end

  integer gp = 0;
  always @(posedge clk) if (!rst && dut.u_pos.we_en && gp < 6) begin
    $display("  POSW#%0d addr=%0d d0=%04x  row_tile=%0d feat=%0d ntok=%0d idx0=%0d tbl_q0=%02x", gp, dut.u_pos.we_addr, dut.u_pos.we_data[15:0], dut.u_pos.row_tile, dut.u_pos.feat, dut.u_pos.n_tok, dut.u_pos.idx_word[15:0], dut.u_pos.pos_tbl_q[7:0]);
    gp = gp + 1;
  end
  integer gr = 0;
  always @(posedge clk) if (!rst && dut.u_pos.state == 3'd2 && dut.u_pos.ph == 2'd2 && gr < 4) begin
    $display("  POSR#%0d lane=%0d pos_tbl_addr=%0d pos_tbl_q[7:0]=%02x  row_buf 저장", gr, dut.u_pos.lane, dut.u_pos.pos_tbl_addr, dut.u_pos.pos_tbl_q[7:0]);
    gr = gr + 1;
  end

  integer pp = 0;
  always @(posedge clk) if (!rst && dut.busy && dut.col_valid && pp < 4) begin
    $display("  COL#%0d inst=%0d n=%0d acc0=%08x | rq_scale=%08x rq_bias=%08x rq_idx=%0d",
             pp, dbg_inst, dut.col_n, dut.col_data[31:0],
             dut.rq_scale, dut.rq_bias, dut.rq_idx);
    $display("        a_rd=%0d/%04x  b_rd=%0d/%02x  op_ain=%0d op_bin=%0d op_k=%0d",
             dut.a_ra_addr, dut.a_ra_data[15:0], dut.gemm_b_rd_addr, dut.gemm_b_data[7:0],
             dut.op_ain, dut.op_bin, dut.op_k);
    pp = pp + 1;
  end

  integer wp = 0;
  always @(posedge clk) if (!rst && dut.busy && dut.a_we_en && wp < 6) begin
    $display("  WRITE#%0d inst=%0d addr=%0d  AOUT=%0d OSTR=%0d fca_mt=%0d fca_n=%0d  d0=%04x",
             wp, dbg_inst, dut.a_we_addr, dut.op_aout, dut.op_ostr,
             dut.fca_mt, dut.fca_n, dut.a_we_data[15:0]);
    wp = wp + 1;
  end

  initial begin
    $display("[tb_evt_head] EvT_Engine 앞단 명령어 7개  (샘플0 t0, 토큰 %0d)", NTOK);
    $readmemh("../data/wmem.hex",  wimg);
    $readmemh("../data/rqmem.hex", rqimg);
    $readmemh("../data/afmem.hex", afimg);
    $readmemh("../data/instmem.hex", instimg);
    $readmemh("../data/board/t0_x.hex",   ximg);
    $readmemh("../data/board/t0_pidx.hex", qimg);
    $readmemh("../data/posmem.hex", posimg);
    $readmemh("../data/latinit.hex", limg);
    $readmemh("../data/golden/PIN.hex",       g_pin);
    $readmemh("../data/golden/proc_ev.1.hex", g_pre);
    $readmemh("../data/golden/proc_ev.4.hex", g_ev1);
    $readmemh("../data/golden/in_proj.K.hex", g_lnx);
    $readmemh("../data/golden/in_proj.Q.hex", g_ln1);
    $readmemh("../data/golden/EV4.hex", g_ev4);
    $readmemh("../data/golden/EV.hex",  g_ev);

    repeat (8) @(posedge clk); rst = 0; @(posedge clk);

    $display("-- 메모리 적재");
    load(3'd0, W_WORDS,  0);  load(3'd2, PB_WORDS, 1);
    load(3'd3, PG_WORDS, 2);  load(3'd4, S_WORDS,  3);
    load_a(R_X,   TT*144, 0);
    load_pos_tbl();  load_pidx();      // pos enc 는 PL 표에서 모읍니다
    // latent 초기값 : `inp_q`(Z) 와 `latent_vectors`(LATV) 가 같은 값에서 시작
    load_lat(R_Z); load_lat(R_LATV);

    $display("-- 실행 (inst 0..7)");
    @(negedge clk); start = 1;
    cyc = 0; fin = 1'b0;
    while (!fin && cyc < 3000000) begin
      @(posedge clk); cyc = cyc + 1;
      if (done) fin = 1'b1;      // **여기서 잡아 둡니다**
    end
    // start 를 내리면 엔진이 IDLE 로 가며 done 을 내립니다. 그 뒤에 done 을
    // 다시 보면 항상 0 이라 정상 종료를 TIMEOUT 으로 오인합니다.
    @(negedge clk); start = 0;
    repeat (8) @(posedge clk);
    if (!fin) begin
      $display("=== TIMEOUT  state=%0d inst=%0d ===", dbg_state, dbg_inst);
      $finish;
    end
    $display("     %0d 사이클", cyc);

    $display("-- LNX k=0 앞 8레인 (하드웨어 / 골든)");
    for (i = 0; i < 8; i = i + 1)
      $display("     lane %0d : %0d / %0d", i,
               $signed(dut.u_a_mem0.mem[R_LNX][i*16 +: 16]),
               $signed(g_lnx[0][i*16 +: 16]));
    $display("-- LNX mt=1 k=0 앞 6레인 (하드웨어 / 골든)");
    for (i = 0; i < 6; i = i + 1)
      $display("     lane %0d : %0d / %0d", i,
               $signed(dut.u_a_mem0.mem[R_LNX + 128][i*16 +: 16]),
               $signed(g_lnx[128][i*16 +: 16]));
    $display("-- LN1 k=0 앞 8레인 (하드웨어 / 골든)");
    for (i = 0; i < 8; i = i + 1)
      $display("     lane %0d : %0d / %0d", i,
               $signed(dut.u_a_mem0.mem[R_LN1][i*16 +: 16]),
               $signed(g_ln1[0][i*16 +: 16]));

    $display("-- A_Mem 대조 (기대값 = 골든 MAC 피연산자)");
    chk("PIN  event_projection", R_PIN, 160, NTOK, 0, 0);
    chk("PRE  preproc",          R_PRE, 128, NTOK, 1, 0);
    chk("EV1  proc_ev.1",        R_EV1, 128, NTOK, 2, 0);
    chk_snap("EV4  proc_ev.4 (RES 전)", 128, NTOK);
    // bf16 합의 반올림 — 골든은 fp32 로 더한 뒤 bf16, 하드웨어는 정수합 뒤 bf16.
    // 정확히 반이 되는 값에서만 1 ulp 갈립니다.
    chk("EV   +x (bf16)",        R_EV,  128, NTOK, 6, 1);
    // 새 LayerNorm 코어는 **알고리즘이 다릅니다** (정수 분산 + rsqrt LUT vs
    // 예전 bf16 누산). 골든과 비트 일치가 원리상 안 되고 실측 편차가 ±1~2 라
    // 그만큼 허용합니다. 앞단(PIN/PRE/EV1)은 여전히 tol 0 입니다.
    chk("LNX  layer_norm_x",     R_LNX, 128, NTOK, 3, 2);
    // 골든은 latent 를 LayerNorm 앞에서 **Q16.-1(lsb 2.0)** 격자에 스냅합니다
    // (20 타임스텝 누적 범위로 잡힌 격자). 하드웨어는 bf16 이라 **더 정밀**하고,
    // 그래서 t=0 처럼 값이 작을 때 int8 출력이 1 LSB 갈립니다. README §5 ⑥.
    chk("LN1  layer_norm_1",     R_LN1, 128, 96,   4, 1);

    if (errors == 0) $display("=== tb_evt_head: %0d checks, TEST PASSED ===", checks);
    else             $display("=== tb_evt_head: %0d/%0d failed, TEST FAILED ===", errors, checks);
    $finish;
  end
endmodule
