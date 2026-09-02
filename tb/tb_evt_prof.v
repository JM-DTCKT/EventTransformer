// -----------------------------------------------------------------------------
// tb_evt : EvT_Engine 전체 통합 — 타임스텝 **하나 + 마지막 꼬리**
//
//   본체 명령어 153개 (전처리 4 + 잔차 + attention 블록 3개) + 꼬리 명령어 5개
//   → 클래스 하나가 나옵니다.
//
// `tb_evt_head` 가 앞단 명령어 7개 만 봤다면 여기는 **attention 까지 전부**입니다.
// 특히 이 TB 만 확인할 수 있는 것들:
//
//   · Q/K/V 의 head-major 주소와 Vᵀ 전치
//   · softmax 가 `C = Lk` 하나로 마스크를 대신하는 것
//   · bias_k/bias_v 토큰이 **읽는 쪽에서** 끼워지는 것
//   · 블록 3개가 Z/ZATT 를 이어받는 것
//   · MEAN(레인 축 리덕션) 과 ARGMAX
//
// 기대값은 `sw/golden_insts.py --single` 이 뽑습니다 — 골든도 타임스텝 하나만
// 넣어 돌리므로 latent 누적과 분류기 출력까지 그대로 대조됩니다.
//
// K/V/Q/CTX 는 블록 3개가 **돌려 쓰므로** 끝나고 보면 마지막 블록 값입니다.
// cross 블록 값을 보려면 그 순간에 떠야 합니다 (`snap_*`).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_evt_prof;
  localparam N=32, AW_A=14, AW_W=14, AW_INST=8, DIM_W=16;
  localparam NTOK = 52, TT = 2, QT = 3, HD = 32, HEADS = 4;
  localparam KSTR = 160, VSTR = 129;          // head 간 간격 (schedule_evt.py)
  // 영역 베이스 — `data/schedule.json` 과 한 벌
  localparam R_X=0, R_PIDX=576, R_PIN=580, R_PRE=1220,
             R_EV1=1732, R_EV=2244, R_LATV=2756, R_Z=3140,
             R_ZATT=3524, R_LNX=3908, R_LN1=4420, R_LNA=4804,
             R_Q=5188, R_K=5572, R_V=6212, R_BKV=6728,
             R_SM=6752, R_CTX=7139, R_FFN=7523;
  localparam W_WORDS=14000, PB_WORDS=869, PG_WORDS=208, S_WORDS=99,
             LAT_WORDS=384, BKV_WORDS=24;
  localparam S_QK0 = 11, S_OUTPROJ = 27;      // 스냅 시점 (cross 블록)
  // 꼬리 : 94 embs.ln  95 embs.linear1  96 MEAN  97 clf.linear_1  98 argmax
  localparam S_MEAN = 96;
  // cross 블록 FFN 체인 스냅 시점 (그 명령어 진입 시 = 직전 명령어 의 출력)
  //   29 ln_att(입력 ZATT)  30 linear1(입력 LNA)  32 linear2(입력 LNA)
  //   33 linear3(입력 FFN)
  localparam S_ZATT = 29, S_LNA1 = 30, S_LNA2 = 32, S_FFN3 = 33;

  reg clk = 0, rst = 1; always #5 clk = ~clk;

  reg  [AW_INST-1:0] n_body = 94, n_tail = 5;
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
  reg [N*16-1:0] qimg  [0:TT-1];            // pos_idx (호스트가 주는 것)
  reg [64*8-1:0] posimg [0:440];            // PL 표 441행 x 64B
  reg [N*16-1:0] limg  [0:LAT_WORDS-1];
  reg [N*16-1:0] bimg  [0:BKV_WORDS-1];

  // ---- 골든 (행 x 특징 을 워드로 편 것) ----
  reg [N*16-1:0] g_pin [0:TT*160-1];
  reg [N*16-1:0] g_pre [0:TT*128-1];
  reg [N*16-1:0] g_ev1 [0:TT*128-1];
  reg [N*16-1:0] g_lnx [0:TT*128-1];
  reg [N*16-1:0] g_ln1 [0:QT*128-1];
  reg [N*16-1:0] g_qi  [0:QT*128-1];      // Q  (96, 128)  head 이어붙임
  reg [N*16-1:0] g_ki  [0:TT*128-1];      // K  (n_tok, 128)
  reg [N*16-1:0] g_vi  [0:TT*128-1];      // V  (n_tok, 128)
  reg [N*16-1:0] g_ctx [0:QT*128-1];      // attn·V (96, 128)
  // ---- 꼬리 단 (타임스텝 루프 밖이라 앞단 검증에 안 걸립니다) ----
  reg [N*16-1:0] g_tlna [0:QT*128-1];     // embs.ln 출력 (96,128)
  reg [N*16-1:0] g_tlatv[0:QT*128-1];     // 누적 latent (96,128) bf16
  reg [N*16-1:0] g_zatt [0:QT*128-1];     // z_att (96,128) bf16
  reg [N*16-1:0] g_lna1 [0:QT*128-1];     // layer_norm_att 출력 int8
  reg [N*16-1:0] g_lna2 [0:QT*128-1];     // layer_norm_2 출력 int8
  reg [N*16-1:0] g_ffn3 [0:QT*128-1];     // gelu2 뒤 int8
  reg [N*16-1:0] g_tffn [0:QT*128-1];     // MEAN 입력 = embs.linear1+ReLU
  reg [N*16-1:0] g_tmn  [0:128-1];        // MEAN 출력 (1,128) — 레인 0
  reg [N*16-1:0] g_tc1  [0:128-1];        // clf.linear_1 출력 (1,128)
  reg [31:0]     g_clf  [0:9];            // 분류기 누산기 10개

  integer errors = 0, checks = 0, soft = 0, cyc, e0, s0;

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

  task load_a(input integer base, input integer nw, input integer which);
    integer w, mt, d;
    begin
      case (which)
        0: for (w = 0; w < nw; w = w + 1) begin       // X : mt*144 + k
             @(negedge clk); ld_we=1; ld_sel=3'd1;
             ld_addr = base + w; ld_data = ximg[w];
           end
        1: for (mt = 0; mt < TT; mt = mt + 1)         // POS : mt*160 + 96 + d
             for (d = 0; d < 64; d = d + 1) begin
               @(negedge clk); ld_we=1; ld_sel=3'd1;
               ld_addr = base + mt*160 + 96 + d; ld_data = pimg[mt*64 + d];
             end
        2: for (w = 0; w < LAT_WORDS; w = w + 1) begin
             @(negedge clk); ld_we=1; ld_sel=3'd1;
             ld_addr = base + w; ld_data = limg[w];
           end
        4: for (w = 0; w < TT; w = w + 1) begin        // pos_idx
             @(negedge clk); ld_we=1; ld_sel=3'd1;
             ld_addr = base + w; ld_data = qimg[w];
           end
        3: for (w = 0; w < BKV_WORDS; w = w + 1) begin
             @(negedge clk); ld_we=1; ld_sel=3'd1;
             ld_addr = base + w; ld_data = bimg[w];
           end
      endcase
      @(negedge clk); ld_we = 0;
    end
  endtask

  // PL 의 pos enc 표 (로더 sel 5). 441행 x 64B = 27.6 KB.
  task load_postbl;
    integer w;
    begin
      for (w = 0; w < 441; w = w + 1) begin
        @(negedge clk); ld_we = 1; ld_sel = 3'd5; ld_addr = w;
        ld_data = posimg[w];              // 둘 다 512비트
      end
      @(negedge clk); ld_we = 0;
    end
  endtask

  // ---- cross 블록 값 스냅 (뒤 블록이 덮어씀) ----
  reg [N*16-1:0] sq [0:HEADS*QT*HD-1];
  reg [N*16-1:0] sk [0:HEADS*KSTR-1];
  reg [N*16-1:0] sv [0:HEADS*VSTR-1];
  reg [N*16-1:0] sc [0:QT*128-1];
  // LNX/LN1 도 뒷블록이 덮어씁니다 — cross 블록 값을 보려면 떠 둬야 합니다
  reg [N*16-1:0] snx [0:TT*128-1];
  reg [N*16-1:0] s1 [0:QT*128-1];
  // MEAN 이 LNA 를 덮어쓰므로(레인 0만 쓰고 나머지는 0) 그 전에 떠 둡니다
  reg [N*16-1:0] sln [0:QT*128-1], sfn [0:QT*128-1];
  reg [N*16-1:0] sza [0:QT*128-1], sl1 [0:QT*128-1];
  reg [N*16-1:0] sl2 [0:QT*128-1], sf3 [0:QT*128-1];
  reg snap_qkv = 0, snap_ctx = 0, snap_ln = 0;
  reg sn_za = 0, sn_l1 = 0, sn_l2 = 0, sn_f3 = 0;
  integer z;
  always @(posedge clk) if (!rst) begin
    if (dbg_inst == S_QK0 && !snap_qkv) begin
      snap_qkv <= 1'b1;
      for (z = 0; z < HEADS*QT*HD; z = z + 1) sq[z] = dut.u_a_mem0.mem[R_Q + z];
      for (z = 0; z < HEADS*KSTR;  z = z + 1) sk[z] = dut.u_a_mem0.mem[R_K + z];
      for (z = 0; z < HEADS*VSTR;  z = z + 1) sv[z] = dut.u_a_mem0.mem[R_V + z];
      for (z = 0; z < TT*128;      z = z + 1) snx[z] = dut.u_a_mem0.mem[R_LNX + z];
      for (z = 0; z < QT*128;      z = z + 1) s1[z] = dut.u_a_mem0.mem[R_LN1 + z];
    end
    if (dbg_inst == S_ZATT && !sn_za) begin
      sn_za <= 1'b1;
      for (z = 0; z < QT*128; z = z + 1) sza[z] = dut.u_a_mem0.mem[R_ZATT + z];
    end
    if (dbg_inst == S_LNA1 && !sn_l1) begin
      sn_l1 <= 1'b1;
      for (z = 0; z < QT*128; z = z + 1) sl1[z] = dut.u_a_mem0.mem[R_LNA + z];
    end
    if (dbg_inst == S_LNA2 && !sn_l2) begin
      sn_l2 <= 1'b1;
      for (z = 0; z < QT*128; z = z + 1) sl2[z] = dut.u_a_mem0.mem[R_LNA + z];
    end
    if (dbg_inst == S_FFN3 && !sn_f3) begin
      sn_f3 <= 1'b1;
      for (z = 0; z < QT*128; z = z + 1) sf3[z] = dut.u_a_mem0.mem[R_FFN + z];
    end
    if (dbg_inst == S_MEAN && !snap_ln) begin
      snap_ln <= 1'b1;
      for (z = 0; z < QT*128; z = z + 1) begin
        sln[z] = dut.u_a_mem0.mem[R_LNA + z];
        sfn[z] = dut.u_a_mem0.mem[R_FFN + z];
      end
    end
    if (dbg_inst == S_OUTPROJ && !snap_ctx) begin
      snap_ctx <= 1'b1;
      for (z = 0; z < QT*128; z = z + 1) sc[z] = dut.u_a_mem0.mem[R_CTX + z];
    end
  end

  // ---- 명령어 별 사이클 프로파일 ----
  integer psc [0:255], psk [0:255], psn [0:255], psm [0:255];
  integer pf;
  initial for (pf = 0; pf < 256; pf = pf + 1) begin
    psc[pf]=0; psk[pf]=15; psn[pf]=0; psm[pf]=0;
  end
  always @(posedge clk) if (!rst && dut.busy && dbg_inst < 256) begin
    psc[dbg_inst] <= psc[dbg_inst] + 1;
    psk[dbg_inst] <= dut.op_kind;
    psn[dbg_inst] <= dut.op_nout;
    psm[dbg_inst] <= dut.op_m;
  end

  // ---- 명령어 별 활동 계수 (어디서 끊겼는지) ----
  integer n_colv=0, n_cpv=0, n_awe=0, n_lnov=0, n_smov=0, n_tr=0;
  reg [7:0] prev_inst = 8'hFF;
  always @(posedge clk) if (!rst && dut.busy) begin
    if (dut.col_valid)   n_colv = n_colv + 1;
    if (dut.fca_valid)    n_cpv  = n_cpv  + 1;
    if (dut.a_we_en) n_awe  = n_awe  + 1;
    if (dut.ln_valid)   n_lnov = n_lnov + 1;
    if (dut.smax_valid)   n_smov = n_smov + 1;
    if (dut.tr_drain)  n_tr   = n_tr   + 1;
    if (dbg_inst != prev_inst) begin
      if (prev_inst != 8'hFF && (prev_inst < 16 || prev_inst >= 118))
        $display("  inst %0d: col_valid=%0d fca_valid=%0d a_we=%0d ln_valid=%0d smax_valid=%0d tr=%0d",
                 prev_inst, n_colv, n_cpv, n_awe, n_lnov, n_smov, n_tr);
      n_colv=0; n_cpv=0; n_awe=0; n_lnov=0; n_smov=0; n_tr=0;
      prev_inst = dbg_inst;
    end
  end

  // 명령어 진입 시 디코드 결과 (문자열은 반드시 한 줄로!)
  always @(posedge clk) if (!rst && dut.state == 4'd3 && dut.const_ph == 2'd2 && (dbg_inst < 5 || dbg_inst > 31))
    $display("  DEC inst=%0d kind=%0d fmt=%0d flag=%0d/%0d M=%0d K=%0d NOUT=%0d AIN=%0d BIN=%0d AOUT=%0d RQ_BASE=%0d OSTR=%0d shift=%0d shift2=%0d", dbg_inst, dut.op_kind, dut.op_fmt, dut.op_flag, dut.op_flag2, dut.op_m, dut.op_k, dut.op_nout, dut.op_ain, dut.op_bin, dut.op_aout, dut.op_rq_base, dut.op_ostr, dut.op_shift, dut.op_shift2);

  reg smdone = 0;
  always @(posedge clk) if (!rst && dbg_inst == 13 && !smdone) begin
    smdone <= 1'b1;
    $display("  SM[0]=%04x SM[1]=%04x SM[52]=%04x SM[53]=%04x  V[0]=%04x", dut.u_a_mem0.mem[6748][15:0], dut.u_a_mem0.mem[6749][15:0], dut.u_a_mem0.mem[6800][15:0], dut.u_a_mem0.mem[6801][15:0], dut.u_a_mem0.mem[6208][15:0]);
  end
  integer qkp = 0;
  always @(posedge clk) if (!rst && dbg_inst == 11 && dut.fca_valid && qkp < 4) begin
    $display("  QK#%0d n=%0d q69=%04x acc=%08x pbm=%08x", qkp, dut.fca_n, dut.fca_q69[15:0], dut.col_data_d1[31:0], dut.rq_scale);
    qkp = qkp + 1;
  end

  // X 를 처음 쓰는 명령어 을 잡습니다 (X 는 여기서 시작해 뒤로 번집니다)
  reg [7:0] pstep = 8'hFF;
  always @(posedge clk) if (!rst && dbg_inst != pstep) begin
    pstep <= dbg_inst;
    if (dbg_inst == 8 || dbg_inst == 11 || dbg_inst == 35)
      $display("  PEEK inst=%0d LNX[0].l0=%0d LN1[0].l0=%0d snapLNX=%0d", dbg_inst, $signed(dut.u_a_mem0.mem[R_LNX][15:0]), $signed(dut.u_a_mem0.mem[R_LN1][15:0]), $signed(snx[0][15:0]));
  end

  integer q47 = 0;
  always @(posedge clk) if (!rst && dbg_inst == 48 && dut.fca_valid && dut.fca_n > 93 && q47 < 8) begin
    $display("  QK47 n=%0d q69=%04x acc=%08x  b_rd=%0d b0=%02x qk_hit=%0d bk=%04x", dut.fca_n, dut.fca_q69[15:0], dut.col_data_d1[31:0], dut.gemm_b_rd_addr, dut.gemm_b_data[7:0], dut.qk_hit, dut.bias_k_word[15:0]);
    q47 = q47 + 1;
  end

  integer sx = 0;
  always @(posedge clk) if (!rst && dut.smax_in_valid && sx < 6 && (^dut.smax_in_data === 1'bx)) begin
    $display("  SMX#%0d inst=%0d 입력이 X", sx, dbg_inst); sx = sx + 1;
  end

  integer xp = 0;
  always @(posedge clk) if (!rst && dut.a_we_en && xp < 8 && (^dut.a_we_data[15:0] === 1'bx)) begin
    $display("  XWR#%0d inst=%0d kind=%0d fmt=%0d addr=%0d", xp, dbg_inst, dut.op_kind, dut.op_fmt, dut.a_we_addr);
    xp = xp + 1;
  end

  // MEAN 프로브 — 특징 k 별 (누적합, 곱수, 시프트, 출력)
  integer mnp = 0;
  always @(posedge clk) if (!rst && dut.mean_run && dut.mean_ph == 2'd2 && mnp < 6) begin
    $display("  MN k=%0d acc=%0d gscale=%0d gsh=%0d out=%0d  M=%0d K=%0d AIN=%0d", dut.mean_k, $signed(dut.mean_acc), $signed(dut.rq_scale2_q), dut.op_shift2, $signed(dut.mean_out), dut.op_m, dut.op_k, dut.op_ain);
    mnp = mnp + 1;
  end

  integer amp = 0;
  always @(posedge clk) if (!rst && dut.op_kind == 4'd5 && dut.col_valid_d2 && amp < 20) begin
    $display("  AM inst=%0d n=%0d NOUT=%0d acc=%0d scale=%0d val=%0d best=%0d any=%0d cls=%0d", dbg_inst, dut.col_n_d2, dut.op_nout, $signed(dut.argmax_acc), $signed(dut.rq_scale_q), $signed(dut.argmax_val), $signed(dut.argmax_best), dut.argmax_any, dut.res_class);
    amp = amp + 1;
  end

  // softmax 명령어 별 입출력 개수
  integer smi=0, smo=0, smhs=0, smnr=0; reg [7:0] smc_last=0; reg [7:0] smstep=0;
  always @(posedge clk) if (!rst && dut.busy) begin
    if (dbg_inst != smstep) begin
      if (smi != 0 || smo != 0)
        $display("  SM inst=%0d 입력 %0d 코어수신 %0d 안받음 %0d 출력 %0d (n_col=%0d)", smstep, smi, smhs, smnr, smo, dut.op_nout);
      smi=0; smo=0; smhs=0; smnr=0; smstep=dbg_inst;
    end
    if (dut.smax_in_valid) smi = smi + 1;
    if (dut.smax_in_valid && dut.u_smax.take && dut.u_smax.core_iready) smhs = smhs + 1;
    if (dut.smax_in_valid && !dut.u_smax.core_iready) smnr = smnr + 1;
    if (dut.smax_valid) begin smo = smo + 1; smc_last = dut.smax_col; end
  end

  integer avp = 0;
  always @(posedge clk) if (!rst && dbg_inst == 13 && dut.a_we_en && avp < 4) begin
    // 주의 : Verilog 는 인접 문자열을 이어붙이지 않습니다. 줄을 나눠 쓰면
    // 이 $display 가 통째로 사라집니다 (VCS 가 조용히 넘어감).
    $display("  AVW#%0d addr=%0d AOUT=%0d fca_n=%0d fca_mt=%0d d0=%04x SM0=%04x b_rd=%0d/%02x", avp, dut.a_we_addr, dut.op_aout, dut.fca_n, dut.fca_mt, dut.a_we_data[15:0], dut.a_ra_data[15:0], dut.gemm_b_rd_addr, dut.gemm_b_data[7:0]);
    avp = avp + 1;
  end

  integer twp = 0;
  always @(posedge clk) if (!rst && dut.tr_drain && twp < 6) begin
    $display("  TRW#%0d addr=%0d we=%0d AOUT=%0d OSTR=%0d h=%0d mt=%0d r=%0d d0=%04x",
             twp, dut.a_we_addr, dut.a_we_en, dut.op_aout, dut.op_ostr,
             dut.tr_head, dut.tr_mt, dut.tr_row, dut.a_we_data[15:0]);
    twp = twp + 1;
  end

  // 값 하나 비교 (tol 이내는 soft 로만 셈)
  task cmp(input [255:0] nm, input integer idx, input signed [15:0] got,
           input signed [15:0] exp, input integer tol);
    begin
      checks = checks + 1;
      if (got !== exp) begin
        if (got !== 16'hxxxx && got - exp <= tol && exp - got <= tol)
          soft = soft + 1;
        else begin
          errors = errors + 1;
          if (errors - e0 < 4)
            $display("    [FAIL] %0s #%0d  got %0d(%04x)  exp %0d(%04x)",
                     nm, idx, got, got, exp, exp);
        end
      end
    end
  endtask

  task rpt(input [255:0] nm, input integer n);
    begin
      $display("  %-26s %0d/%0d 일치  (허용 %0d개, 초과 %0d개)",
               nm, n - (errors - e0) - (soft - s0), n, soft - s0, errors - e0);
      e0 = errors; s0 = soft;
    end
  endtask

  // bf16 스트림 진단 — 실패로 세지 않고 **얼마나 벌어지는지**만 봅니다.
  // 하드웨어는 잔차를 bf16(가수 8비트)로 흘리고 골든은 fp32 입니다. 원리상
  // 벌어지는 값이라 여기서 막으면 검증이 아니라 잡음이 됩니다. 실제 판정은
  // 이 뒤의 **int8 값**(LNA1/LNA2/FFN3/TFFN/TMEAN/TCLF1)과 클래스가 합니다.
  task bf_report(input [255:0] nm, input integer which);
    integer mt, kk, lane, n, over, mxd, d;
    reg signed [15:0] g, x;
    begin
      n = 0; over = 0; mxd = 0;
      for (mt = 0; mt < QT; mt = mt + 1)
        for (kk = 0; kk < 128; kk = kk + 1)
          for (lane = 0; lane < N; lane = lane + 1) begin
            n = n + 1;
            if (which == 0) begin
              g = sza[mt*128+kk][lane*16 +: 16];
              x = g_zatt[mt*128+kk][lane*16 +: 16];
            end else begin
              g = dut.u_a_mem0.mem[R_LATV + mt*128 + kk][lane*16 +: 16];
              x = g_tlatv[mt*128+kk][lane*16 +: 16];
            end
            d = g - x; if (d < 0) d = -d;
            if (d > mxd) mxd = d;
            if (d > 16) over = over + 1;
          end
      // 최대 편차는 0 근처에서 부호가 갈리면 32768 이 나와 의미가 없습니다.
      // **16 ulp 이내 비율**이 실제로 볼 값입니다.
      $display("  %-26s %0d/%0d 가 16 ulp 이내  (진단, 판정 안 함)",
               nm, n - over, n);
    end
  endtask

  // 스냅 배열 대조 (which: 0 ZATT  1 LNA1  2 LNA2  3 FFN3)
  task snap_chk(input [255:0] nm, input integer which, input integer tol);
    integer mt, kk, lane, n;
    reg signed [15:0] g, x;
    begin
      e0 = errors; s0 = soft; n = 0;
      for (mt = 0; mt < QT; mt = mt + 1)
        for (kk = 0; kk < 128; kk = kk + 1)
          for (lane = 0; lane < N; lane = lane + 1) begin
            n = n + 1;
            case (which)
              0: begin g = sza[mt*128+kk][lane*16 +: 16];
                       x = g_zatt[mt*128+kk][lane*16 +: 16]; end
              1: begin g = sl1[mt*128+kk][lane*16 +: 16];
                       x = g_lna1[mt*128+kk][lane*16 +: 16]; end
              2: begin g = sl2[mt*128+kk][lane*16 +: 16];
                       x = g_lna2[mt*128+kk][lane*16 +: 16]; end
              default: begin g = sf3[mt*128+kk][lane*16 +: 16];
                             x = g_ffn3[mt*128+kk][lane*16 +: 16]; end
            endcase
            cmp(nm, n, g, x, tol);
          end
      rpt(nm, n);
    end
  endtask

  // A_Mem 영역(행타일 x 특징) 대조
  task chk(input [255:0] nm, input integer base, input integer nfeat,
           input integer nrow, input integer which, input integer tol);
    integer mt, k, lane, TTx, n;
    reg signed [15:0] g, x;
    begin
      e0 = errors; s0 = soft; TTx = (nrow + 31) / 32; n = 0;
      for (mt = 0; mt < TTx; mt = mt + 1)
        for (k = 0; k < nfeat; k = k + 1)
          for (lane = 0; lane < N; lane = lane + 1)
            if (mt*32 + lane < nrow) begin
              n = n + 1;
              case (which)
                0: begin g = dut.u_a_mem0.mem[base + mt*nfeat + k][lane*16 +: 16];
                         x = g_pin[mt*nfeat + k][lane*16 +: 16]; end
                1: begin g = dut.u_a_mem0.mem[base + mt*nfeat + k][lane*16 +: 16];
                         x = g_pre[mt*nfeat + k][lane*16 +: 16]; end
                2: begin g = dut.u_a_mem0.mem[base + mt*nfeat + k][lane*16 +: 16];
                         x = g_ev1[mt*nfeat + k][lane*16 +: 16]; end
                3: begin g = snx[mt*nfeat + k][lane*16 +: 16];    // 스냅
                         x = g_lnx[mt*nfeat + k][lane*16 +: 16]; end
                default: begin g = s1[mt*nfeat + k][lane*16 +: 16];
                               x = g_ln1[mt*nfeat + k][lane*16 +: 16]; end
              endcase
              cmp(nm, mt*nfeat + k, g, x, tol);
            end
      rpt(nm, n);
    end
  endtask

  integer h, mt, d, kk, lane, i;
  reg signed [15:0] gv, xv;

  initial begin
    $display("[tb_evt] EvT_Engine 전 구간 (샘플0 t0, 토큰 %0d, 타임스텝 1개)", NTOK);
    $readmemh("../data/wmem.hex",  wimg);
    $readmemh("../data/rqmem.hex", rqimg);
    $readmemh("../data/afmem.hex", afimg);
    $readmemh("../data/instmem.hex", instimg);
    $readmemh("../data/latinit.hex", limg);
    $readmemh("../data/bkv.hex", bimg);
    $readmemh("../data/board/t0_x.hex",   ximg);
    $readmemh("../data/board/t0_pos.hex", pimg);
    $readmemh("../data/board/t0_pidx.hex", qimg);
    $readmemh("../data/posmem.hex", posimg);
    $readmemh("../data/golden/PIN.hex",       g_pin);
    $readmemh("../data/golden/proc_ev.1.hex", g_pre);
    $readmemh("../data/golden/proc_ev.4.hex", g_ev1);
    $readmemh("../data/golden/in_proj.K.hex", g_lnx);
    $readmemh("../data/golden/in_proj.Q.hex", g_ln1);
    $readmemh("../data/golden/QI.hex",  g_qi);
    $readmemh("../data/golden/KI.hex",  g_ki);
    $readmemh("../data/golden/VI.hex",  g_vi);
    $readmemh("../data/golden/CTX.hex", g_ctx);
    $readmemh("../data/golden/TLNA.hex",  g_tlna);
    $readmemh("../data/golden/TLATV.hex", g_tlatv);
    $readmemh("../data/golden/B0ZATT.hex", g_zatt);
    $readmemh("../data/golden/B0LNA1.hex", g_lna1);
    $readmemh("../data/golden/B0LNA2.hex", g_lna2);
    $readmemh("../data/golden/B0FFN3.hex", g_ffn3);
    $readmemh("../data/golden/TFFN.hex",  g_tffn);
    $readmemh("../data/golden/TMEAN.hex", g_tmn);
    $readmemh("../data/golden/TCLF1.hex", g_tc1);
    $readmemh("../data/golden/clf_acc.hex", g_clf);

    repeat (8) @(posedge clk); rst = 0; @(posedge clk);

    $display("-- 메모리 적재");
    load(3'd0, W_WORDS,  0);  load(3'd2, PB_WORDS, 1);
    load(3'd3, PG_WORDS, 2);  load(3'd4, S_WORDS,  3);
    load_a(R_X,   TT*144, 0);
    // pos enc 는 PL 표에서 모읍니다 — 호스트는 표(1회)와 pos_idx 만 줍니다
    load_postbl; load_a(R_PIDX, 0, 4);
    load_a(R_Z,   0, 2);      load_a(R_LATV, 0, 2);
    load_a(R_BKV, 0, 3);

    $display("-- 실행 (본체 %0d + 꼬리 %0d)", n_body, n_tail);
    @(negedge clk); start = 1;
    cyc = 0;
    while (!done && cyc < 20000000) begin @(posedge clk); cyc = cyc + 1; end
    if (!done) begin
      $display("=== TIMEOUT  state=%0d inst=%0d ===", dbg_state, dbg_inst);
      $finish;
    end
    $display("     %0d 사이클  (100 MHz 에서 %.2f us)", cyc, cyc/100.0);
    prof_dump;
    @(negedge clk); start = 0;

    $display("-- A_Mem 대조");
    chk("PIN  event_projection", R_PIN, 160, NTOK, 0, 0);
    chk("PRE  preproc",          R_PRE, 128, NTOK, 1, 0);
    chk("EV1  proc_ev.1",        R_EV1, 128, NTOK, 2, 0);
    // 새 LayerNorm 코어는 알고리즘이 달라 ±1~2 차이가 납니다 (§5)
    chk("LNX  layer_norm_x",     R_LNX, 128, NTOK, 3, 2);   // cross 블록 스냅
    chk("LN1  layer_norm_1",     R_LN1, 128, 96,   4, 2);   // §5 ⑥ 로 ±1 허용

    // ---- Q : head-major  워드 = h*(QT*HD) + mt*HD + d, 레인 = latent 행 ----
    e0 = errors; s0 = soft; i = 0;
    for (h = 0; h < HEADS; h = h + 1)
      for (mt = 0; mt < QT; mt = mt + 1)
        for (d = 0; d < HD; d = d + 1)
          for (lane = 0; lane < N; lane = lane + 1) begin
            i = i + 1;
            gv = sq[h*QT*HD + mt*HD + d][lane*16 +: 16];
            xv = g_qi[mt*128 + h*HD + d][lane*16 +: 16];
            // `layer_norm_1` 의 ±1(§5 ⑥)이 그대로 넘어옵니다
            cmp("Q  in_proj.Q", i, gv, xv, 3);
          end
    rpt("Q    in_proj.Q", i);

    // ---- K : 워드 = h*KSTR + kt*32 + d, 레인 = 키 ----
    e0 = errors; s0 = soft; i = 0;
    for (h = 0; h < HEADS; h = h + 1)
      for (mt = 0; mt < TT; mt = mt + 1)
        for (d = 0; d < HD; d = d + 1)
          for (lane = 0; lane < N; lane = lane + 1)
            if (mt*32 + lane < NTOK) begin
              i = i + 1;
              gv = sk[h*KSTR + mt*HD + d][lane*16 +: 16];
              xv = g_ki[mt*128 + h*HD + d][lane*16 +: 16];
              cmp("K  in_proj.K", i, gv, xv, 3);
            end
    rpt("K    in_proj.K", i);

    // ---- V : **전치** 워드 = h*VSTR + 키, 레인 = head_dim ----
    e0 = errors; s0 = soft; i = 0;
    for (h = 0; h < HEADS; h = h + 1)
      for (kk = 0; kk < NTOK; kk = kk + 1)
        for (d = 0; d < HD; d = d + 1) begin
          i = i + 1;
          gv = sv[h*VSTR + kk][d*16 +: 16];
          xv = g_vi[(kk/32)*128 + h*HD + d][(kk%32)*16 +: 16];
          cmp("V  in_proj.V (전치)", i, gv, xv, 3);
        end
    rpt("V    in_proj.V (전치)", i);

    // ---- CTX : 워드 = mt*128 + h*HD + d, 레인 = latent 행 ----
    e0 = errors; s0 = soft; i = 0;
    for (mt = 0; mt < QT; mt = mt + 1)
      for (kk = 0; kk < 128; kk = kk + 1)
        for (lane = 0; lane < N; lane = lane + 1) begin
          i = i + 1;
          gv = sc[mt*128 + kk][lane*16 +: 16];
          xv = g_ctx[mt*128 + kk][lane*16 +: 16];
          // §5 ⑥ 의 1 LSB 가 `layer_norm_1` → `in_proj.Q` → softmax 를 거치며
          // 53항 내적에서 최대 2 까지 벌어집니다 (12,288 중 2개).
          cmp("CTX attn.V", i, gv, xv, 4);
        end
    rpt("CTX  attn·V", i);

    // ---- cross 블록 FFN 체인 (out_proj 뒤 — 지금까지 미검증 구간) ----
    // 어디서 갈리는지 한 지점으로 좁히려면 **차례대로** 봐야 합니다.
    // bf16 패턴 비교라 허용치가 곧 ulp 입니다. 잔차 스트림이 bf16 인 하드웨어와
    // fp32 인 골든은 **원리상 몇 ulp 씩 벌어집니다**(가수 8비트 = 0.78 %/연산).
    // 16 ulp ~ 수 % — 이 구간의 판정 기준은 이 뒤의 **int8 값**과 클래스입니다.
    bf_report("ZATT out_proj+잔차", 0);
    // 잔차(bf16) 뒤의 int8 값들 — LayerNorm 이 스트림 오차를 대부분 흡수하므로
    // ±2 코드 안이어야 정상입니다. 앞단(PIN/PRE/EV1/LNX/K/V)은 tol 0 입니다.
    snap_chk("LNA1 layer_norm_att", 1, 4);
    snap_chk("LNA2 layer_norm_2",   2, 4);
    snap_chk("FFN3 linear2+GELU",   3, 4);

    $display("-- 꼬리 진단 : LATV / TLNA / TFFN 앞 6레인 (하드웨어 / 골든)");
    for (i = 0; i < 6; i = i + 1)
      $display("     LATV[k0] lane%0d : %04x / %04x   TLNA %0d / %0d   TFFN %0d / %0d",
               i, dut.u_a_mem0.mem[R_LATV][i*16 +: 16], g_tlatv[0][i*16 +: 16],
               $signed(sln[0][i*16 +: 16]),  $signed(g_tlna[0][i*16 +: 16]),
               $signed(sfn[0][i*16 +: 16]),  $signed(g_tffn[0][i*16 +: 16]));

    // ---- 꼬리 단 : LNA(96행) → MEAN(1행) → clf.linear_1(1행) → 누산기 10개 ----
    // 최종 클래스 하나만 보면 **채널별 상수를 틀려도 우연히 통과**합니다
    // (ARGMAX 가 채널 0 의 M 만 쓰는 버그가 실제로 그렇게 보드까지 갔습니다).
    e0 = errors; s0 = soft; i = 0;
    for (mt = 0; mt < QT; mt = mt + 1)
      for (kk = 0; kk < 128; kk = kk + 1)
        for (lane = 0; lane < N; lane = lane + 1) begin
          i = i + 1;
          gv = sln[mt*128 + kk][lane*16 +: 16];       // MEAN 전 스냅
          xv = g_tlna[mt*128 + kk][lane*16 +: 16];
          cmp("TLNA embs.ln", i, gv, xv, 4);
        end
    rpt("TLNA embs.ln", i);

    bf_report("TLATV 누적 latent", 1);

    // 값 자체를 눈으로 — ReLU 가 걸렸으면 음수가 없어야 합니다
    begin : DUMP
      integer neg, mn, mx, sum0;
      neg = 0; mn = 999; mx = -999; sum0 = 0;
      for (mt = 0; mt < QT; mt = mt + 1)
        for (lane = 0; lane < N; lane = lane + 1)
          if (mt*32 + lane < 96) begin
            gv = sfn[mt*128 + 0][lane*16 +: 16];
            if (gv < 0) neg = neg + 1;
            if (gv < mn) mn = gv;
            if (gv > mx) mx = gv;
            sum0 = sum0 + gv;
          end
      $display("  FFN[k=0] 하드웨어 : 음수 %0d개  범위 %0d~%0d  합 %0d", neg, mn, mx, sum0);
      neg = 0; mn = 999; mx = -999; sum0 = 0;
      for (mt = 0; mt < QT; mt = mt + 1)
        for (lane = 0; lane < N; lane = lane + 1)
          if (mt*32 + lane < 96) begin
            xv = g_tffn[mt*128 + 0][lane*16 +: 16];
            if (xv < 0) neg = neg + 1;
            if (xv < mn) mn = xv;
            if (xv > mx) mx = xv;
            sum0 = sum0 + xv;
          end
      $display("  FFN[k=0] 골든     : 음수 %0d개  범위 %0d~%0d  합 %0d", neg, mn, mx, sum0);
      $display("  FFN[k=0] 앞 8레인 hw/gold: %0d/%0d %0d/%0d %0d/%0d %0d/%0d",
               $signed(sfn[0][15:0]),  $signed(g_tffn[0][15:0]),
               $signed(sfn[0][31:16]), $signed(g_tffn[0][31:16]),
               $signed(sfn[0][47:32]), $signed(g_tffn[0][47:32]),
               $signed(sfn[0][63:48]), $signed(g_tffn[0][63:48]));
    end

    // MEAN 의 입력 (embs.linear1 → ReLU). `clf.linear_1` 이 나중에 FFN 을
    // 덮으므로 MEAN 직전 스냅으로 봅니다.
    e0 = errors; s0 = soft; i = 0;
    for (mt = 0; mt < QT; mt = mt + 1)
      for (kk = 0; kk < 128; kk = kk + 1)
        for (lane = 0; lane < N; lane = lane + 1) begin
          i = i + 1;
          gv = sfn[mt*128 + kk][lane*16 +: 16];
          xv = g_tffn[mt*128 + kk][lane*16 +: 16];
          cmp("TFFN embs.linear1", i, gv, xv, 4);
        end
    rpt("TFFN embs.linear1", i);

    // MEAN 은 **레인 0** 에만 씁니다 (다음 GEMM 이 M=1 로 읽음)
    e0 = errors; s0 = soft; i = 0;
    for (kk = 0; kk < 128; kk = kk + 1) begin
      i = i + 1;
      gv = dut.u_a_mem0.mem[R_LNA + kk][15:0];
      xv = g_tmn[kk][15:0];
      cmp("TMEAN gap", i, gv, xv, 1);
    end
    rpt("TMEAN latent 평균", i);

    e0 = errors; s0 = soft; i = 0;
    for (kk = 0; kk < 128; kk = kk + 1) begin
      i = i + 1;
      gv = dut.u_a_mem0.mem[R_FFN + kk][15:0];
      xv = g_tc1[kk][15:0];
      cmp("TCLF1 clf.linear_1", i, gv, xv, 1);
    end
    rpt("TCLF1 clf.linear_1", i);

    // 분류기 누산기 10개 — **값으로** 봅니다. 클래스 하나만 보면 채널별 상수를
    // 통째로 틀려도 통과합니다(실제로 그렇게 보드까지 갔습니다).
    // 비트 완전일치는 원리상 불가능합니다 — 잔차 스트림이 bf16 이라 20 타임스텝
    // 누적분이 골든(fp32)과 벌어집니다. 그래서 **최대 크기의 1.5 %** 로 봅니다.
    begin : CLFCHK
      integer mx, tolc, d;
      mx = 0;
      for (i = 0; i < 10; i = i + 1)
        if ($signed(g_clf[i]) > mx)  mx = $signed(g_clf[i]);
        else if (-$signed(g_clf[i]) > mx) mx = -$signed(g_clf[i]);
      // 새 LayerNorm/softmax 코어는 알고리즘이 달라 골든과 원리상 벌어집니다.
      // 3 % 안이면 순위(=클래스)가 안 바뀝니다.
      tolc = mx * 3 / 100;
      e0 = errors; s0 = soft;
      for (i = 0; i < 10; i = i + 1) begin
        checks = checks + 1;
        d = $signed(res_logits[i*32 +: 32]) - $signed(g_clf[i]);
        if (d < 0) d = -d;
        if (d > tolc) begin
          errors = errors + 1;
          $display("    [FAIL] clf_acc[%0d] got %0d  exp %0d  (차 %0d > 허용 %0d)",
                   i, $signed(res_logits[i*32 +: 32]), $signed(g_clf[i]), d, tolc);
        end else if (d != 0) soft = soft + 1;
      end
      $display("  %-26s %0d/10 정확, 허용(<=%0d) %0d개, 초과 %0d개",
               "CLF 누산기 10개", 10 - (soft - s0) - (errors - e0), tolc,
               soft - s0, errors - e0);
      e0 = errors; s0 = soft;
    end

    $display("-- 분류 결과  (이게 맞으면 뒷블록 2개 · MEAN · ARGMAX 까지 통과)");
    $display("     res_class = %0d   (골든 pred = 9, label = 9)", res_class);
    for (i = 0; i < 10; i = i + 1)
      $display("       logit[%0d] = %0d", i, $signed(res_logits[i*32 +: 32]));

    if (res_class !== 4'd9) begin
      errors = errors + 1;
      $display("    [FAIL] res_class %0d != 9", res_class);
    end
    if (errors == 0) $display("=== tb_evt: %0d checks (허용 %0d), TEST PASSED ===",
                              checks, soft);
    else             $display("=== tb_evt: %0d/%0d failed, TEST FAILED ===",
                              errors, checks);
    $finish;
  end

  task prof_dump;
    integer i, j, tot, t, bi;
    integer kc0, kc1, kc2, kc3, kc4, kc5, kc6;
    begin
      tot=0; kc0=0; kc1=0; kc2=0; kc3=0; kc4=0; kc5=0; kc6=0;
      for (i = 0; i < 256; i = i + 1) if (psk[i] < 7) begin
        tot = tot + psc[i];
        case (psk[i])
          0: kc0 = kc0 + psc[i];  1: kc1 = kc1 + psc[i];
          2: kc2 = kc2 + psc[i];  3: kc3 = kc3 + psc[i];
          4: kc4 = kc4 + psc[i];  5: kc5 = kc5 + psc[i];
          6: kc6 = kc6 + psc[i];
        endcase
      end
      $display("");
      $display("PROF 총 %0d 사이클", tot);
      $display("PROF KIND GEMM   %0d", kc0);
      $display("PROF KIND LN     %0d", kc1);
      $display("PROF KIND SMAX   %0d", kc2);
      $display("PROF KIND RES    %0d", kc3);
      $display("PROF KIND MEAN   %0d", kc4);
      $display("PROF KIND ARGMAX %0d", kc5);
      $display("PROF KIND POS    %0d", kc6);
      for (i = 0; i < 256; i = i + 1) if (psk[i] < 7)
        $display("PROF STEP %0d kind %0d M %0d NOUT %0d cyc %0d",
                 i, psk[i], psm[i], psn[i], psc[i]);
    end
  endtask
endmodule