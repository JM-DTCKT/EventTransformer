// -----------------------------------------------------------------------------
// tb_accel_evt : 보드 흐름 그대로 — AXI-Lite 제어 + DMA 적재 + 타임스텝 핸드셰이크
//
// `tb_evt` 는 엔진에 직접 `ld_we` 를 찔러 넣었지만, 여기는 **보드에서 실제로
// 일어나는 순서**를 그대로 밟습니다:
//
//   1회      W / PB / PG / Step 을 DMA (LOAD_SEL → LOAD_BASE → 스트림)
//   샘플마다  latinit → Z, LATV / bkv → BKV, N_TSTEP 쓰고 start
//   타임스텝  STATUS.tok_req 대기 → t 읽기 → X/PIN DMA → TOK_N → TOK_ACK
//   끝        STATUS.done 대기 → RES_CLASS
//
// 이걸 통과해야 레지스터 맵·로더 폭(목적지마다 2/4 beat)·핸드셰이크가 맞는
// 것이고, 보드 SW 는 이 순서를 그대로 옮기면 됩니다.
//
// 타임스텝 1개만 돌립니다 (`tb_evt` 와 같은 조건) — 기대 클래스 9.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_accel_evt;
  localparam N = 32, SW = 128;
  localparam R_CTRL=12'h000, R_STAT=12'h004, R_NBODY=12'h008, R_NTAIL=12'h00C,
             R_NTIME=12'h010, R_LSEL=12'h014, R_LBASE=12'h018, R_VER=12'h01C,
             R_CYC=12'h020, R_EPS=12'h024, R_DBASE=12'h028, R_DLEN=12'h02C,
             R_WLOAD=12'h030, R_TOKN=12'h034, R_TACK=12'h038, R_CLASS=12'h03C;
  // 워드 수 (data/config.json · schedule.json 과 한 벌)
  localparam W_WORDS=7904, PB_WORDS=869, PG_WORDS=208, S_WORDS=99,
             LAT_WORDS=384, BKV_WORDS=24, POS_ROWS=441;
  localparam R_X=0, R_PIDX=576, R_PIN=580, R_LATV=2756,
             R_Z=3140, R_BKV=6728;
  localparam NTOK = 52, TT = 2, EXP_CLASS = 9;

  reg aclk = 0, aresetn = 0; always #5 aclk = ~aclk;

  reg  [11:0] awaddr = 0, araddr = 0;
  reg  [31:0] wdata = 0;
  reg         awvalid = 0, wvalid = 0, arvalid = 0;
  wire        awready, wready, bvalid, arready, rvalid;
  wire [31:0] rdata;

  reg         s_tvalid = 0, s_tlast = 0;
  reg  [SW-1:0] s_tdata = 0;
  wire        s_tready;
  wire        m_tvalid, m_tlast;
  wire [SW-1:0] m_tdata;

  Top dut (
    .aclk(aclk), .aresetn(aresetn),
    .s_axi_awaddr(awaddr), .s_axi_awprot(3'd0), .s_axi_awvalid(awvalid),
    .s_axi_awready(awready), .s_axi_wdata(wdata), .s_axi_wstrb(4'hF),
    .s_axi_wvalid(wvalid), .s_axi_wready(wready), .s_axi_bresp(),
    .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
    .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid),
    .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rresp(),
    .s_axi_rvalid(rvalid), .s_axi_rready(1'b1),
    .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready), .s_axis_tdata(s_tdata),
    .s_axis_tkeep({SW/8{1'b1}}), .s_axis_tlast(s_tlast),
    .m_axis_tvalid(m_tvalid), .m_axis_tready(1'b1), .m_axis_tdata(m_tdata),
    .m_axis_tkeep(), .m_axis_tlast(m_tlast));

  integer errors = 0, checks = 0, i, t, cyc;
  reg [31:0] rv;

  task axi_w(input [11:0] a, input [31:0] d);
    begin
      @(negedge aclk); awaddr = a; wdata = d; awvalid = 1; wvalid = 1;
      @(posedge aclk); while (!(awready && wready)) @(posedge aclk);
      @(negedge aclk); awvalid = 0; wvalid = 0;
    end
  endtask

  task axi_r(input [11:0] a);
    begin
      @(negedge aclk); araddr = a; arvalid = 1;
      @(posedge aclk); while (!rvalid) @(posedge aclk);
      rv = rdata;
      @(negedge aclk); arvalid = 0;
    end
  endtask

  // ---- 메모리 이미지 (hex 한 줄 = 워드 하나, MSB 먼저) ----
  reg [255:0] wimg  [0:W_WORDS-1];
  reg [255:0] rqimg [0:PB_WORDS-1];
  reg [255:0] afimg [0:PG_WORDS-1];
  reg [255:0] instimg  [0:S_WORDS-1];
  reg [511:0] limg  [0:LAT_WORDS-1];
  reg [511:0] bimg  [0:BKV_WORDS-1];
  reg [511:0] ximg  [0:TT*144-1];
  reg [511:0] pimg  [0:TT*64-1];            // (참조용)
  reg [511:0] qimg  [0:TT-1];               // pos_idx
  reg [511:0] posimg[0:POS_ROWS-1];         // PL 표

  // 워드 하나를 128b 비트로 쪼개 흘립니다 (하위 128b 가 먼저).
  task push(input [511:0] w, input integer nbeat, input integer last);
    integer b;
    begin
      for (b = 0; b < nbeat; b = b + 1) begin
        @(negedge aclk);
        s_tdata  = w[b*SW +: SW];
        s_tvalid = 1;
        s_tlast  = (last && b == nbeat - 1);
        @(posedge aclk); while (!s_tready) @(posedge aclk);
      end
      @(negedge aclk); s_tvalid = 0; s_tlast = 0;
    end
  endtask

  // sel 0 W  1 A  2 PB  3 PG  4 Step   (A_Mem 만 4 beat)
  task dma_load(input integer sel, input integer base, input integer nw,
                input integer which);
    integer k, nb;
    begin
      axi_w(R_LSEL,  sel);
      axi_w(R_LBASE, base);
      // A_Mem(1) 과 POS 표(5) 만 512b — 나머지는 256b (로더와 한 벌)
    nb = (sel == 1 || sel == 5) ? 4 : 2;
      for (k = 0; k < nw; k = k + 1)
        case (which)
          0: push({256'd0, wimg[k]},  nb, k == nw-1);
          1: push({256'd0, rqimg[k]}, nb, k == nw-1);
          2: push({256'd0, afimg[k]}, nb, k == nw-1);
          3: push({256'd0, instimg[k]},  nb, k == nw-1);
          4: push(limg[k],            nb, k == nw-1);
          5: push(bimg[k],            nb, k == nw-1);
          6: push(ximg[k],            nb, k == nw-1);
          7: push(pimg[k],            nb, k == nw-1);
          8: push(qimg[k],            nb, k == nw-1);
          9: push(posimg[k],          nb, k == nw-1);
        endcase
      repeat (4) @(posedge aclk);
      axi_r(R_WLOAD);
      checks = checks + 1;
      if (rv !== nw) begin
        errors = errors + 1;
        $display("  [FAIL] sel=%0d 적재 %0d 워드 (기대 %0d)", sel, rv, nw);
      end else
        $display("  sel=%0d base=%0d  %0d 워드 적재 ✅", sel, base, nw);
    end
  endtask

  // pos enc 는 이제 **PL 의 표**에서 모읍니다 (`Pos_Gather`).
  // 호스트가 보내는 것은 표(1회, sel 5)와 `pos_idx`(타임스텝마다) 뿐입니다.

  initial begin
    $display("[tb_accel_evt] AXI 래퍼 — 보드 순서 그대로 (타임스텝 1개)");
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

    repeat (8) @(posedge aclk); aresetn = 1; repeat (4) @(posedge aclk);

    axi_r(R_VER);
    checks = checks + 1;
    if (rv !== 32'h4556_5401) begin
      errors = errors + 1; $display("  [FAIL] VERSION %08x", rv);
    end else $display("  VERSION %08x ✅", rv);

    $display("-- 상수 적재 (1회)");
    dma_load(0, 0, W_WORDS,  0);
    dma_load(2, 0, PB_WORDS, 1);
    dma_load(3, 0, PG_WORDS, 2);
    dma_load(4, 0, S_WORDS,  3);
    dma_load(5, 0, POS_ROWS, 9);       // pos enc 표 — 샘플과 무관, 1회

    $display("-- 샘플 시작값");
    dma_load(1, R_Z,    LAT_WORDS, 4);
    dma_load(1, R_LATV, LAT_WORDS, 4);
    dma_load(1, R_BKV,  BKV_WORDS, 5);

    axi_w(R_NBODY, 94);  axi_w(R_NTAIL, 5);  axi_w(R_NTIME, 1);
    axi_w(R_EPS, 32'h3727c5ac);

    $display("-- 실행");
    axi_w(R_CTRL, 32'h1);                       // start

    for (t = 0; t < 1; t = t + 1) begin
      // tok_req 대기 (STATUS[14])
      cyc = 0;
      axi_r(R_STAT);
      while (!rv[14] && cyc < 100000) begin axi_r(R_STAT); cyc = cyc + 1; end
      if (!rv[14]) begin
        $display("  [FAIL] tok_req 안 옴 (STATUS=%08x)", rv);
        errors = errors + 1; t = 99;
      end else begin
        $display("  t=%0d 요청 (STATUS=%08x)", rv[22:17], rv);
        dma_load(1, R_X,    TT*144, 6);
        dma_load(1, R_PIDX, TT,     8);
        axi_w(R_TOKN, NTOK);
        axi_w(R_TACK, 1);
      end
    end

    // done 대기 (STATUS[0])
    cyc = 0;
    axi_r(R_STAT);
    while (!rv[0] && cyc < 4000000) begin axi_r(R_STAT); cyc = cyc + 1; end
    checks = checks + 1;
    if (!rv[0]) begin
      $display("  [FAIL] done 안 뜸 (STATUS=%08x)", rv); errors = errors + 1;
    end else begin
      axi_r(R_CYC);
      $display("  done — %0d 사이클 (100 MHz 에서 %.2f us)", rv, rv/100.0);
    end

    axi_r(R_CLASS);
    checks = checks + 1;
    if (rv[3:0] !== EXP_CLASS[3:0]) begin
      errors = errors + 1;
      $display("  [FAIL] RES_CLASS %0d (기대 %0d)", rv[3:0], EXP_CLASS);
    end else
      $display("  RES_CLASS %0d ✅ (골든 예측 = 정답)", rv[3:0]);

    if (errors == 0)
      $display("=== tb_accel_evt: %0d checks, TEST PASSED ===", checks);
    else
      $display("=== tb_accel_evt: %0d/%0d failed, TEST FAILED ===", errors, checks);
    $finish;
  end
endmodule
