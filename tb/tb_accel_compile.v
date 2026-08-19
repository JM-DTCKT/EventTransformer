// tb_accel_compile : Evt_Accel 엘라보레이션 확인 (기능 검증은 tb_accel_evt)
`timescale 1ns/1ps
module tb_accel_compile;
  reg aclk=0, aresetn=0; always #5 aclk=~aclk;
  wire [31:0] rdata; wire awr, wr, bv, arr, rv;
  wire sr, mv; wire [127:0] md; wire [15:0] mk; wire ml;
  Evt_Accel #(.EXP_LUT_FILE("../../nl_export/lut/exp.hex"),
              .RCP_LUT_FILE("../../nl_export/lut/recip.hex"),
              .RSQRT_LUT_FILE("../../nl_export/lut/rsqrt.hex")) dut (
    .aclk(aclk), .aresetn(aresetn),
    .s_axi_awaddr(12'd0), .s_axi_awprot(3'd0), .s_axi_awvalid(1'b0),
    .s_axi_awready(awr), .s_axi_wdata(32'd0), .s_axi_wstrb(4'hF),
    .s_axi_wvalid(1'b0), .s_axi_wready(wr), .s_axi_bresp(), .s_axi_bvalid(bv),
    .s_axi_bready(1'b1), .s_axi_araddr(12'd0), .s_axi_arprot(3'd0),
    .s_axi_arvalid(1'b0), .s_axi_arready(arr), .s_axi_rdata(rdata),
    .s_axi_rresp(), .s_axi_rvalid(rv), .s_axi_rready(1'b1),
    .s_axis_tvalid(1'b0), .s_axis_tready(sr), .s_axis_tdata(128'd0),
    .s_axis_tkeep(16'hFFFF), .s_axis_tlast(1'b0),
    .m_axis_tvalid(mv), .m_axis_tready(1'b1), .m_axis_tdata(md),
    .m_axis_tkeep(mk), .m_axis_tlast(ml));
  initial begin
    repeat(4) @(posedge aclk); aresetn=1;
    repeat(20) @(posedge aclk);
    $display("=== tb_accel_compile: 엘라보레이션 OK, TEST PASSED ===");
    $finish;
  end
endmodule
