# =============================================================================
# ZCU102 용 EvT(DVS128 제스처) 가속기 — 프로젝트 → BD → 합성 → 임플 → .xsa
#
#   source /opt/Xilinx/Vivado/2022.2/settings64.sh
#   vivado -mode batch -source build.tcl              # 전부
#   vivado -mode batch -source build.tcl -tclargs bd  # BD 까지만
#   vivado -mode batch -source build.tcl -tclargs all build2   # 다른 출력 폴더
#
# 배선은 `fpga_nl` 과 같습니다:
#
#   PS ─HPM0(32b)─→ smc_ctrl ─┬─→ axi_dma_0 / S_AXI_LITE
#                             └─→ evt_accel_0 / s_axi      제어·핸드셰이크·결과
#   PS ←─HP0(128b)─ smc_data ←┬── axi_dma_0 / M_AXI_MM2S
#                             └── axi_dma_0 / M_AXI_S2MM
#   axi_dma_0 / M_AXIS_MM2S(128b) ──→ evt_accel_0 / s_axis  W/A/RQ/AF/INST 적재
#   axi_dma_0 / S_AXIS_S2MM(128b) ←── evt_accel_0 / m_axis  A_Mem 덤프
#
# `fpga_nl` 과 다른 점:
#
#  1. **DMA 가 자주 돕니다.** 명령어 프로그램은 한 번이지만 X/pos enc 는 타임스텝
#     마다 들어갑니다 (샘플당 20번 x 2). 가장 큰 전송은 여전히 W_Mem 448 KB 한
#     방이라 `c_sg_length_width` 는 26 그대로입니다.
#  2. **LUT 이 4종**입니다 (gelu 는 이제 PWL 이라 `.vh` 로 들어가지만, exp/recip/
#     rsqrt 는 여전히 `$readmemh`). 못 읽어도 **경고로 끝나므로** 합성 로그를
#     검사합니다 — 0 으로 채워진 채 보드에 올라가면 softmax 가 전부 0 입니다.
#  3. **`Gelu_Lut.vh`** 는 `\`include` 라 add_files 대신 include 경로가 필요합니다
#     (`rtl/` 안에 같이 있으므로 자동으로 잡힙니다).
# =============================================================================

set STAGE [expr {$argc > 0 ? [lindex $argv 0] : "all"}]

set here  [file normalize [file dirname [info script]]]
set rtl   [file normalize $here/../rtl]
set root  [file normalize $here/../..]
set luts  [file normalize $root/nl_export/lut]
set outd  [file normalize $here/../[expr {$argc > 1 ? [lindex $argv 1] : "build"}]]

set PART  xczu9eg-ffvb1156-2-e
set BOARD xilinx.com:zcu102:part0:3.4
set PROJ  evt_zcu102
set BD    design_1
# PL 클럭 (MHz) — 세 번째 인자로 바꿉니다. 기본 100 은 예전 동작 그대로입니다.
#   vivado -mode batch -source build.tcl -tclargs all build_150 150
set FREQ  [expr {$argc > 2 ? [lindex $argv 2] : 100}]
set NJOBS 4

file mkdir $outd

# =============================================================================
# 1. 프로젝트
# =============================================================================
create_project $PROJ $outd/$PROJ -part $PART -force
set_property board_part $BOARD [current_project]

# rtl/ 안의 심볼릭 링크는 glob 이 그대로 따라갑니다 (Gelu/Bf16/LayerNorm/Softmax/
# Requant/Mac_OS 의 실체 파일). 공용 모듈을 복사하지 않는 이유입니다.
# rtl/ 은 모듈별 하위 디렉토리로 정리돼 있습니다 (core/ axi/ gemm_core/ …).
# 최상위 Top.v 와 하위 디렉토리를 모두 훑습니다.
add_files -norecurse [concat [glob -nocomplain $rtl/*.v] [glob -nocomplain $rtl/*/*.v]]

# LUT 초기화 파일 — 반드시 프로젝트 안에 있어야 $readmemh 가 찾습니다
# `$readmemh` LUT 은 더 이상 없습니다 — 새 softmax/LayerNorm 코어는 상수를
# `\`include` 헤더로 들고 있고, 예전 코어(exp/recip/rsqrt .hex)는 안 씁니다.
#
# 헤더는 **프로젝트에 넣고 파일 타입을 지정**해야 합니다 — glob *.v 로는 안
# 잡히고, 없으면 create_bd_cell 이 통째로 실패합니다.
#   Gelu_Lut   PWL GELU 의 base/delta 64쌍
#   exp2/recip 새 softmax 코어 (`SOFTMAX/`)
#   rsqrt      새 LayerNorm 코어 (`LAYERNORM/`)
foreach vh {gelu/Gelu_Lut.vh softmax/Exp2_Lut.vh softmax/Recip_Lut.vh layernorm/Rsqrt_Lut.vh} {
    if {![file exists $rtl/$vh]} { error ">>> rtl/$vh 없음" }
    add_files -norecurse -fileset sources_1 $rtl/$vh
    set_property file_type {Verilog Header} [get_files $rtl/$vh]
}
# `is_global_include` 는 **켜면 안 됩니다** — 이 헤더는 모듈 안에 들어가는
# `initial` 블록이라 단독 파싱하면 문법 오류(CRITICAL WARNING)가 납니다.
# 프로젝트에 들어 있기만 하면 `\`include` 가 찾습니다.
update_compile_order -fileset sources_1

# =============================================================================
# 2. 블록 디자인
# =============================================================================
create_bd_design $BD

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]

set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0      {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__USE__M_AXI_GP1      {0} \
    CONFIG.PSU__USE__M_AXI_GP2      {0} \
    CONFIG.PSU__USE__S_AXI_GP2      {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE     {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $FREQ \
] [get_bd_cells zynq_ultra_ps_e_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_0

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells smc_ctrl]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_data
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells smc_data]

# AXI DMA — Simple mode. 가장 큰 전송이 W_Mem 235,520 바이트 한 방이라
# c_sg_length_width 를 26 으로 올립니다 (기본 14 는 16 KB 상한).
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg              {0} \
    CONFIG.c_sg_length_width         {26} \
    CONFIG.c_include_mm2s_dre        {0} \
    CONFIG.c_include_s2mm_dre        {0} \
    CONFIG.c_m_axi_mm2s_data_width   {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_mm2s_burst_size         {16} \
    CONFIG.c_m_axi_s2mm_data_width   {128} \
    CONFIG.c_s_axis_s2mm_tdata_width {128} \
    CONFIG.c_s2mm_burst_size         {16} \
] [get_bd_cells axi_dma_0]

create_bd_cell -type module -reference Top evt_accel_0
puts "\n>>> evt_accel_0 추론된 인터페이스:"
foreach ip [get_bd_intf_pins evt_accel_0/*] {
    puts [format "      %-28s %s" [get_property NAME $ip] [get_property VLNV $ip]]
}

# =============================================================================
# 배선
# =============================================================================
set CLK  [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
set RSTN [get_bd_pins rst_0/peripheral_aresetn]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_0/ext_reset_in]
connect_bd_net $CLK [get_bd_pins rst_0/slowest_sync_clk]

foreach p {zynq_ultra_ps_e_0/maxihpm0_fpd_aclk zynq_ultra_ps_e_0/saxihp0_fpd_aclk
           smc_ctrl/aclk smc_data/aclk axi_dma_0/s_axi_lite_aclk
           axi_dma_0/m_axi_mm2s_aclk axi_dma_0/m_axi_s2mm_aclk evt_accel_0/aclk} {
    connect_bd_net $CLK [get_bd_pins $p]
}
foreach p {smc_ctrl/aresetn smc_data/aresetn axi_dma_0/axi_resetn evt_accel_0/aresetn} {
    connect_bd_net $RSTN [get_bd_pins $p]
}

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
                    [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] \
                    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M01_AXI] \
                    [get_bd_intf_pins evt_accel_0/s_axi]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
                    [get_bd_intf_pins smc_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
                    [get_bd_intf_pins smc_data/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_data/M00_AXI] \
                    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
                    [get_bd_intf_pins evt_accel_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins evt_accel_0/m_axis] \
                    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

puts "\n================ ADDRESS MAP ================"
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces]] {
    catch {puts [format "  %-56s %s  %s" $seg \
        [get_property OFFSET $seg] [get_property RANGE $seg]]}
}

make_wrapper -files [get_files $BD.bd] -top
add_files -norecurse $outd/$PROJ/$PROJ.gen/sources_1/bd/$BD/hdl/${BD}_wrapper.v
set_property top ${BD}_wrapper [current_fileset]
update_compile_order -fileset sources_1

if {$STAGE eq "bd"} { puts "\n>>> BD 까지 완료"; return }

# =============================================================================
# 3. 합성
# =============================================================================
# 합성 단계 리타이밍 — 조합 로직을 가로질러 레지스터를 옮긴다. 우리가 손으로
# 6곳에 한 파이프라이닝의 자동화판이다.  배치를 모르므로 배선 지배 경로에는
# 직접 효과가 없지만, 넷리스트가 바뀌어 배치 결과 자체가 달라진다.
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

launch_runs synth_1 -jobs $NJOBS
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error ">>> 합성 실패 — $outd/$PROJ/$PROJ.runs/synth_1/runme.log"
}

# ---- 헤더 상수가 실제로 들어갔는지 ----
# `\`include` 를 못 찾으면 합성이 에러를 내지만, 경로가 꼬여 빈 배열로 넘어가는
# 일이 있었습니다 (`fpga_nl` 의 $readmemh 와 같은 부류). 로그를 훑습니다.
set slog $outd/$PROJ/$PROJ.runs/synth_1/runme.log
if {[file exists $slog]} {
    set fh [open $slog r]; set txt [read $fh]; close $fh
    set bad 0
    foreach f {Gelu_Lut.vh Exp2_Lut.vh Recip_Lut.vh Rsqrt_Lut.vh} {
        if {[regexp "Cannot open include file.*$f" $txt] ||
            [regexp "$f.*not found" $txt]} {
            puts ">>> \[치명\] $f 를 못 읽었습니다"
            incr bad
        }
    }
    if {$bad} { error ">>> 헤더 상수 적재 실패" }
    puts ">>> 헤더 상수 4개 모두 읽힘"
}

open_run synth_1 -name synth_1
puts "\n================ POST-SYNTH UTILIZATION ================"
report_utilization
puts ">>> DSP48E2 = [llength [get_cells -hier -filter {REF_NAME =~ DSP48E2*}]]"
puts ">>> RAMB36   = [llength [get_cells -hier -filter {REF_NAME =~ RAMB36*}]]"
puts ">>> RAMB18   = [llength [get_cells -hier -filter {REF_NAME =~ RAMB18*}]]"
close_design

# =============================================================================
# 4. 임플 / 비트스트림
# =============================================================================
# 배치/배선 노력을 올린다.  예전 코어로도 WNS 는 +0.094 ns 였고, 새 코어는
# DSP 73 % / LUT 57 % 로 더 빡빡하다.  이 전략은 place 후·route 후 두 번
# phys_opt_design 을 돌린다 (런타임은 늘지만 합성 결과는 바뀌지 않는다).
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

launch_runs impl_1 -to_step write_bitstream -jobs $NJOBS
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error ">>> 임플 실패 — $outd/$PROJ/$PROJ.runs/impl_1/runme.log"
}

open_run impl_1
puts "\n================ POST-IMPL ================"
report_utilization
report_timing_summary -delay_type max -max_paths 5
set wns [get_property SLACK [get_timing_paths -max_paths 1 -delay_type max]]
puts ">>> WNS = $wns ns"

# =============================================================================
# 5. Vitis 용 .xsa
# =============================================================================
set xsa $outd/evt_zcu102.xsa
write_hw_platform -fixed -include_bit -force $xsa
validate_hw_platform $xsa
puts "\n>>> .xsa = $xsa"
