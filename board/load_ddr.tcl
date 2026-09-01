# DDR 에 상수·샘플 이미지를 올립니다 (XSCT).
#
#   xsct% source .../board/load_ddr.tcl
#
# 주소는 `sw/main_evt.c` 의 DDR_* 상수와 **한 벌**입니다.
# 파일 경로는 이 스크립트 위치 기준입니다 — 서버 절대경로를 박으면
# 보드 쪽 XSCT 에서 못 찾습니다 (`fpga_nl` 에서 겪은 것).
#
# A53 이 리셋 상태면 `dow -data` 가 "Cannot write memory if not stopped" 로
# 실패합니다. **먼저 Vitis 에서 Debug As → Launch Hardware 로 main 에 멈춘 뒤**
# 이 스크립트를 돌리세요 (그래야 DDR 컨트롤러도 초기화돼 있습니다).

set here [file dirname [file normalize [info script]]]
set data [file normalize $here/../data]

proc put {f addr} {
    if {![file exists $f]} { error ">>> missing: $f" }
    puts [format "  %-30s -> 0x%08X  (%d B)" [file tail $f] $addr [file size $f]]
    dow -data $f $addr
}

targets -set -filter {name =~ "Cortex-A53 #0"}

# 이미 멈춰 있으면 `stop` 은 **에러**("Already stopped")를 던지고, 그러면
# `source` 가 여기서 끝나 아래 적재가 통째로 안 돕니다. 디버거로 main 에
# 멈춰 둔 정상 경로에서 오히려 걸립니다 — 그래서 catch 로 감쌉니다.
catch {stop}

# 정지 상태인지 확인 (APU Reset 이면 dow 가 실패합니다)
if {[catch {state} st]} { set st "?" }
puts "target state: $st"

put $data/wmem.bin                      0x10000000
put $data/rqmem.bin                     0x10100000
put $data/afmem.bin                     0x10110000
put $data/instmem.bin                   0x10120000
put $data/latinit.bin                   0x10130000
put $data/bkv.bin                       0x10140000
put $data/posmem.bin                    0x10150000
put $data/board/amem_x.int16.bin        0x20000000
put $data/board/amem_pidx.int16.bin     0x30000000
put $data/board/board_index.int32.bin   0x40000000
put $data/board/board_samples.int32.bin 0x40100000

# 올라갔는지 한 군데만 확인 (fpga_nl 에서 이 한 줄이 여러 번 살렸습니다)
puts "\nall images written"
puts "verify: mrd 0x10120000 (first instmem word)"
mrd 0x10120000 4
