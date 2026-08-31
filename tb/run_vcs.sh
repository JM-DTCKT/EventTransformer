#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
BUILD="$SCRIPT_DIR/build"; mkdir -p "$BUILD"
: "${XILINX_VIVADO:=/opt/Xilinx/Vivado/2022.2}"
U="$XILINX_VIVADO/data/verilog/src/unisims"
TB="${1:-tb_evt}"
# 컴파일이 실패해도 **직전 바이너리가 남아** 그대로 돌아갑니다 — 고친 적 없는
# 코드를 고쳤다고 착각하기 딱 좋습니다. 먼저 지웁니다.
# `.daidir` 도 같이 지웁니다 — 바이너리만 지우면 VCS 가 "design hasn't changed"
# 로 **컴파일을 건너뛰고** 바이너리를 안 만듭니다 (증분 판정은 daidir 의
# 타임스탬프를 봅니다). 그러면 이번엔 "바이너리 없음"으로 조용히 실패합니다.
rm -rf "$BUILD/simv_$TB" "$BUILD/simv_$TB.daidir"
echo ">> compile $TB"
# EVT_PROF=1 ./run_vcs.sh tb_evt  -> 끝에 step 별 사이클 표를 찍습니다
vcs -full64 -sverilog -timescale=1ns/1ps -notice ${EVT_PROF:++define+EVT_PROF} \
    -y "$U" +libext+.v +incdir+"${U}_dr" +incdir+. +incdir+../rtl +incdir+../rtl/gelu +incdir+../rtl/softmax +incdir+../rtl/rsqrt \
    $TB.v ../rtl/*.v ../rtl/*/*.v "$XILINX_VIVADO/data/verilog/src/glbl.v" \
    -top $TB -top glbl -Mdir="$BUILD/csrc_$TB" -o "$BUILD/simv_$TB" \
    -l "$BUILD/$TB.comp.log" > /dev/null 2>&1
[ -x "$BUILD/simv_$TB" ] || { echo "  [COMPILE FAIL]"; grep -A4 "Error-" "$BUILD/$TB.comp.log"|head -20; exit 1; }
echo ">> run"
"$BUILD/simv_$TB" -l "$BUILD/$TB.run.log" > /dev/null 2>&1
grep -E '^\[|^--|^  |\[FAIL\]|TEST PASSED|TEST FAILED|TIMEOUT' "$BUILD/$TB.run.log" | head -30
grep -q "TEST PASSED" "$BUILD/$TB.run.log" && exit 0 || exit 1
