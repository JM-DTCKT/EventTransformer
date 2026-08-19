#!/usr/bin/env bash
# 보드로 가져갈 것만 묶습니다.
#
#   ./make_bundle.sh          # 전체 264 샘플
#   ./make_bundle.sh 8        # 앞 8 샘플만 (첫 브링업용, 수 MB)
#
# 어느 비트스트림을 넣을지는 `XSA` 로 고릅니다. 기본은 **최신 빌드**입니다:
#
#   ./make_bundle.sh 8                                   # build_pp (ping-pong, §1.6)
#   XSA=build/evt_zcu102.xsa ./make_bundle.sh 8          # 예전 보드 검증본
#
# 예전 검증본이 아닌 XSA 를 넣으면 파일명에 `_pp` 가 붙어 섞이지 않습니다.
#
# `amem_pos.int16.bin`(38.7 MB)과 `amem_pin` 은 **안 넣습니다** — pos enc 는 이제
# PL 의 표(`posmem.bin`, 27.6 KB)에서 모으고 호스트는 `pos_idx` 만 보냅니다.
set -eu
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
n="${1:-0}"
XSA="${XSA:-build_pp/evt_zcu102.xsa}"
[ "$XSA" = "build/evt_zcu102.xsa" ] && tag="" || tag="_pp"
if [ "$n" = "0" ]; then out="$root/evt_board_bundle${tag}_all.tar.gz"
else out="$root/evt_board_bundle${tag}_n$n.tar.gz"; fi

if [ "$n" != "0" ]; then
    echo ">> 샘플 $n 개로 다시 export"
    ( cd "$root/sw" && \
      "${EVT:-/hai/home/sgh/.conda/envs/evt_new/bin/python}" export_board_evt.py --n "$n" )
fi

need=(
    "$XSA"
    sw/main_evt.c
    board/load_ddr.tcl
    data/wmem.bin data/pbmem.bin data/pgmem.bin data/stepmem.bin
    data/latinit.bin data/bkv.bin data/posmem.bin
    data/board/amem_x.int16.bin data/board/amem_pidx.int16.bin
    data/board/board_index.int32.bin data/board/board_samples.int32.bin
    data/schedule.json data/config.json
)
for f in "${need[@]}"; do
    [ -f "$root/$f" ] || { echo "!! 없음: $f"; exit 1; }
done

tar -czf "$out" -C "$root" "${need[@]}"
echo ">> $out  ($(du -h "$out" | cut -f1))"
echo ">> XSA = $XSA  ($(date -r "$root/$XSA" +%Y-%m-%d\ %H:%M))"
echo
echo "로컬에서:"
echo "  scp $(hostname):$out ."
echo "  tar xzf $(basename "$out")"
