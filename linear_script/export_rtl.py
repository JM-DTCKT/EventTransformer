"""정수 체크포인트 -> 보드/시뮬에 바로 올리는 상수 + 골든 벡터.

    python export_rtl.py               # 기본 32장
    python export_rtl.py --n 64

`rtl_export/` 에 두 형식으로 같은 내용을 씁니다.

    *.hex   시뮬용   `$readmemh` — 2의 보수 hex, 한 줄에 하나
    *.bin   보드용   raw — int8 은 1바이트, int32 는 **리틀엔디언 4바이트**

## 가장 중요한 것: weight 는 **전치해서** 내보냅니다

`Mac_OS/Linear_Top` 은 `C[m][n] = Σ_k a_mem[m][k] · b_mem[k][n]` 를 계산하고
**전치를 하지 않습니다**. 우리가 원하는 것은 `Y = X · Wᵀ` 이므로 호출측이

    b_mem[k][n] = w_int[n][k]          (E_in × E_out, 행 우선)

로 넣어야 합니다. PyTorch 의 `nn.Linear.weight` 는 `[E_out][E_in]` 이라 그대로
넣으면 **틀립니다**. 여기서 미리 전치해 `L<i>_bmem.*` 으로 내보내고, 내보낸
레이아웃으로 `Mac_OS` 의 식을 그대로 재계산해 골든 ACC 와 대조합니다
(`--no_verify` 로 끄지 않는 한 export 가 자체 검증 없이는 끝나지 않습니다).

## 파일

상수 (보드 상주)
    L<i>_bmem.int8.{hex,bin}    [E_in][E_out]  → Mac_OS.b_mem      ← **전치됨**
    L<i>_bias.int32.{hex,bin}   [E_out]        → Requant_Int.bias
    L<i>_mult.int32.{hex,bin}   [E_out]        → Requant_Int.mult
                                shift 는 manifest.json

골든 벡터 (검증용, 실제 MNIST 테스트 이미지)
    L<i>_amem.int8.{hex,bin}    [M][E_in]      → Mac_OS.a_mem
    L<i>_acc.int32.{hex,bin}    [M][E_out]     ← Mac_OS + bias 가 내야 할 값
    L<i>_out.int8.{hex,bin}     [M][E_out]     ← Requant_Int 가 내야 할 값
    argmax.int32.{hex,bin}      [M]            최종 클래스
    labels.int32.{hex,bin}      [M]            정답 (참고)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data as data_mod                                  # noqa: E402
from quantize import int_forward                         # noqa: E402

PT = os.path.join(SCRIPT_DIR, 'mlp_int8.pt')
OUT_DIR = os.path.join(SCRIPT_DIR, 'rtl_export')


def emit(name, t, bits):
    """`<name>.hex` (시뮬) + `<name>.bin` (보드) 동시 생성. 반환: 원소 수."""
    v = np.asarray(t.reshape(-1).cpu().long().tolist(), dtype=np.int64)
    nib = bits // 4
    with open(f'{OUT_DIR}/{name}.hex', 'w') as f:
        for x in v.tolist():
            f.write(f'{(x & ((1 << bits) - 1)):0{nib}x}\n')
    dt = np.int8 if bits == 8 else np.dtype('<i4')       # int32 는 리틀엔디언
    v.astype(dt).tofile(f'{OUT_DIR}/{name}.bin')
    return int(v.size)


def mac_os_reference(a_mem, b_mem, b_int):
    """`Mac_OS` 가 하는 식을 **내보낸 레이아웃 그대로** 재계산.

        C[m][n] = Σ_k a_mem[m][k] · b_mem[k][n]   (+ requant 프리애더의 b_int[n])

    전치를 잘못했으면 여기서 바로 어긋납니다.
    """
    return a_mem.astype(np.int64) @ b_mem.astype(np.int64) + b_int.astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=32, help='골든 벡터 이미지 수 (Mac_OS 의 M)')
    ap.add_argument('--no_verify', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pl = torch.load(PT, map_location='cpu', weights_only=False)
    L = pl['layers']

    _, test_loader = data_mod.loaders(batch_size=args.n, workers=0)
    x, y = next(iter(test_loader))
    pred, trace = int_forward(pl, x.reshape(x.shape[0], -1), keep_trace=True)

    manifest = dict(
        note='acc[m][n] = sum_k a_mem[m][k]*b_mem[k][n] + b_int[n];  '
             'out[n] = sat((acc*M[n] + 2**(sh-1)) >> sh)',
        weight_layout='b_mem[E_in][E_out] = w_int[E_out][E_in] 을 전치한 것 (Mac_OS 용)',
        int32_endian='little', dims=list(pl['dims']), n_vectors=int(args.n),
        fp32_accuracy=pl['fp32_accuracy'], int8_accuracy=pl['int8_accuracy'],
        layers=[])

    files, total_bytes, mismatches = {}, 0, []
    for i, (e, tr) in enumerate(zip(L, trace)):
        E_out, E_in = e['shape']
        tag = f'L{i}'

        # ---- weight 전치: [E_out][E_in] -> [E_in][E_out] = b_mem[k][n] ----
        w_int = e['w_int'].long()                        # (E_out, E_in)
        b_mem = w_int.t().contiguous()                   # (E_in,  E_out)

        n_w = emit(f'{tag}_bmem.int8',  b_mem,     8)
        n_b = emit(f'{tag}_bias.int32', e['b_int'], 32)
        n_m = emit(f'{tag}_mult.int32', e['M'],     32)
        emit(f'{tag}_amem.int8',  tr['x_int'], 8)
        emit(f'{tag}_acc.int32',  tr['acc'],  32)
        const_bytes = n_w * 1 + n_b * 4 + n_m * 4
        total_bytes += const_bytes

        # ---- 내보낸 레이아웃으로 Mac_OS 식을 재계산해 골든 ACC 와 대조 ----
        if not args.no_verify:
            ref = mac_os_reference(np.array(tr['x_int'].cpu().long().tolist()),
                                   np.array(b_mem.cpu().tolist()),
                                   np.array(e['b_int'].cpu().tolist()))
            got = np.array(tr['acc'].cpu().tolist())
            if not (ref == got).all():
                mismatches.append(f'{tag}: b_mem 레이아웃으로 재계산한 acc 가 골든과 다름')

        rec = dict(
            name=e['name'], index=i, E_out=int(E_out), E_in=int(E_in),
            shift=int(e['shift']), input_step=float(e['s_x']),
            output_step=(None if e['out_step'] is None else float(e['out_step'])),
            relu_fused=bool(e['relu']), unsigned_out=bool(e['relu']),
            out_range=([0, 127] if e['relu'] else None),
            consumer=('argmax(acc*M)' if tr['out'] is None else f'L{i+1}.a_mem (int8)'),
            max_abs_acc=int(tr['acc'].abs().max()),
            max_abs_b_int=int(e['b_int'].abs().max()),
            max_abs_M=int(e['M'].abs().max()),
            const_bytes=const_bytes,
            files=dict(b_mem=f'{tag}_bmem.int8', bias=f'{tag}_bias.int32',
                       mult=f'{tag}_mult.int32',
                       golden_a_mem=f'{tag}_amem.int8', golden_acc=f'{tag}_acc.int32'))
        if tr['out'] is not None:
            emit(f'{tag}_out.int8', tr['out'], 8)
            rec['files']['golden_out'] = f'{tag}_out.int8'
        manifest['layers'].append(rec)
        files[tag] = rec['files']

    emit('argmax.int32', pred, 32)
    emit('labels.int32', y, 32)
    manifest['const_bytes_total'] = total_bytes
    manifest['golden_batch_accuracy'] = float((pred == y).float().mean())

    if mismatches:
        for m in mismatches:
            print(f'  [FAIL] {m}')
        raise SystemExit('전치 검증 실패 — 내보내지 않음')

    with open(f'{OUT_DIR}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'-> {OUT_DIR}\n')
    print(f"{'layer':>6s} {'b_mem [k][n]':>14s} {'sh':>3s} {'max|acc|':>9s} "
          f"{'상수 bytes':>10s}  소비자")
    for r in manifest['layers']:
        shape = f"[{r['E_in']}][{r['E_out']}]"
        print(f"{r['name']:>6s} {shape:>14s} {r['shift']:>3d} {r['max_abs_acc']:>9d} "
              f"{r['const_bytes']:>10,d}  {r['consumer']}")
    print(f"\n상수 합계        : {total_bytes:,} bytes ({total_bytes/1024:.1f} KiB)")
    print(f"fp32 체크포인트   : {109386*4:,} bytes ({109386*4/1024:.1f} KiB)  "
          f"→ {109386*4/total_bytes:.2f}x 축소")
    print(f"전치 검증        : {len(L)}개 레이어 전부 b_mem 레이아웃으로 재계산해 골든 ACC 와 일치 ✅")
    print(f"골든 배치 {args.n}장   : {manifest['golden_batch_accuracy']*100:.1f}%")
    print(f"전체 테스트셋     : fp32 {pl['fp32_accuracy']*100:.2f}%  ->  "
          f"int8 {pl['int8_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
