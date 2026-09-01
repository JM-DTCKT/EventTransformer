"""샘플별 A_Mem 입력 이미지 — 토큰 + positional encoding 을 워드 레이아웃으로

    conda activate evt_new
    python3 export_board_evt.py --n 4        # 앞 4 샘플
    python3 export_board_evt.py              # 전체 264

## 호스트가 무엇을 준비하나

엔진이 타임스텝 하나를 돌기 전에 A_Mem 의 두 영역이 채워져 있어야 합니다:

    X    [X_base   + mt*144 + k      ] 레인 i = token[mt*32+i][k]     k=0..143
    PIN  [PIN_base + mt*160 + 96 + d ] 레인 i = POS[pos_idx[i]][d]    d=0..63

`PIN` 의 앞 96워드는 엔진이 `event_projection` 결과로 채웁니다. 뒤 64워드가 pos enc
이고, **호스트가 미리 펴서** 넣습니다.

## pos enc 를 하드웨어로 안 하는 이유

pos enc 는 토큰마다 **다른 테이블 행**(`pos_idx`)을 읽습니다. A_Mem 워드 하나는
32레인이 서로 다른 토큰이므로, 하드웨어로 하면 **레인마다 다른 주소**가 필요합니다
— 32포트 게더거나 32사이클 순차입니다.

테이블(21x21x64)은 고정이고 `pos_idx` 는 전처리에서 이미 나오므로 호스트가 펴는
편이 낫습니다. 회로가 하나 없어집니다.

## 자체 검증

워드 레이아웃은 축이 뒤집히기 쉬운 자리라, 만든 이미지에서 **토큰을 되읽어**
원본과 대조합니다. `pack_nl.py` 의 W_Mem 자체검증과 같은 장치입니다.
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))
RDS = '/hai/home/sgh/01_assignment/EventTransformer/real_dvs_script'

N, TOKEN_DIM, POS_DIM, PIN_DIM = 32, 144, 64, 160
POS_GRID = 21


def lane_words(rows, nfeat, n_row):
    """(n_row, nfeat) int8 → 워드 리스트. 워드 k 의 레인 i = rows[mt*32+i][k]

    타일이 모자란 레인은 0 입니다 (`Gemm_Core` 의 edge mask 와 같은 규칙).
    """
    TT = (n_row + N - 1) // N
    out = np.zeros((TT * nfeat, N), dtype=np.int16)
    for mt in range(TT):
        lo, hi = mt * N, min((mt + 1) * N, n_row)
        blk = rows[lo:hi]                                   # (m, nfeat)
        out[mt * nfeat:(mt + 1) * nfeat, :hi - lo] = blk.T   # 축 전환
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--dst', default=os.path.join(DATA, 'board'))
    ap.add_argument('--sim_sample', type=int, default=0,
                    help='시뮬용으로 이 샘플의 **전 타임스텝**을 고정 stride 로 덤프')
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    cfg = json.load(open(os.path.join(DATA, 'config.json')))
    sch = json.load(open(os.path.join(DATA, 'schedule.json')))
    reg = {r['name']: r for r in sch['regions']}
    pos = np.fromfile(os.path.join(DATA, 'posenc.int8.bin'),
                      dtype=np.int8).reshape(POS_GRID, POS_GRID, POS_DIM)

    meta = json.load(open(os.path.join(RDS, 'data', 'meta.json')))
    tok = np.fromfile(os.path.join(RDS, 'data', 'tokens.int8.bin'),
                      dtype=np.int8).reshape(meta['n_token'], TOKEN_DIM)
    pidx = np.fromfile(os.path.join(RDS, 'data', 'pos_idx.int16.bin'), dtype='<i2')
    index = np.fromfile(os.path.join(RDS, 'data', 'index.int32.bin'),
                        dtype='<i4').reshape(-1, 4)
    samples = np.fromfile(os.path.join(RDS, 'data', 'samples.int32.bin'),
                          dtype='<i4').reshape(-1, 4)
    labels = np.fromfile(os.path.join(RDS, 'data', 'labels.int32.bin'), dtype='<i4')

    n_sample = args.n or len(samples)
    print(f'[export_board_evt] 샘플 {n_sample}/{len(samples)}')
    print(f"  X   base {reg['X']['base']:>5}  {reg['X']['words']} 워드")
    print(f"  PIN base {reg['PIN']['base']:>5}  {reg['PIN']['words']} 워드 "
          f"(앞 96 = 엔진, 뒤 64 = pos enc)")

    x_all, p_all, nt_all, sample_idx = [], [], [], []
    # 보드용 : pos enc 는 **PL 의 표**(`Pos_Gather`)에서 모읍니다. 호스트는
    # `pos_idx` 만 보냅니다 — 워드 mt 의 레인 i = 토큰 mt*32+i 의 표 인덱스.
    # 전에는 펼친 이미지가 96.7 MB 였는데 이제 0.5 MB 입니다.
    pidx_all, bidx = [], []
    x_cur = pidx_cur = 0
    bad = 0
    for s in range(n_sample):
        off, T = int(samples[s][0]), int(samples[s][1])
        sample_idx.append([len(nt_all), T, 0, 0])
        for t in range(T):
            o, n = int(index[off + t][0]), int(index[off + t][1])
            rows = tok[o:o + n].astype(np.int16)              # (n, 144)
            xw = lane_words(rows, TOKEN_DIM, n)
            # pos enc : 토큰 i → 테이블 행 pos_idx[i]
            pi = pidx[o:o + n].astype(np.int64)
            pe = pos[pi // POS_GRID, pi % POS_GRID].astype(np.int16)   # (n, 64)
            pw = lane_words(pe, POS_DIM, n)

            pw_idx0 = np.zeros((TT if False else (n + N - 1)//N, N), dtype=np.int16)
            for mt in range((n + N - 1)//N):
                lo, hi = mt*N, min((mt+1)*N, n)
                pw_idx0[mt, :hi-lo] = pi[lo:hi]

            # ---- 자체 검증 : 워드에서 토큰을 되읽어 원본과 대조 ----
            TT = (n + N - 1) // N
            for mt in range(TT):
                lo, hi = mt * N, min((mt + 1) * N, n)
                back = xw[mt * TOKEN_DIM:(mt + 1) * TOKEN_DIM, :hi - lo].T
                if not np.array_equal(back, rows[lo:hi]):
                    bad += 1
                backp = pw[mt * POS_DIM:(mt + 1) * POS_DIM, :hi - lo].T
                if not np.array_equal(backp, pe[lo:hi]):
                    bad += 1
                # pos_idx 도 되읽어 원본과 대조 (PL 이 이 인덱스로 표를 봅니다)
                if not np.array_equal(pw_idx0[mt, :hi - lo], pi[lo:hi]):
                    bad += 1

            TTn = (n + N - 1) // N
            pw_idx = np.zeros((TTn, N), dtype=np.int16)
            for mt in range(TTn):
                lo, hi = mt * N, min((mt + 1) * N, n)
                pw_idx[mt, :hi - lo] = pi[lo:hi]
            pidx_all.append(pw_idx)
            bidx.append([x_cur, xw.shape[0], pidx_cur, TTn, n, 0])
            x_cur    += xw.shape[0]
            pidx_cur += TTn

            x_all.append(xw)
            p_all.append(pw)
            nt_all.append(n)
        if (s + 1) % 50 == 0:
            print(f'  {s+1} 샘플', end='\r')

    X = np.concatenate(x_all)
    P = np.concatenate(p_all)
    X.astype('<i2').tofile(os.path.join(args.dst, 'amem_x.int16.bin'))
    P.astype('<i2').tofile(os.path.join(args.dst, 'amem_pos.int16.bin'))
    np.asarray(nt_all, dtype='<i2').tofile(os.path.join(args.dst, 'n_tok.int16.bin'))
    np.asarray(sample_idx, dtype='<i4').tofile(os.path.join(args.dst, 'samples.int32.bin'))
    labels[:n_sample].astype('<i4').tofile(os.path.join(args.dst, 'labels.int32.bin'))

    # ---- 보드가 그대로 읽는 형태 ----
    PIN_IMG = np.concatenate(pidx_all)
    PIN_IMG.astype('<i2').tofile(os.path.join(args.dst, 'amem_pidx.int16.bin'))
    np.asarray(bidx, dtype='<i4').tofile(os.path.join(args.dst, 'board_index.int32.bin'))
    bsam = [[sample_idx[s][0], sample_idx[s][1], int(labels[s]), 0]
            for s in range(n_sample)]
    np.asarray(bsam, dtype='<i4').tofile(os.path.join(args.dst, 'board_samples.int32.bin'))

    # ---- 시뮬용 : 한 샘플의 전 타임스텝을 **고정 stride** 로 ----
    # 20 타임스텝 TB 는 타임스텝마다 X/PIN 을 새로 채웁니다 (A_Mem 에 한 벌만
    # 들어감). 타임스텝마다 워드 수가 달라 오프셋을 계산하게 하면 TB 가 복잡해지고
    # 틀리기 쉬우므로, **최악치 타일 수(4)로 자리를 고정**해 둡니다. 남는 자리는
    # 0 이고 엔진은 `n_tok` 만큼만 읽으므로 무해합니다.
    ss = args.sim_sample
    TTM = (128 + N - 1) // N                       # TOK_MAX 기준 타일 수
    off_s, T_s = int(samples[ss][0]), int(samples[ss][1])
    xs = np.zeros((T_s * TTM * TOKEN_DIM, N), dtype=np.int16)
    ps = np.zeros((T_s * TTM * POS_DIM, N), dtype=np.int16)
    nt_s = []
    for t in range(T_s):
        o, n = int(index[off_s + t][0]), int(index[off_s + t][1])
        rows = tok[o:o + n].astype(np.int16)
        pi = pidx[o:o + n].astype(np.int64)
        pe = pos[pi // POS_GRID, pi % POS_GRID].astype(np.int16)
        xw, pw = lane_words(rows, TOKEN_DIM, n), lane_words(pe, POS_DIM, n)
        xs[t * TTM * TOKEN_DIM: t * TTM * TOKEN_DIM + len(xw)] = xw
        ps[t * TTM * POS_DIM:  t * TTM * POS_DIM  + len(pw)] = pw
        nt_s.append(n)
    # 시뮬용 pos_idx (고정 stride TTM 워드/타임스텝)
    pxs = np.zeros((T_s * TTM, N), dtype=np.int16)
    for t in range(T_s):
        o, n = int(index[off_s + t][0]), int(index[off_s + t][1])
        pi = pidx[o:o + n].astype(np.int16)
        for mt in range((n + N - 1) // N):
            lo, hi = mt * N, min((mt + 1) * N, n)
            pxs[t * TTM + mt, :hi - lo] = pi[lo:hi]
    with open(os.path.join(args.dst, 'sim_pidx.hex'), 'w') as f:
        for w in pxs:
            f.write(''.join(f'{int(v) & 0xFFFF:04x}' for v in w[::-1]) + '\n')

    for nm, arr in (('sim_x', xs), ('sim_pos', ps)):
        with open(os.path.join(args.dst, f'{nm}.hex'), 'w') as f:
            for w in arr:
                f.write(''.join(f'{int(v) & 0xFFFF:04x}' for v in w[::-1]) + '\n')
    with open(os.path.join(args.dst, 'sim_ntok.hex'), 'w') as f:
        for n in nt_s:
            f.write(f'{n:04x}\n')
    print(f'\n  시뮬용 샘플 {ss} : 타임스텝 {T_s}개  토큰 {nt_s}')
    print(f'    sim_x.hex   {xs.shape[0]:,} 워드 (타임스텝당 {TTM*TOKEN_DIM})')
    print(f'    sim_pos.hex {ps.shape[0]:,} 워드 (타임스텝당 {TTM*POS_DIM})')

    # 첫 타임스텝을 시뮬용 hex 로도 (통합 TB 가 바로 읽습니다)
    n0 = nt_all[0]
    TT0 = (n0 + N - 1) // N
    with open(os.path.join(args.dst, 't0_pidx.hex'), 'w') as f:
        for w in pidx_all[0]:
            f.write(''.join(f'{int(v) & 0xFFFF:04x}' for v in w[::-1]) + '\n')

    for nm, arr, nw in (('t0_x', x_all[0], TT0 * TOKEN_DIM),
                        ('t0_pos', p_all[0], TT0 * POS_DIM)):
        with open(os.path.join(args.dst, f'{nm}.hex'), 'w') as f:
            for w in arr[:nw]:
                f.write(''.join(f'{int(v) & 0xFFFF:04x}' for v in w[::-1]) + '\n')

    print(f'\n-> {args.dst}\n')
    print(f"  amem_x.int16.bin    {X.shape[0]:>9,} 워드  {X.nbytes/1024/1024:>7.1f} MB")
    print(f"  amem_pos.int16.bin  {P.shape[0]:>9,} 워드  {P.nbytes/1024/1024:>7.1f} MB")
    print(f"  amem_pidx.int16.bin {PIN_IMG.shape[0]:>9,} 워드  "
          f"{PIN_IMG.nbytes/1024:>7.1f} KB  (pos_idx — 표는 PL BRAM 에)")
    print(f"  board_index         {len(bidx):>9,} 타임스텝 x 6 int32")
    print(f"  타임스텝 {len(nt_all):,}개  토큰/스텝 {min(nt_all)}~{max(nt_all)}")
    print(f"  샘플0 t0 : 토큰 {n0}개 → X {TT0*TOKEN_DIM} 워드, POS {TT0*POS_DIM} 워드")
    print(f"\n  워드 레이아웃 자체 검증 : {'불일치 %d건 ✗' % bad if bad else '전 타일 일치 ✅'}")


if __name__ == '__main__':
    main()
