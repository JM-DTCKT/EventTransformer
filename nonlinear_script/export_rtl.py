"""골든 벡터 + LUT → RTL 이 읽을 파일

    python export_rtl.py            # 기본 32장
    python export_rtl.py --n 64

## 무엇을 내보내는가

### 1. LUT — 그대로 BRAM 초기화에 씁니다

    gelu.hex     16,384 x 16b   Q4.11 → Q4.11,  코드 [-8192, 8192)
    exp.hex       8,192 x 16b   Q6.9(≤0) → Q1.14
    recip.hex    16,384 x 16b   127·2^22 / S
    rsqrt.hex       256 x 16b   bf16 rsqrt (가수 7b + 지수 홀짝 → 전수)

### 2. 네트워크 골든 — **포맷 경계마다 tap**

MLP 프로젝트에서는 레이어 출력만 봤는데, 여기는 유닛이 많아 그러면 어디가 틀렸는지
못 짚습니다. 그래서 경계마다 찍습니다:

    stem.acc     INT32     GEMM 이 맞나
    stem.bf16    bf16      Requant_Bf16 이 맞나
    *.xhat       Q4.11     LayerNorm 통계 앞단(mean/var/rsqrt)이 맞나
    *.norm.out   Q4.11     LN_Affine 이 맞나
    *.norm.int8  int8      Q4.11 → int8 requant 가 맞나
    *.fc1.q411   Q4.11     INT32 → Q4.11 requant 가 맞나
    *.gelu.q411  Q4.11     GELU LUT 이 맞나
    *.gelu.int8  int8      Q4.11 → int8 requant 가 맞나
    *.res.bf16   bf16      bf16 가산기가 맞나
    head.q69     Q6.9      INT32 → Q6.9 requant 가 맞나
    softmax.u8   uint8     softmax 전체가 맞나

### 3. 유닛 테스트 벡터 — 보드 모드 A 용

16비트 포맷은 입력 공간이 65,536개뿐이라 **전수**가 가능합니다. 유닛 하나만 통과시켜
호스트에서 대조하면, 네트워크와 무관하게 그 모듈만 검증됩니다.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data as data_mod                     # noqa: E402
import formats as F                         # noqa: E402
from quantize import int_forward            # noqa: E402

PT = os.path.join(SCRIPT_DIR, 'ffn_int8.pt')
OUT = os.path.join(SCRIPT_DIR, 'rtl_export')


# =============================================================================
def bf16_bits(t):
    """bf16 값 → 16비트 패턴 (int)."""
    b = t.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    return b


def emit(name, vals, bits, sub=''):
    """한 줄에 하나씩 2의 보수 hex + raw .bin."""
    d = os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    v = np.asarray(torch.as_tensor(vals).reshape(-1).cpu().tolist(), dtype=np.int64)
    mask, nib = (1 << bits) - 1, bits // 4
    with open(os.path.join(d, f'{name}.hex'), 'w') as f:
        for x in v.tolist():
            f.write(f'{(int(x) & mask):0{nib}x}\n')
    dt = {8: np.int8, 16: np.dtype('<i2'), 32: np.dtype('<i4')}[bits]
    v.astype(dt).tofile(os.path.join(d, f'{name}.bin'))
    return int(v.size)


# =============================================================================
def build_rsqrt_lut():
    """bf16 rsqrt 전수 LUT.

    bf16 = 부호1 + 지수8 + 가수7.  v = 2^e · 1.m 이면

        rsqrt(v) = 2^(-(e-p)/2) · rsqrt(2^p · 1.m),    p = (e - bias) mod 2

    지수 홀짝 p 와 가수 m(7비트)만 있으면 되므로 **2 × 128 = 256 엔트리**로 전수입니다.
    반환값은 rsqrt(2^p · 1.m) 의 bf16 비트패턴이고, 결과는 (0.5, 1] 구간입니다.
    호출측은 지수만 따로 더해 주면 됩니다.
    """
    out = []
    for p in (0, 1):
        for m in range(128):
            v = (1.0 + m / 128.0) * (2.0 ** p)
            out.append(int(bf16_bits(torch.tensor(1.0 / np.sqrt(v)))))
    return torch.tensor(out, dtype=torch.long)


def bits_to_bf16(bits):
    """16비트 패턴 → bf16 값 (fp32 컨테이너)."""
    a = (np.asarray(torch.as_tensor(bits).reshape(-1).tolist(), dtype=np.int64)
         & 0xFFFF).astype(np.uint16)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.bfloat16).to(torch.float32)


def check_rsqrt_lut(lut, n=200000):
    """256엔트리 LUT 이 골든 `_rsqrt_bf16` 를 정확히 재현하는지 무작위 확인.

    RTL 은 이 세 줄만 하면 됩니다:  가수 7b + 지수 홀짝으로 LUT 을 읽고, 지수 필드에
    `-(e-p)/2` 를 더한다.
    """
    from quantize import _rsqrt_bf16
    g = torch.Generator().manual_seed(0)
    v = torch.exp(torch.rand(n, generator=g) * 24 - 10)          # 4.5e-5 ~ 3e6
    ref = _rsqrt_bf16(v)

    bits = bf16_bits(F.bf16(v))
    e = ((bits >> 7) & 0xFF) - 127                                # 무편향 지수
    m = bits & 0x7F                                               # 가수
    p = ((e % 2) + 2) % 2                                         # 지수 홀짝
    man_bits = lut[p * 128 + m]                                   # rsqrt(2^p·1.m)
    got_bits = man_bits - (((e - p) // 2) << 7)                   # 지수만 보정
    got = bits_to_bf16(got_bits)
    bad = int((got != ref).sum())
    return bad, n


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pl = torch.load(PT, map_location='cpu', weights_only=False)
    M = args.n
    nb = pl['dims']['n_block']

    # =========================================================================
    # 1. LUT
    # =========================================================================
    rsqrt_lut = build_rsqrt_lut()
    bad, ntest = check_rsqrt_lut(rsqrt_lut)
    print(f'rsqrt LUT (256엔트리) 전수성 확인 : {ntest-bad}/{ntest} 일치'
          f"{'  ✅' if bad == 0 else '  ** 불일치 **'}")

    n_g = emit('gelu',  pl['gelu_lut'],  16, 'lut')
    n_e = emit('exp',   pl['exp_lut'],   16, 'lut')
    n_r = emit('recip', pl['recip_lut'], 16, 'lut')
    n_s = emit('rsqrt', rsqrt_lut,       16, 'lut')
    print(f'LUT : gelu {n_g:,} / exp {n_e:,} / recip {n_r:,} / rsqrt {n_s} 엔트리')
    print(f'      BRAM 환산 {(n_g+n_e+n_r+n_s)*16/36864:.1f} x BRAM36')

    # =========================================================================
    # 2. 가중치 / 파라미터
    # =========================================================================
    L = pl['layers']
    wb = 0
    for k, e in L.items():
        if e.get('kind') == 'layernorm':
            wb += emit(f'{k}_gamma', e['gamma'], 16, 'param') * 2
            wb += emit(f'{k}_beta',  e['beta'],  16, 'param') * 2
            wb += emit(f'{k}_mult',  e['M'],     32, 'param') * 4
        elif e.get('kind') == 'gelu':
            wb += emit(f'{k}_mult', e['M'], 32, 'param') * 4
        else:
            # b_mem[k][n] = w_int[n][k] 전치 — Mac_OS 는 C = A·B 이고 전치를 안 함
            wb += emit(f'{k}_bmem', e['w_int'].long().t().contiguous(), 8, 'param')
            wb += emit(f'{k}_bias', e['b_int'], 32, 'param') * 4
            if e['M'] is None:                       # bf16 소비자 → step 을 bf16 으로
                wb += emit(f'{k}_step', bf16_bits(e['step'].float()), 16, 'param') * 2
            else:
                wb += emit(f'{k}_mult', e['M'], 32, 'param') * 4
    print(f'파라미터 : {wb:,} bytes')

    # =========================================================================
    # 3. 네트워크 골든 (포맷 경계마다 tap)
    # =========================================================================
    _, test_loader = data_mod.loaders(batch_size=M, workers=0)
    x, y = next(iter(test_loader))
    tap = {}
    p = int_forward(pl, x.reshape(x.shape[0], -1), tap=tap)

    WID = {'stem.in': 8, 'stem.acc': 32, 'head.q69': 16, 'softmax.u8': 8}
    for k in list(tap):
        if k.endswith('.int8'):
            WID[k] = 8
        elif k.endswith('.bf16'):
            WID[k] = 16
        elif k.endswith(('.xhat', '.out', '.q411', '.q69')):
            WID[k] = 16
    print(f"\n{'tap':<18}{'형상':>12}{'폭':>4}  포맷")
    fmt_of = {8: 'int8/uint8', 16: 'Q/bf16', 32: 'INT32'}
    for k, v in tap.items():
        w = WID.get(k, 16)
        vv = bf16_bits(v) if k.endswith('.bf16') else v
        emit(k.replace('.', '_'), vv, w, 'golden')
        print(f'{k:<18}{str(tuple(v.shape)):>12}{w:>4}  {fmt_of[w]}')

    emit('labels', y, 32, 'golden')
    acc = float((p.argmax(-1) == y).float().mean())

    # =========================================================================
    # 4. 유닛 테스트 벡터 (보드 모드 A) — 16비트는 전수
    # =========================================================================
    codes = torch.arange(F.I16_MIN, F.I16_MAX + 1, dtype=torch.long)   # 65,536
    emit('in_q411', codes, 16, 'unit')
    emit('gelu_out', F.gelu_q(codes, pl['gelu_lut']), 16, 'unit')

    # softmax: 무작위 + 경계 (전부 같음/한쪽 극단/포화)
    g = torch.Generator().manual_seed(1)
    rows = [torch.randint(-8192, 8192, (256, 10), generator=g)]
    rows.append(torch.zeros(1, 10, dtype=torch.long))                  # 전부 같음 → 균등
    rows.append(torch.tensor([[32767] + [-32768] * 9]))                # 극단
    rows.append(torch.full((1, 10), 32767, dtype=torch.long))          # 전부 포화
    smax_in = torch.cat(rows, dim=0)
    emit('smax_in',  smax_in, 16, 'unit')
    emit('smax_out', F.softmax_u8(smax_in, pl['exp_lut'], pl['recip_lut']), 8, 'unit')

    # bf16 rsqrt: 부호 0 인 bf16 패턴 **전수** 32,768개 = 양수 입력 전 영역.
    # denormal(지수 0)과 Inf/NaN(지수 255)은 하드웨어가 0 을 내도록 정의했습니다.
    rs_in = torch.arange(0, 1 << 15, dtype=torch.long)
    vals = bits_to_bf16(rs_in)
    ok = (rs_in >> 7 > 0) & (rs_in >> 7 < 255)
    rs_out = torch.where(ok, bf16_bits(torch.rsqrt(vals)), torch.zeros_like(rs_in))
    emit('rsqrt_in',  rs_in,  16, 'unit')
    emit('rsqrt_out', rs_out, 16, 'unit')

    # LN affine: xhat/gamma/beta 를 무작위로 (전수는 2^48 이라 불가)
    xh = torch.randint(-32768, 32768, (4096,), generator=g)
    gm = torch.randint(-16384, 16385, (4096,), generator=g)
    bt = torch.randint(-8192, 8192, (4096,), generator=g)
    emit('lnaff_xhat', xh, 16, 'unit'); emit('lnaff_gamma', gm, 16, 'unit')
    emit('lnaff_beta', bt, 16, 'unit')
    emit('lnaff_out', F.ln_affine(xh, gm, bt), 16, 'unit')

    json.dump(dict(n_vectors=M, dims=pl['dims'],
                   fp32_accuracy=pl['fp32_accuracy'], int_accuracy=pl['int_accuracy'],
                   golden_batch_accuracy=acc,
                   n_gelu=n_g, n_exp=n_e, n_recip=n_r, n_rsqrt=n_s,
                   formats=dict(gelu='Q4.11', softmax_in='Q6.9', softmax_out='uint8',
                                ln_xhat='Q4.11', ln_gamma='Q1.14', ln_beta='Q4.11',
                                ln_out='Q4.11', residual='bf16'),
                   ln_shift=F.LN_SHIFT, eps_fp32='0x3727c5ac',
                   layer_cfg={k: dict(kind=e.get('kind', 'linear'),
                                      shift=(None if e.get('shift') is None else int(e['shift'])),
                                      shape=(list(e['shape']) if 'shape' in e else None),
                                      consumer=e.get('consumer'))
                              for k, e in L.items()}),
              open(os.path.join(OUT, 'manifest.json'), 'w'), indent=2)

    print(f'\n-> {OUT}')
    print(f'  골든 배치 {M}장 정확도 : {acc*100:.1f}%')
    print(f'  전체 테스트셋         : fp32 {pl["fp32_accuracy"]*100:.2f}%  ->  '
          f'int {pl["int_accuracy"]*100:.2f}%')
    print(f'  유닛 벡터             : GELU 전수 65,536 / softmax {smax_in.shape[0]}행 / '
          f'LN affine 4,096')


if __name__ == '__main__':
    main()
