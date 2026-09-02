"""EvT(DVS128_10) fpga_export → 보드 온칩 메모리 이미지 + 실행 스케줄

    conda activate evt_new
    python3 pack_evt.py

`fpga_nl/sw/pack_nl.py` 와 같은 역할입니다. 다만 양자화를 여기서 하지 않습니다 —
`quantization/export_fpga.py` 가 만든 `fpga_export/DVS128_10/*.bin` + `manifest.json`
을 **읽어서 재배치만** 합니다. 상수를 새로 계산하면 골든과 어긋날 여지가 생깁니다.

## 메모리

    W_Mem   256b   가중치 b_mem[k][n], **n 으로 뱅킹** (int8 레인 32)
    A_Mem   512b   활성값, 32레인 x 16b (int8 / bf16 / Q4.11 공용)
    Requant_Mem 256b  채널별 {mult|scale, bias} 4쌍/워드   → data/rqmem.bin
    Affine_Mem  256b  LayerNorm {gamma, beta} 8쌍/워드     → data/afmem.bin

`fpga_nl` 과 같은 규칙이라 `Gemm_Core` · `Format_Cast_Act` 가 그대로 붙습니다.

## 레이아웃 규칙 (이게 곧 RTL 스펙)

    W_Mem[w_base + nt*K + k] 레인 j = w_int[nt*32+j][k]      ← 전치
    A_Mem[a_base + mt*K + k] 레인 i = x[mt*32+i][k]

`Gemm_Core` 는 A·B 둘 다 "워드 = reduce 인덱스, 레인 = non-reduce" 로 읽습니다.
그래서 attention 의 `Q·Kᵀ`/`attn·V` 도 같은 규칙에 그대로 들어맞습니다
(자세한 것은 `rtl/gemm_core/Gemm_Core.v` 머리말).

## positional encoding 을 호스트가 미리 붙입니다

`preproc` 의 입력 160 = projection 출력 96 + fourier pos enc 64 입니다. pos enc 는
토큰마다 **다른 테이블 행**을 읽어야 하는데, A_Mem 워드 하나는 32레인이 서로 다른
토큰이므로 **레인마다 다른 주소**가 됩니다. 하드웨어로 하면 레인당 게더가 필요합니다.

pos enc 는 `pos_idx` 만으로 정해지고 테이블은 고정이므로 **호스트가 미리 워드
레이아웃으로 펴서** 보냅니다. 회로가 하나 없어지고, DDR 은 4 GB 라 부담이 안 됩니다.

## 실행 스케줄

타임스텝 하나가 아래를 돕니다. `T` (≤20) 만큼 반복하며 latent 를 누적합니다.

     0  GEMM  event_projection  144→96   (토큰) → Q4.11 → GELU → int8
     1  GEMM  preproc           160→128  (토큰) → Q4.11 → GELU → int8
     2  GEMM  proc_events.1     128→128  → ReLU → int8
     3  GEMM  proc_events.4     128→128  → ReLU → int8, + x (residual)
    -- attention 블록 3개 (cross 1 + latent 2), 블록마다 --
     a  LN    layer_norm_x / layer_norm_1
     b  GEMM  in_proj 384x128  → Q/K/V int8  (V 는 Transpose32 경유)
     c  ATTN  head 4개 : Q·Kᵀ → Q6.9 → softmax → uint8 → ·V → int8
     d  GEMM  out_proj → bf16, + residual
     e  LN    layer_norm_att → linear1 → GELU → LN2 → linear2 → GELU → linear3 → +
    -- 타임스텝 끝 : latent += z --
    끝에 한 번  LN → proc_embs.linear1 → ReLU → latent 96 평균 → CLF → argmax
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# fpga_export 를 가진 저장소. 환경변수 EVT_ROOT 로 덮어쓸 수 있다.
EVT = os.environ.get('EVT_ROOT', '/hai/home/sjm/EvT_quant')
# 기본은 재학습 DVS128_10 의 A8W8 export. --export 로 다른 것을 지정할 수 있다
# (예: pretrained + A8W4 = 'paper_DVS128_10__a8w4').
EXPORT = os.path.join(EVT, 'quantization', 'fpga_export', 'DVS128_10')
DEF_DST = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))

sys.path.insert(0, os.path.join(EVT, 'quantization'))
from export_fpga import make_requant                 # noqa: E402  상수식은 한 곳에서만

N = 32                                   # PE 배열 변 = 레인 수
E, LATENT, HEADS, HEAD_DIM = 128, 96, 4, 32
POS_DIM = 64
N_CLASS = 10

# (명령어 KIND/FMT 상수는 `schedule_evt.py` 가 갖고 있습니다 — 여기서는 안 씁니다)


BLOCKS = ['backbone.proc_memory_blocks.0.cross_attention',
          'backbone.proc_memory_blocks.0.latent_attentions.0',
          'backbone.proc_memory_blocks.0.latent_attentions.1']

# ---------------------------------------------------------------------------
# GELU 뒤 재양자화 : 어느 레이어의 Q4.11 출력이 **어느 격자로** 떨어지나
#
#   골든은 `s_out(F.gelu(s_in(z)))` 로 Q4.11 실수를 내고, 그 다음 소비자가
#   자기 입력 scale 로 int8 화합니다. 그래서 목표 격자는 **다음 레이어의
#   input scale** 입니다. 여기서만 적어 두고 값은 manifest 에서 읽습니다.
#
#   `linear1` 만 예외입니다 — 다음이 `layer_norm_2` 라 int8 격자가 없습니다.
#   LayerNorm 은 스케일 불변이므로 하드웨어는 **Q4.11 16비트를 그대로 저장**하고
#   LN 입력단에서 int→bf16 변환만 합니다 (골든과 완전히 동일).
#   → 재양자화 없음. 스케줄러가 FLAG "raw16" 로 표시합니다.
# ---------------------------------------------------------------------------
PREPROC = 'backbone.preproc_block_events.seq_init.0'

GELU_NEXT = {
    'backbone.event_projection.seq_init.0':
        'backbone.preproc_block_events.seq_init.0',
    'backbone.preproc_block_events.seq_init.0':
        'backbone.proc_event_blocks.0.seq_init.1',
}
for _b in BLOCKS:
    GELU_NEXT[f'{_b}.linear2'] = f'{_b}.linear3'
GELU_RAW16 = {f'{_b}.linear1' for _b in BLOCKS}      # 재양자화 없이 Q4.11 저장

# ---------------------------------------------------------------------------
# LayerNorm 뒤 재양자화 : Q4.11 출력 → 다음 GEMM 이 읽는 int8 격자
#
#   LayerNorm 도 GELU 와 같습니다 — 골든은 Q4.11 실수를 내고 다음 소비자가 자기
#   입력 scale 로 int8 화합니다. `layer_norm_x` 의 소비자는 in_proj 의 K·V 밴드,
#   `layer_norm_1` 은 Q 밴드입니다 (K 와 V 의 입력 scale 은 **같습니다** — 골든의
#   `k_in_scales`/`v_in_scales` 가 세 블록 모두 동일해, 하드웨어가 LNX 를 한 벌만
#   저장해도 됩니다).
# ---------------------------------------------------------------------------
LN_NEXT = {'backbone.proc_embs_block.layer_norm':
           ('layer', 'backbone.proc_embs_block.linear1')}
for _b in BLOCKS:
    LN_NEXT[f'{_b}.layer_norm_x']   = ('band', f'{_b}.attention', 'K')
    LN_NEXT[f'{_b}.layer_norm_1']   = ('band', f'{_b}.attention', 'Q')
    LN_NEXT[f'{_b}.layer_norm_att'] = ('layer', f'{_b}.linear1')
    LN_NEXT[f'{_b}.layer_norm_2']   = ('layer', f'{_b}.linear2')
LN_LSB = 2.0 ** -11                                  # LayerNorm 출력도 Q4.11

GELU_LSB = 2.0 ** -11                                # Q4.11

# ---------------------------------------------------------------------------
# 활성함수 뒤 **두 번째** int8→int8 재양자화
#
# 골든은 ReLU 앞뒤로 두 번 양자화합니다 — 앞은 ReLU 사이트의 격자, 뒤는 다음
# GEMM 의 입력 격자입니다. 둘이 다른 자리가 있습니다:
#
#     seq_init.1   0.036535 (ReLU 입력) → 0.020563 (seq_init.4 입력)  x1.777
#     clf.linear_1 0.119583               → 0.115334                  x1.037
#
# 뒤 격자가 **더 촘촘해서** 코드가 127 을 넘어 포화하는 자리까지 골든과 같아야
# 하므로, 한 번에 접어 넣을 수 없습니다. 하드웨어도 두 번 합니다 — 곱수는
# 레이어 뒤 빈 칸(`RQ_BASE + NOUT`, GELU 스칼라와 같은 자리)에 넣고 SHIFT2 를 씁니다.
# ---------------------------------------------------------------------------
# `proc_embs_block` 의 gap(latent 96개 평균) 은 **평균의 나누기까지 곱수에**
# 접습니다 — 엔진은 32레인 합을 타일마다 누적한 뒤 이 곱수 하나로 끝냅니다.
MEAN_NEXT = ('backbone.proc_embs_block.linear1', 'models_clf.0.linear_1', 96)

REQ2_NEXT = {
    'backbone.proc_event_blocks.0.seq_init.1':
        'backbone.proc_event_blocks.0.seq_init.4',
    'models_clf.0.linear_1': 'models_clf.0.linear_2',
}


def bf16_bits(x):
    """float → bf16 16비트 (RNE). `Requant_Bf16` 이 읽는 상수 형식."""
    b = struct.unpack('<I', struct.pack('<f', float(x)))[0]
    b += 0x7FFF + ((b >> 16) & 1)
    return (b >> 16) & 0xFFFF


def rd(name, dtype):
    p = os.path.join(EXPORT, name)
    return np.fromfile(p, dtype=dtype)


def emit_words(dst, name, words, bits):
    """워드 리스트(레인 값 리스트) → .bin + .hex

    `bits=8` 은 레인당 1바이트입니다. `bits=4` 는 **니블 팩** — 레인 j 를 워드의
    비트 `[4j +: 4]` 에 넣습니다 (짝수 레인이 하위 니블). RTL 의

        gemm_b_from_w[j*8 +: 8] = sext(w_rd_data[j*4 +: 4])

    와 한 벌이라, 여기 순서를 바꾸면 가중치가 조용히 뒤섞입니다.
    """
    nb = N * bits // 8
    with open(os.path.join(dst, f'{name}.bin'), 'wb') as fb, \
         open(os.path.join(dst, f'{name}.hex'), 'w') as fh:
        for w in words:
            if bits == 4:
                assert all(-8 <= int(v) <= 7 for v in w), 'int4 범위를 벗어난 가중치'
                raw = bytes((int(w[i]) & 0xF) | ((int(w[i + 1]) & 0xF) << 4)
                            for i in range(0, len(w), 2))
            else:
                raw = b''.join(int(v).to_bytes(bits // 8, 'little', signed=True)
                               for v in w)
            assert len(raw) == nb, (len(raw), nb)
            fb.write(raw)
            fh.write(raw[::-1].hex() + '\n')
    return len(words)


def emit_raw(dst, name, words_bytes):
    with open(os.path.join(dst, f'{name}.bin'), 'wb') as fb, \
         open(os.path.join(dst, f'{name}.hex'), 'w') as fh:
        for raw in words_bytes:
            fb.write(raw)
            fh.write(raw[::-1].hex() + '\n')
    return len(words_bytes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dst', default=DEF_DST)
    ap.add_argument('--export', default=None,
                    help='fpga_export 하위 디렉토리 이름 (기본 DVS128_10)')
    ap.add_argument('--w_pack_bits', type=int, default=8,
                    help='wmem 레인 폭. 8 = 기존 레이아웃, 4 = 니블 팩 (RTL 변경 필요)')
    args = ap.parse_args()
    global EXPORT
    if args.export:
        EXPORT = os.path.join(EVT, 'quantization', 'fpga_export', args.export)
    assert os.path.isdir(EXPORT), f'missing {EXPORT}'
    print(f'export: {EXPORT}')
    os.makedirs(args.dst, exist_ok=True)

    mf = json.load(open(os.path.join(EXPORT, 'manifest.json')))
    layers = {L['name']: L for L in mf['layers']}
    attn = {a['block']: a for a in mf['attention']}
    fxp = {f['name']: f for f in mf['fx_params']}

    # =========================================================================
    # W_Mem : 워드[w_base + nt*K + k] 레인 j = w_int[nt*32+j][k]   (전치)
    # =========================================================================
    w_words, w_base, w_shape = [], {}, {}
    for name, L in layers.items():
        Eo, Ei = L['shape']
        w = rd(L['weight_file'], np.int8).astype(np.int64).reshape(Eo, Ei)
        w_base[name] = len(w_words)
        w_shape[name] = (Eo, Ei)
        for nt in range((Eo + N - 1) // N):
            for k in range(Ei):
                w_words.append([int(w[nt * N + j, k]) if nt * N + j < Eo else 0
                                for j in range(N)])

    # =========================================================================
    # Requant_Mem : 채널별 {mult, bias} 8바이트, 4채널/워드
    #   bf16 소비자는 mult 자리에 bf16 scale 을 넣습니다 (fpga_nl 과 같은 규칙)
    # =========================================================================
    rq_pairs, rq_base = [], {}
    gelu_mult, gelu_shift, gelu_target = {}, {}, {}
    for name, L in layers.items():
        Eo, Ei = L['shape']
        rq_base[name] = len(rq_pairs)
        b = rd(L['bias_file'], np.int32).astype(np.int64)
        rq = L.get('requant', {})
        if 'mult_file' in rq:
            M = rd(rq['mult_file'], np.int32).astype(np.int64)
        elif 'scale_file' in rq:
            # bf16 소비자 : 곱수 대신 **bf16 scale 16비트**가 들어갑니다.
            # 필드가 `requant` 밑에 있습니다 — `output` 밑이 아닙니다.
            M = rd(rq['scale_file'], np.uint16).astype(np.int64)
        else:
            raise SystemExit(f'{name}: requant 에 mult_file/scale_file 이 없습니다')
        assert len(b) == Eo and len(M) == Eo, (name, len(b), len(M), Eo)
        # 곱수가 0 이면 그 채널은 출력이 통째로 0 이 됩니다. 조용히 지나가면
        # 보드에서야 드러나므로 여기서 막습니다.
        if int((M == 0).sum()):
            raise SystemExit(f'{name}: 곱수 0 인 채널 {int((M==0).sum())}개')
        for c in range(Eo):
            rq_pairs.append((int(M[c]), int(b[c])))
        # ---- 채널 바로 뒤 한 칸 = GELU 뒤 int8 재양자화 곱수 ----
        # 엔진이 명령어 시작 시 `RQ_BASE + NOUT` 을 한 번 읽습니다(`ST_CONST`). 레이어를
        # 빈틈없이 붙여 담으면 그 자리가 **다음 레이어의 채널 0** 이 됩니다 —
        # 통합 TB 에서 결과가 −128 로 포화해 드러났습니다. 모든 레이어 뒤에
        # 한 칸씩 두어 자리를 고정합니다(GELU 가 없는 레이어는 안 읽힘).
        if name in GELU_NEXT:
            tgt = layers[GELU_NEXT[name]]['input']['step']
            Mg, kg = make_requant([GELU_LSB / tgt])
            gelu_mult[name], gelu_shift[name] = int(Mg[0]), int(kg)
            gelu_target[name] = float(tgt)
        elif name == MEAN_NEXT[0]:
            tgt = layers[MEAN_NEXT[1]]['input']['step']
            Mg, kg = make_requant([L['output']['lsb'] / (MEAN_NEXT[2] * tgt)])
            gelu_mult[name], gelu_shift[name] = int(Mg[0]), int(kg)
            gelu_target[name] = float(tgt)
        elif name in REQ2_NEXT:
            tgt = layers[REQ2_NEXT[name]]['input']['step']
            Mg, kg = make_requant([L['output']['lsb'] / tgt])
            gelu_mult[name], gelu_shift[name] = int(Mg[0]), int(kg)
            gelu_target[name] = float(tgt)
        else:
            gelu_mult[name], gelu_shift[name] = 1, 0
        rq_pairs.append((gelu_mult[name], 0))

    # attention 의 QK / AV 는 **블록당 스칼라 1개**입니다
    attn_rq = {}
    for blk, a in attn.items():
        attn_rq[blk] = dict(
            qk=(len(rq_pairs), int(a['QK']['mult']), int(a['QK']['shift'])))
        rq_pairs.append((int(a['QK']['mult']), 0))
        attn_rq[blk]['av'] = (len(rq_pairs), int(a['AV']['mult']),
                              int(a['AV']['shift']))
        rq_pairs.append((int(a['AV']['mult']), 0))

    # ---- LayerNorm 뒤 재양자화 (스칼라 1개씩) ----
    ln_rq, ln_shift, ln_target = {}, {}, {}
    for ln, tgt in LN_NEXT.items():
        if tgt[0] == 'layer':
            scale = layers[tgt[1]]['input']['step']
        else:                                    # in_proj 밴드
            bands = layers[tgt[1]]['input']['bands']
            scale = next(b['step'] for b in bands if b['name'] == tgt[2])
        Mn, kn = make_requant([LN_LSB / scale])
        ln_rq[ln], ln_shift[ln], ln_target[ln] = len(rq_pairs), int(kn), float(scale)
        rq_pairs.append((int(Mn[0]), 0))

    while len(rq_pairs) % 4:
        rq_pairs.append((0, 0))
    rq_words = []
    for i in range(0, len(rq_pairs), 4):
        raw = b''
        for mv, bv in rq_pairs[i:i + 4]:
            raw += struct.pack('<i', mv if mv < 2**31 else mv - 2**32)
            raw += struct.pack('<i', bv)
        rq_words.append(raw)

    # ---- LayerNorm 코어의 고정소수점 창 (`in_shift`) ----
    # 새 코어(`rtl/layernorm/LayerNorm_Unit`)는 내부가 Q8.15(±256) 입니다. 입력을
    # `value * 2^xsh` 로 옮겨 넣는데, 이 값은 LayerNorm 이 스케일 불변이라
    # **정확도가 아니라 정밀도**만 바꿉니다.
    #
    # 골든의 캘리브레이션 Qm.n 에서 유추하면 **안 됩니다.** 그 격자는 20 타임스텝
    # 누적 최악치라 실제 분포보다 수천 배 넓습니다 — `layer_norm_1` 은 Q16.-1
    # (범위 65536)로 잡혀 있지만 실측 입력은 **0.74** 입니다. 그대로 xsh=-9 를
    # 주면 정밀도가 5비트로 줄어 결과가 무너집니다 (실제로 그렇게 나왔습니다).
    #
    # 그래서 골든 덤프에서 잰 실측치로 정합니다 (`data/golden/*.hex`):
    #
    #     latinit (ln_1 입력)     0.74      EV   (ln_x 입력)   2.52
    #     ZATT    (ln_att 입력)   5.47      LATV (embs.ln)     7.94  (1 타임스텝)
    #
    # 블록 안 LayerNorm 들은 잔차 스트림이라 타임스텝이 지나도 이 범위에 머뭅니다.
    # `proc_embs_block.layer_norm` 만 **20 타임스텝 누적**을 받으므로 20배를 봅니다.
    #   블록 LN : max ~16 → 16*2^2 = 64  (여유 2비트)
    #   embs LN : max ~160 → 160*2^-2 = 40
    LN_XSH_BLOCK, LN_XSH_EMBS = 2, -2
    ln_shift_map = {}
    for ln in LN_NEXT:
        ln_shift_map[ln] = (LN_XSH_EMBS if ln.endswith('proc_embs_block.layer_norm')
                            else LN_XSH_BLOCK)

    # =========================================================================
    # Affine_Mem : LayerNorm {gamma(Q1.14), beta(Q4.11)} 4바이트, 8특징/워드
    # =========================================================================
    af_pairs, af_base = [], {}
    ln_names = sorted({f['name'].rsplit('.', 1)[0] for f in mf['fx_params']
                       if f['name'].endswith(('.weight', '.bias'))})
    for ln in ln_names:
        gw, gb = fxp.get(ln + '.weight'), fxp.get(ln + '.bias')
        if gw is None or gb is None:
            continue
        g = rd(gw['file'], np.int16).astype(np.int64)
        be = rd(gb['file'], np.int16).astype(np.int64)
        af_base[ln] = len(af_pairs)
        for i in range(len(g)):
            af_pairs.append((int(g[i]), int(be[i])))
    while len(af_pairs) % 8:
        af_pairs.append((0, 0))
    af_words = []
    for i in range(0, len(af_pairs), 8):
        raw = b''
        for gv, bv in af_pairs[i:i + 8]:
            raw += struct.pack('<h', gv) + struct.pack('<h', bv)
        af_words.append(raw)

    # =========================================================================
    # 자체 검증 : W_Mem 레이아웃으로 골든 acc 를 재계산
    #   손으로 적은 전치가 틀리면 여기서 걸립니다. `pack_nl.py` 와 같은 장치입니다.
    # =========================================================================
    bad = []
    rng = np.random.default_rng(0)
    for name, L in layers.items():
        Eo, Ei = L['shape']
        w = rd(L['weight_file'], np.int8).astype(np.int64).reshape(Eo, Ei)
        x = rng.integers(-128, 128, size=Ei)
        for c in (0, Eo // 2, Eo - 1):
            nt, j = c // N, c % N
            s = sum(int(x[k]) * w_words[w_base[name] + nt * Ei + k][j]
                    for k in range(Ei))
            ref = int(x @ w[c])
            if s != ref:
                bad.append(f'{name}[{c}]')
    if bad:
        raise SystemExit(f'W_Mem 레이아웃 불일치: {bad[:6]}')

    # =========================================================================
    # 출력
    # =========================================================================
    nw = emit_words(args.dst, 'wmem', w_words, args.w_pack_bits)
    n_rq = emit_raw(args.dst, 'rqmem', rq_words)
    n_af = emit_raw(args.dst, 'afmem', af_words)

    # =========================================================================
    # positional encoding 테이블 — **소비자 격자로 옮겨 담습니다**
    #
    # export 된 표는 자기 자신의 scale(0.010073) 위에 있습니다. 하지만 하드웨어의
    # A_Mem 한 행은 `preproc` 의 입력 벡터 160개 전부이고, GEMM 은 **입력 scale 이
    # 하나**입니다 (`requant` 의 M[c] 에 그 하나가 접혀 있음). 골든도 마찬가지로
    # `cat([gelu_out, pos_embs])` 를 통째로 preproc 의 입력 scale 로 양자화합니다.
    #
    #     code_hw = round(code_tbl * scale_tbl / scale_preproc)      ratio 0.1189
    #
    # 이걸 빠뜨리면 pos 64워드가 통째로 8배 크게 들어가 통합 TB 에서 PIN 의 뒤쪽
    # 64워드가 전부 틀립니다 (앞쪽 96워드는 맞으므로 바로 짚힙니다).
    # =========================================================================
    pe = mf['pos_encoding']
    pos_tbl = rd(pe['file'], np.int8).astype(np.float64).reshape(pe['shape'])
    pos_scale = layers[PREPROC]['input']['step']
    pos = np.clip(np.round(pos_tbl * pe['step'] / pos_scale), -128, 127).astype(np.int8)
    pos.tofile(os.path.join(args.dst, 'posenc.int8.bin'))

    # ---- PL 온칩 표 (`Pos_Gather`) ----
    # 행 = pos_idx = (y//6)*21 + (x//6), 한 행이 64특징 = 64바이트.
    # 이걸 BRAM 에 두면 호스트가 타임스텝마다 펴서 보내던 96.7 MB 가 없어지고
    # `pos_idx`(토큰당 2 B) 만 오면 됩니다.
    G = pe['shape'][0]
    pos_rows = pos.reshape(G * G, POS_DIM)
    with open(os.path.join(args.dst, 'posmem.bin'), 'wb') as fb, \
         open(os.path.join(args.dst, 'posmem.hex'), 'w') as fh:
        for r in range(G * G):
            raw = bytes((int(v) & 0xFF) for v in pos_rows[r])
            fb.write(raw)
            fh.write(raw[::-1].hex() + '\n')
    print(f"  posmem  {G*G} 행 x {POS_DIM} B = {G*G*POS_DIM/1024:.1f} KB "
          f"(PL BRAM, `Pos_Gather`)")
    print(f"\npos enc 재양자화 : step {pe['step']:.6g} → {pos_scale:.6g} "
          f"(x{pe['step']/pos_scale:.4f})  코드 범위 {pos.min()}~{pos.max()}")

    # =========================================================================
    # latent 초기값 (`backbone.memory_vertical`) → A_Mem 워드 레이아웃 bf16
    #
    # `EvT.forward` 는 `inp_q` 와 `latent_vectors` **둘 다** 이 값으로 시작해
    #   inp_q = block(...);  latent_vectors += inp_q
    # 를 돕니다. 하드웨어의 Z 와 LATV 영역에 같은 이미지를 넣습니다.
    # =========================================================================
    lv = next(f for f in mf['fx_params'] if f['name'] == 'backbone.memory_vertical')
    lat = rd(lv['file'], np.int16).astype(np.float64).reshape(lv['shape'])
    lat = lat * (2.0 ** -lv['frac_bits'])
    n_lat, E_lat = lat.shape
    lat_words = []
    for mt in range((n_lat + N - 1) // N):
        for k in range(E_lat):
            lat_words.append([bf16_bits(lat[mt * N + j, k]) if mt * N + j < n_lat
                              else 0 for j in range(N)])
    with open(os.path.join(args.dst, 'latinit.bin'), 'wb') as fb, \
         open(os.path.join(args.dst, 'latinit.hex'), 'w') as fh:
        for w in lat_words:
            raw = b''.join(int(v).to_bytes(2, 'little') for v in w)
            fb.write(raw)
            fh.write(raw[::-1].hex() + '\n')

    # =========================================================================
    # bias_k / bias_v — attention 의 **추가 키/값 토큰** (int8 코드)
    #   K 영역 : head 마다 예약 칸 1개, 레인 = head_dim
    #   V 영역 : head 마다 예약 칸 1개, 레인 = head_dim   (같은 모양)
    # =========================================================================
    # 순서는 스케줄의 BKV 영역과 **한 벌** : [블록][k, v][head], 레인 = head_dim
    bkv_words = []
    for blk in BLOCKS:
        L = layers[blk + '.attention']
        for bk in ('bias_k', 'bias_v'):
            v = rd(L[f'{bk}_int8_file'], np.int8).astype(np.int64)
            for h in range(HEADS):
                bkv_words.append([int(v[h * HEAD_DIM + d]) for d in range(N)])
    with open(os.path.join(args.dst, 'bkv.bin'), 'wb') as fb, \
         open(os.path.join(args.dst, 'bkv.hex'), 'w') as fh:
        for w in bkv_words:
            raw = b''.join(int(x).to_bytes(2, 'little', signed=True) for x in w)
            fb.write(raw)
            fh.write(raw[::-1].hex() + '\n')

    cfg = dict(
        dataset='DVS128_10', N=N, E=E, LATENT=LATENT,
        HEADS=HEADS, HEAD_DIM=HEAD_DIM, N_CLASS=N_CLASS,
        T_MAX=20, TOK_MAX=128,
        words=dict(w=nw, pb=n_rq, pg=n_af),
        w_base=w_base, w_shape={k: list(v) for k, v in w_shape.items()},
        rq_base=rq_base, af_base=af_base, attn_rq=attn_rq,
        ln_rq=ln_rq, ln_shift=ln_shift, ln_target=ln_target,
        ln_xsh=ln_shift_map,
        gelu_shift=gelu_shift, gelu_mult=gelu_mult, gelu_target=gelu_target,
        gelu_raw16=sorted(GELU_RAW16), req2=sorted(REQ2_NEXT),
        mean_layer=MEAN_NEXT[0],
        input_scales={n: L['input']['step'] for n, L in layers.items()
                     if 'step' in L.get('input', {})},
        shifts={n: L.get('requant', {}).get('shift') for n, L in layers.items()},
        in_proj_bands={n: L['requant']['bands'] for n, L in layers.items()
                       if L['kind'] == 'in_proj'},
        bkv=dict(file='bkv.bin', words=len(bkv_words),
                 order='[block][k,v][head], lane = head_dim'),
        latent_init=dict(file='latinit.bin', shape=[n_lat, E_lat],
                         words=len(lat_words)),
        pos_encoding=dict(shape=pe['shape'], scale=pos_scale,
                          table_scale=pe['step'], file='posenc.int8.bin',
                          pl_table=dict(file='posmem.bin', rows=pe['shape'][0]**2,
                                        feat=POS_DIM)),
        input_scale=layers['backbone.event_projection.seq_init.0']['input']['step'],
    )
    json.dump(cfg, open(os.path.join(args.dst, 'config.json'), 'w'), indent=1)

    print(f'-> {args.dst}\n')
    print(f"{'mem':>6} {'워드':>8} {'KB':>8}")
    print(f"{'W':>6} {nw:>8,} {nw*32/1024:>8.1f}")
    print(f"{'RQ':>6} {n_rq:>8,} {n_rq*32/1024:>8.1f}   (채널 {len(rq_pairs):,}개)")
    print(f"{'AF':>6} {n_af:>8,} {n_af*32/1024:>8.1f}   (LayerNorm {len(af_base)}개)")
    print(f"{'POS':>6} {'':>8} {pos.nbytes/1024:>8.1f}   {tuple(pos.shape)}")
    print(f"{'LATINIT':>6} {len(lat_words):>8,} {len(lat_words)*64/1024:>8.1f}   "
          f"memory_vertical {tuple(lat.shape)} → bf16")
    print(f"{'BKV':>6} {len(bkv_words):>8} {len(bkv_words)*64/1024:>8.1f}   "
          f"bias_k/bias_v int8 ([블록][k,v][head])")
    print(f'\n레이어 {len(layers)}개  (linear {sum(1 for L in layers.values() if L["kind"]=="linear")}'
          f', in_proj {sum(1 for L in layers.values() if L["kind"]=="in_proj")})')
    print(f'attention 블록 {len(attn)}개')
    print('\nW_Mem 레이아웃 자체 검증 : 전 레이어 표본 재계산 일치 ✅')
    print('\nGELU 뒤 재양자화 (채널 뒤 한 칸, `RQ_BASE + NOUT`)')
    for n2 in GELU_NEXT:
        err = abs(gelu_mult[n2] / 2.0**gelu_shift[n2]
                  - GELU_LSB / gelu_target[n2]) / (GELU_LSB / gelu_target[n2])
        print(f'  {n2:<58} scale {gelu_target[n2]:.6g}  '
              f'M={gelu_mult[n2]} sh={gelu_shift[n2]}  상대오차 {err:.2e}')
    for n2 in sorted(GELU_RAW16):
        print(f'  {n2:<58} 재양자화 없음 (Q4.11 16b 저장 → LayerNorm)')
    print('\n활성함수 뒤 2차 재양자화 (같은 칸, GSH 사용)')
    for n2 in REQ2_NEXT:
        print(f'  {n2:<58} {layers[n2]["output"]["lsb"]:.6g} → '
              f'{gelu_target[n2]:.6g}  M={gelu_mult[n2]} sh={gelu_shift[n2]}')
    print(f'  {MEAN_NEXT[0]:<58} 평균({MEAN_NEXT[2]}) + 재양자화  '
          f'M={gelu_mult[MEAN_NEXT[0]]} sh={gelu_shift[MEAN_NEXT[0]]}')
    print('\nLayerNorm 고정소수점 창 (새 코어의 in_shift, 실측 범위 기준)')
    for k2 in sorted(ln_shift_map):
        print(f'  {k2:<58} xsh={ln_shift_map[k2]:>3}')
    print('\nLayerNorm 뒤 재양자화 (Requant_Mem 스칼라 1개씩)')
    for ln in LN_NEXT:
        print(f'  {ln:<58} scale {ln_target[ln]:.6g}  '
              f'RQ[{ln_rq[ln]}] sh={ln_shift[ln]}')


if __name__ == '__main__':
    main()
