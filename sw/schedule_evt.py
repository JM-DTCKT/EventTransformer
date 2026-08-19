"""EvT 실행 스케줄 + A_Mem 주소 계획 — RTL 을 쓰기 전에 여기서 확정합니다

    python3 schedule_evt.py            # 스케줄 출력 + 자체 검증
    python3 schedule_evt.py --tok 123  # 최악 토큰 수로

## 왜 파이썬이 먼저인가

`fpga_nl` 의 `FFN_Engine` 은 12 step 짜리였는데도 위상·주소 버그가 5개 나왔습니다.
EvT 는 타임스텝 하나가 **step 40개 남짓**이고 attention 3블록 x head 4 가 들어갑니다.
주소식을 RTL 안에서 처음 맞추려 하면 디버깅 표면이 감당이 안 됩니다.

그래서 **주소 계획과 실행 순서를 먼저 파이썬으로 확정**하고, 여기서
 ① 영역 겹침 없음  ② 모든 피연산자가 쓰이기 전에 생산됨  ③ 메모리 상한
을 검사한 다음 RTL 은 이 표를 실행만 하게 합니다.

## `Gemm_Core_Ev` 의 주소 규칙 (바꿀 수 없는 제약)

    A 워드 = a_base + mt*K + k      레인 i = A[mt*32+i][k]
    B 워드 = b_base + nt*K + k      레인 j = B[k][nt*32+j]

**둘 다 워드가 reduce 인덱스, 레인이 non-reduce 인덱스**입니다. attention 을 이
틀에 넣으면 저장 레이아웃이 자동으로 정해집니다.

## attention 이 head-major 여야 하는 이유

`Q·Kᵀ` 는 reduce 가 head_dim(32)입니다. 그러면 B 의 타일 간격이 `K=32` 여야 하는데
(`b_base + nt*32 + d`), K 를 채널순(`Kbase + kt*128 + c`)으로 저장하면 간격이 128 이라
맞지 않습니다. 그래서 Q/K 는 **head → 타일 → d** 순으로 저장합니다:

    Q[h][mt][d] = Q_base + h*(QT*32) + mt*32 + d       QT = ceil(96/32) = 3
    K[h][kt][d] = K_base + h*(KT*32) + kt*32 + d       KT = ceil(Lk/32)

이러면 head h, 행타일 mt 의 QK GEMM 은

    M=32, K=32, Nout=Lk
    a_base = Q_base + h*QT*32 + mt*32
    b_base = K_base + h*KT*32

로 끝나고 코어는 손댈 게 없습니다.

## softmax 가 GEMM 을 끊는 이유

`Tile_Ctrl` 은 `for mt { for nt }` 라 행 타일이 바깥입니다. QK GEMM 전체가 ~880
사이클인데 `Softmax_Attn` 은 행 타일 하나에 ~4·N·Lk (Lk=53 이면 6,800) 사이클이라,
GEMM 이 다음 행 타일로 넘어가는 동안 softmax 는 아직 첫 타일을 처리 중입니다.
그래서 **행 타일마다 GEMM → softmax → AV 를 끊어** 돌립니다 (M=32 로 호출).

## 소비자 코드

    0 INT8(+ReLU)   1 Q4.11→GELU→int8   2 bf16   3 Q6.9(softmax 직결)
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))

N = 32
E, LAT, HEADS, HD, NCLS = 128, 96, 4, 32, 10
TOK_MAX = 128

C_INT8, C_Q411, C_BF16, C_Q69 = 0, 1, 2, 3
K_GEMM, K_LN, K_SMAX, K_RES, K_MEAN, K_ARGMAX, K_POS = 0, 1, 2, 3, 4, 5, 6
KIND = {K_GEMM: 'GEMM', K_LN: 'LN', K_SMAX: 'SMAX', K_RES: 'RES',
        K_MEAN: 'MEAN', K_ARGMAX: 'ARGM', K_POS: 'POS'}
CONS = {C_INT8: 'int8', C_Q411: 'Q4.11+GELU', C_BF16: 'bf16', C_Q69: 'Q6.9'}

BLOCKS = ['backbone.proc_memory_blocks.0.cross_attention',
          'backbone.proc_memory_blocks.0.latent_attentions.0',
          'backbone.proc_memory_blocks.0.latent_attentions.1']


# proc_events.seq_init.4 의 출력 격자 = ReLU 입력 사이트(manifest nonlinear_formats)
SEQ5_STEP = 0.019990921020507812


RSH = 20                     # `EvT_Engine.v` 의 localparam RSH 와 같은 값


def fx_step(x):
    """step → round(step * 2^RSH) (16비트에 들어가야 함)"""
    v = int(round(float(x) * (1 << RSH)))
    assert 0 < v < (1 << 16), (x, v)
    return v


def bf16_bits(x):
    """float → bf16 16비트 (RNE)"""
    import struct
    b = struct.unpack('<I', struct.pack('<f', float(x)))[0]
    b += 0x7FFF + ((b >> 16) & 1)
    return (b >> 16) & 0xFFFF


def tiles(n):
    return (n + N - 1) // N


class Arena:
    """A_Mem 영역 할당기 — 겹침을 만들지 않기 위해 손으로 주소를 안 적습니다."""

    def __init__(self):
        self.at = 0
        self.regions = []

    def alloc(self, name, words, note=''):
        base = self.at
        self.at += words
        self.regions.append(dict(name=name, base=base, words=words, note=note))
        return base

    def report(self):
        return self.regions, self.at


def load_consts():
    """`pack_evt.py` 가 만든 config.json 에서 레이어별 shift / PB 베이스를 읽습니다.

    이걸 손으로 적으면 22개 레이어를 다 틀립니다. **한 곳(manifest)에서만** 옵니다.
    """
    cfg = json.load(open(os.path.join(DATA, 'config.json')))
    return cfg


def build(n_tok):
    """토큰 수 `n_tok` 인 타임스텝 하나의 step 목록 + A_Mem 영역.

    ## 영역 크기는 **최악치(TOK_MAX)로 고정**합니다

    `n_tok` 은 타임스텝마다 다릅니다(실측 16~123). 영역 크기를 `n_tok` 에 맞추면
    베이스 주소가 매번 바뀌어 **step 프로그램을 타임스텝마다 다시 만들어야** 합니다
    (5,182벌). 최악치로 고정하면 프로그램이 **정적**이 되고, `n_tok` 에 따라 바뀌는
    것은 `M / K / NOUT / C` 네 필드뿐입니다 — 엔진이 `n_tok` 레지스터 하나로
    발행 시점에 채워 넣습니다.

    낭비는 A_Mem 507 KB 중 일부이고 상한 1 MB 안이라 문제가 없습니다.
    """
    Lk = n_tok + 1                       # +1 = bias_k 토큰 (절대 마스킹 안 됨)
    TT, QT, KT = tiles(TOK_MAX), tiles(LAT), tiles(TOK_MAX + 1)
    KSTR, VSTR = KT * HD, TOK_MAX + 1          # head 간 간격

    a = Arena()
    # ---- 토큰 경로 ----
    R = {}
    R['X']    = a.alloc('X',    TT * 144, '입력 토큰 int8 (호스트)')
    # pos_idx : 워드 mt 의 레인 i = 토큰 mt*32+i 의 표 인덱스 (int16).
    # 타임스텝당 최대 4워드(246 B) 라 따로 포트를 두지 않고 A_Mem 을 씁니다.
    R['PIDX'] = a.alloc('PIDX', TT,        'pos_idx (호스트) — Pos_Gather 입력')
    R['PIN']  = a.alloc('PIN',  TT * 160, 'preproc 입력 = [projection 96 | pos enc 64]')
    R['PRE']  = a.alloc('PRE',  TT * E,   'preproc 출력 int8')
    R['EV1']  = a.alloc('EV1',  TT * E,   'proc_events.1 출력 int8(ReLU)')
    R['EV']   = a.alloc('EV',   TT * E,   'proc_events 최종 (+x) int8  → K/V 원본')
    # ---- latent 경로 ----
    R['LATV'] = a.alloc('LATV', QT * E,   'latent_vectors 누적 bf16')
    R['Z']    = a.alloc('Z',    QT * E,   '현재 z (블록 간 이어짐) bf16')
    R['ZATT'] = a.alloc('ZATT', QT * E,   'z_att = attn 출력 + z_input bf16')
    R['LNX']  = a.alloc('LNX',  max(TT, QT) * E, 'layer_norm_x 출력 int8')
    R['LN1']  = a.alloc('LN1',  QT * E,   'layer_norm_1 출력 int8')
    R['LNA']  = a.alloc('LNA',  QT * E,   'layer_norm_att / _2 출력 int8')
    # ---- attention 작업영역 ----
    R['Q']    = a.alloc('Q',    HEADS * QT * HD, 'Q  head-major int8')
    R['K']    = a.alloc('K',    HEADS * KSTR, 'K  head-major int8')
    R['V']    = a.alloc('V',    HEADS * VSTR, 'Vᵀ (Transpose32 출력) int8, stride 고정')
    # `bias_k`/`bias_v` 는 학습된 **추가 키/값 토큰** 입니다. 키 인덱스가
    # `n_tok` 이라 자리가 타임스텝마다 달라 K/V 에 써 넣으려면 읽고-고쳐-쓰기가
    # 필요합니다. 대신 호스트가 여기 한 번 넣어 두고 **엔진이 읽는 쪽에서**
    # 끼워 넣습니다. K/V 영역은 블록 3개가 돌려 쓰므로 예약 칸을 K/V 안에 둘 수
    # 없습니다 — 블록마다 값이 다르기 때문입니다.
    #   레이아웃 : [블록][k/v][head]  레인 = head_dim
    R['BKV']  = a.alloc('BKV',  len(BLOCKS) * 2 * HEADS, 'bias_k/bias_v 토큰 int8')
    R['SM']   = a.alloc('SM',   TOK_MAX + 1,     'softmax 출력 uint8 (행타일 1개분)')
    R['CTX']  = a.alloc('CTX',  QT * E,          'attn·V 결과 int8 (head 이어붙임)')
    R['FFN']  = a.alloc('FFN',  QT * E,          '블록 내 FFN 중간 int8')

    CFG = load_consts()
    SH, PBB, APB = CFG['shifts'], CFG['pb_base'], CFG['attn_pb']
    S = []

    def gemm(name, layer, M, K, Nout, a_base, b_off, cons, sh, out,
             pb=None, act=0, gpb=None, gsh=0, note='', row0=0):
        # shift / PB / W 베이스는 **manifest 에서** 옵니다 (`sh`/`pb` 는 덮어쓰기용).
        # bf16 소비자는 shift 가 없습니다(None) → 0.
        #
        # `b_off` 는 그 레이어 안에서의 **오프셋**입니다 — in_proj 의 K/V 밴드가
        # `4*E`/`8*E` (출력 타일 4개분) 만큼 들어간 자리를 가리킵니다. 베이스는
        # 손으로 적지 않습니다. (처음에 `b_off` 를 그대로 BIN 에 넣어 **모든 GEMM
        # 이 W_Mem 0번지**를 읽었습니다 — event_projection 만 우연히 맞았습니다.)
        if sh is None:
            sh = SH.get(layer) or 0
        if pb is None:
            pb = (PBB.get(layer) or 0) + row0     # in_proj 밴드는 행 오프셋만큼
        S.append(dict(kind=K_GEMM, name=name, layer=layer, M=M, K=K, NOUT=Nout,
                      AIN=a_base, BIN=CFG['w_base'][layer] + b_off,
                      AOUT=out, CONS=cons, SHIFT=sh,
                      PB=pb, ACT=act, GPB=gpb, GSH=gsh, note=note))

    def ln(name, mod, M, a_in, a_out, sh, note=''):
        # LayerNorm 은 **두 메모리**를 읽습니다 — PG(gamma/beta, 특징별)와
        # PB(Q4.11→int8 재양자화 스칼라 1개). 둘의 베이스가 달라서 필드도 둘입니다:
        #   PB   = 재양자화 스칼라 자리      OSTR = PG 베이스
        # (처음엔 PB 하나로 둘 다 가리키게 해 두었는데, 서로 다른 메모리라
        #  같은 값일 수가 없습니다.)
        # GSH 는 LN 에서 안 쓰던 칸이라 **코어의 고정소수점 창**(in_shift)을
        # 실어 보냅니다. 6비트 signed 로 인코딩합니다.
        S.append(dict(kind=K_LN, name=name, layer=mod, M=M, K=E, NOUT=E,
                      AIN=a_in, AOUT=a_out,
                      SHIFT=sh if sh else CFG['ln_shift'][mod],
                      GSH=CFG['ln_xsh'][mod] & 0x3F,
                      PB=CFG['ln_pb'][mod], OSTR=CFG['pg_base'][mod], note=note))

    # =========================================================================
    # ① 토큰 전처리 (타임스텝마다)
    # =========================================================================
    # pos enc 는 **PL 의 표에서** 모읍니다 (`Pos_Gather`). 전에는 호스트가 펴서
    # 보냈고 그 이미지가 96.7 MB 였습니다 — 표 27.6 KB + pos_idx 0.47 MB 를
    # 200배로 펼친 셈이라 온칩으로 옮겼습니다.
    S.append(dict(kind=K_POS, name='pos_gather', M=n_tok, K=64, NOUT=64,
                  AIN=R['PIDX'], AOUT=R['PIN'], OSTR=160,
                  note='PIN 뒤쪽 64워드 (앞 96 은 event_projection)'))

    gemm('event_projection', 'backbone.event_projection.seq_init.0',
         n_tok, 144, 96, R['X'], 0, C_Q411, None, R['PIN'],
         note='144→96, Q4.11→GELU→int8. PIN 앞쪽 96워드에 씀 (stride 160)')
    S[-1]['OSTR'] = 160
    S[-1]['VAR'] = V_M
    # `preproc` 의 reduce 는 160 = projection 출력 96 + pos enc 64 입니다. 코어가
    # `a_base + mt*K + k` 로 읽으므로 **타일마다 160 워드가 연속**이어야 합니다.
    # 그래서 별도 PRJ 영역을 두지 않고 event_projection 이 여기 앞쪽(0..95)에
    # 직접 쓰고, 호스트가 뒤쪽(96..159)에 pos enc 를 적재합니다.
    # → event_projection 의 **출력 stride 는 NOUT(96) 이 아니라 160** 입니다.
    gemm('preproc', 'backbone.preproc_block_events.seq_init.0',
         n_tok, 160, E, R['PIN'], 0, C_Q411, None, R['PRE'],
         note='160→128, Q4.11→GELU→int8')
    gemm('proc_ev.1', 'backbone.proc_event_blocks.0.seq_init.1',
         n_tok, E, E, R['PRE'], 0, C_INT8, None, R['EV1'], act=1,
         note='ReLU')
    gemm('proc_ev.4', 'backbone.proc_event_blocks.0.seq_init.4',
         n_tok, E, E, R['EV1'], 0, C_INT8, None, R['EV'], act=1,
         note='ReLU, 이후 +x (add_x_input)')
    S.append(dict(kind=K_RES, name='proc_ev.res', M=n_tok, K=E,
                  AIN=R['PRE'], AOUT=R['EV'], note='x_input 더하기'))

    # =========================================================================
    # ② attention 블록 3개
    # =========================================================================
    for bi, blk in enumerate(BLOCKS):
        short = blk.split('.')[-1]
        # cross 는 K/V 가 토큰, latent 는 K/V 도 latent
        kv_src, kv_rows, kv_T = ((R['EV'], n_tok, TT) if bi == 0
                                 else (R['Z'], LAT, QT))
        kv_lk = (n_tok + 1) if bi == 0 else (LAT + 1)
        kvKT = tiles(kv_lk)

        ln(f'{short}.ln_x', f'{blk}.layer_norm_x', kv_rows, kv_src, R['LNX'], 0)
        ln(f'{short}.ln_1', f'{blk}.layer_norm_1', LAT, R['Z'], R['LN1'], 0)

        # in_proj 은 **GEMM 3개** 입니다.
        #   Q  rows   0-127  A = layer_norm_1   → head-major
        #   K  rows 128-255  A = layer_norm_x   → head-major
        #   V  rows 256-383  A = layer_norm_x   → **Transpose32 경유** → Vᵀ
        #
        # K 와 V 는 A 가 같지만 **목적지 라우팅이 다릅니다**. 하나의 GEMM(Nout=256)
        # 으로 묶어도 (mt,nt) 타일 수가 4+4 = 8 로 같아 **비용이 동일**하므로,
        # step 하나에 출력 하나라는 단순한 모델을 택합니다.
        gemm(f'{short}.in_proj.Q', f'{blk}.attention', LAT, E, E,
             R['LN1'], 0, C_INT8, None, R['Q'],
             note='rows 0-127 → Q (head-major)')
        S[-1]['OSTR'] = QT * HD        # head 간 간격 = latent 타일수 x 32
        gemm(f'{short}.in_proj.K', f'{blk}.attention', kv_rows, E, E,
             R['LNX'], 4 * E, C_INT8, None, R['K'], row0=E,
             note='rows 128-255 → K (head-major). 마지막 1칸은 bias_k ROM')
        S[-1]['OSTR'] = KSTR           # head 간 간격 (bias_k 칸 포함, 최악치 고정)
        gemm(f'{short}.in_proj.V', f'{blk}.attention', kv_rows, E, E,
             R['LNX'], 8 * E, C_INT8, None, R['V'], row0=2 * E,
             note='rows 256-383 → Transpose32 → Vᵀ. 마지막 1칸은 bias_v ROM')
        S[-1]['OSTR'] = VSTR           # Vᵀ 는 head 마다 최악치 + bias_v 1칸

        # head x 행타일 : QK → softmax → AV
        for h in range(HEADS):
            for mt in range(QT):
                S.append(dict(kind=K_GEMM, name=f'{short}.qk.h{h}.t{mt}',
                              layer=f'{blk}:QK', M=N, K=HD, NOUT=kv_lk,
                              AIN=R['Q'] + h * QT * HD + mt * HD,
                              BIN=R['K'] + h * KSTR,             # stride 고정
                              # AOUT 은 QK 에서 안 쓰므로 bias_k 워드 주소로 씁니다
                              AOUT=R['BKV'] + (bi * 2 + 0) * HEADS + h,
                              CONS=C_Q69, FLAG2=F2_BKV,
                              # attention 의 재양자화는 **블록당 스칼라 1개**
                              PB=APB[blk]['qk'][0], SHIFT=APB[blk]['qk'][2],
                              # softmax 는 **같은 step** 입니다 — QK 의 컬럼이
                              # 메모리를 안 거치고 바로 들어가고, 나온 uint8 이
                              # OSTR 이 가리키는 SM 영역에 실립니다. step 을 따로
                              # 두면 두 번째 start 가 모아 둔 입력을 지웁니다.
                              OSTR=R['SM'],
                              note='Q6.9 → softmax(C=Lk) → SM. 키 Lk-1 = bias_k'))
                S.append(dict(kind=K_GEMM, name=f'{short}.av.h{h}.t{mt}',
                              layer=f'{blk}:AV', M=N, K=kv_lk, NOUT=HD,
                              AIN=R['SM'], BIN=R['V'] + h * VSTR, FLAG2=F2_BKV,
                              # AV 는 mt=0 하나뿐이라 OSTR 이 남습니다
                              OSTR=R['BKV'] + (bi * 2 + 1) * HEADS + h,
                              AOUT=R['CTX'] + mt * E + h * HD,
                              CONS=C_INT8,
                              PB=APB[blk]['av'][0], SHIFT=APB[blk]['av'][2],
                              note='결과를 채널순으로 이어붙임'))

        gemm(f'{short}.out_proj', f'{blk}.attention.out_proj', LAT, E, E,
             R['CTX'], 0, C_BF16, None, R['ZATT'], note='→ bf16')
        S.append(dict(kind=K_RES, name=f'{short}.res_att', M=LAT, K=E,
                      AIN=R['Z'], AOUT=R['ZATT'], note='z_att = attn + z_input'))

        ln(f'{short}.ln_att', f'{blk}.layer_norm_att', LAT, R['ZATT'], R['LNA'], 0)
        gemm(f'{short}.linear1', f'{blk}.linear1', LAT, E, E,
             R['LNA'], 0, C_Q411, None, R['FFN'], note='GELU')
        ln(f'{short}.ln_2', f'{blk}.layer_norm_2', LAT, R['FFN'], R['LNA'], 0)
        gemm(f'{short}.linear2', f'{blk}.linear2', LAT, E, E,
             R['LNA'], 0, C_Q411, None, R['FFN'], note='GELU')
        gemm(f'{short}.linear3', f'{blk}.linear3', LAT, E, E,
             R['FFN'], 0, C_BF16, None, R['Z'], note='→ bf16')
        S.append(dict(kind=K_RES, name=f'{short}.res_ffn', M=LAT, K=E,
                      AIN=R['ZATT'], AOUT=R['Z'], note='+ z_att'))

    # 타임스텝 끝 : latent_vectors += z
    S.append(dict(kind=K_RES, name='latent.acc', M=LAT, K=E,
                  AIN=R['Z'], AOUT=R['LATV'], note='타임스텝 누적'))

    # =========================================================================
    # ③ 마지막 한 번 (T 루프 밖)
    # =========================================================================
    tail = []
    _eln = 'backbone.proc_embs_block.layer_norm'
    tail.append(dict(kind=K_LN, name='embs.ln', layer=_eln,
                     M=LAT, K=E, NOUT=E, AIN=R['LATV'], AOUT=R['LNA'],
                     SHIFT=CFG['ln_shift'][_eln], PB=CFG['ln_pb'][_eln],
                     GSH=CFG['ln_xsh'][_eln] & 0x3F,
                     OSTR=CFG['pg_base'][_eln]))
    tail.append(dict(kind=K_GEMM, name='embs.linear1',
                     layer='backbone.proc_embs_block.linear1', M=LAT, K=E, NOUT=E,
                     AIN=R['LNA'], BIN=CFG['w_base']['backbone.proc_embs_block.linear1'],
                     AOUT=R['FFN'], CONS=C_INT8, SHIFT=None,
                     ACT=1, note='ReLU'))
    # 평균의 나누기(1/96)와 다음 GEMM 격자로의 재양자화는 **곱수 하나**에
    # 접혀 레이어 뒤 빈 칸(`PB + NOUT`)에 있습니다 — GELU 스칼라와 같은 자리.
    _mln = CFG['mean_layer']
    tail.append(dict(kind=K_MEAN, name='embs.gap', layer=_mln, M=LAT, K=E, NOUT=E,
                     AIN=R['FFN'], AOUT=R['LNA'], PB=CFG['pb_base'][_mln],
                     GSH=CFG['gelu_shift'][_mln],
                     note='latent 96개 평균 → 1행 (레인 0)'))
    tail.append(dict(kind=K_GEMM, name='clf.linear_1', layer='models_clf.0.linear_1',
                     M=1, K=E, NOUT=E, AIN=R['LNA'],
                     BIN=CFG['w_base']['models_clf.0.linear_1'], AOUT=R['FFN'],
                     CONS=C_INT8, SHIFT=None, ACT=1, note='ReLU'))
    tail.append(dict(kind=K_ARGMAX, name='clf.linear_2', layer='models_clf.0.linear_2',
                     M=1, K=E, NOUT=NCLS, AIN=R['FFN'],
                     BIN=CFG['w_base']['models_clf.0.linear_2'],
                     PB=PBB['models_clf.0.linear_2'],
                     SHIFT=SH['models_clf.0.linear_2'],
                     note='argmax (log_softmax 없음)'))

    return S, tail, a, R, dict(Lk=Lk, TT=TT, QT=QT, KT=KT, n_tok=n_tok)


def selfcheck(S, tail, arena, R, dims, aw_a):
    """① 영역 겹침 ② 상한 ③ 생산 전 소비"""
    regs, total = arena.report()
    errs = []
    # ① 겹침
    srt = sorted(regs, key=lambda r: r['base'])
    for x, y in zip(srt, srt[1:]):
        if x['base'] + x['words'] > y['base']:
            errs.append(f"영역 겹침 {x['name']}(끝 {x['base']+x['words']}) "
                        f"~ {y['name']}(시작 {y['base']})")
    # ② 상한
    if total > (1 << aw_a):
        errs.append(f'A_Mem 초과 {total} > {1<<aw_a}')
    # ③ 생산 전 소비 — 영역 단위로 봅니다 (타임스텝 재귀는 예외 처리)
    # 호스트가 채우거나 초기값인 영역 (PIN 뒤쪽 64워드가 pos enc)
    # 호스트가 채우는 영역 : 토큰 · pos enc · latent 초기값 · bias 토큰
    # PIN 은 이제 **전부 엔진이** 씁니다 (앞 96 event_projection, 뒤 64 Pos_Gather)
    produced = {R['X'], R['PIDX'], R['LATV'], R['Z'], R['BKV']}
    for st in S + tail:
        for key in ('AIN', 'BIN'):
            v = st.get(key)
            if v is None or key == 'BIN' and st.get('kind') == K_GEMM and v == 0:
                continue
            reg = max((r for r in regs if r['base'] <= v < r['base'] + r['words']),
                      key=lambda r: r['base'], default=None)
            if reg and reg['base'] not in produced:
                errs.append(f"{st['name']}: {key}={v} ({reg['name']}) 이 아직 생산 전")
        # 출력 자리는 AOUT 이 기본이지만, **Q6.9 step 은 softmax 출력이 OSTR**
        # 이 가리키는 SM 영역입니다 (AOUT 은 bias_k 워드 주소로 씁니다).
        outs = [st.get('AOUT')]
        if st.get('CONS') == C_Q69:
            outs = [st.get('OSTR')]
        for ov in outs:
            if ov is None:
                continue
            reg = max((r for r in regs if r['base'] <= ov < r['base'] + r['words']),
                      key=lambda r: r['base'], default=None)
            if reg:
                produced.add(reg['base'])
    return errs, total


# =============================================================================
# step 워드 인코딩 — `rtl/EvT_Engine.v` 의 디코더와 **한 벌**이어야 합니다
#
#   [ 31: 0] KIND[3:0] CONS[5:4] ACT[7:6] VAR[11:8] FLAG[15:12]
#            SHIFT[21:16] GSH[27:22]
#   [ 63:32] M   [ 95:64] K   [127:96] NOUT
#   [159:128] AIN   [191:160] BIN   [223:192] AOUT
#            FLAG2[31:28]
#   [255:224] PB[15:0] | OSTR[31:16]
#
#   VAR   [0] M←n_tok  [1] NOUT←Lk  [2] K←Lk  [3] C←Lk
#   FLAG  [0] Transpose32 경유  [1] head-major 주소(OSTR=stride)  [2] B는 A_Mem
#         [3] raw16 — Q4.11 을 int8 로 재양자화하지 않고 **16비트 그대로** 저장
#   FLAG2 [0] LayerNorm 입력이 **정수 코드** (int→bf16 변환 후 투입)
#         [1] RES 두 피연산자가 정수 코드 — PB/OSTR 의 bf16 상수로 스케일
#         [2] B 피연산자에 bias_k/bias_v 토큰이 붙음 (키 인덱스 n_tok)
#         [3] 활성함수 뒤 2차 재양자화 (곱수는 GELU 스칼라와 같은 칸, GSH)
# =============================================================================
V_M, V_NOUT, V_K, V_C = 1, 2, 4, 8
F_TR, F_HM, F_BA, F_RAW16 = 1, 2, 4, 8
# FLAG2 (워드0 [31:28]) — 나중에 추가된 두 가지
F2_LNINT, F2_RESQ, F2_BKV, F2_REQ2 = 1, 2, 4, 8


def set_ostr(S):
    """OSTR(출력 stride) 기본값 = NOUT. 다른 step 이 명시적으로 덮어씁니다.

    head-major step 은 head 간 간격을 넣습니다.
    """
    for st in S:
        if st['kind'] != K_GEMM:
            continue
        if 'OSTR' in st:
            continue
        st['OSTR'] = st.get('NOUT') or 0


def annotate(S):
    """이름 규칙으로 VAR/FLAG 를 붙입니다.

    손으로 step 마다 적으면 반드시 어긋나므로, **어떤 step 이 토큰 수에 의존하는가**
    라는 규칙 하나로 일괄 결정합니다.
    """
    for st in S:
        nm = st['name']
        var = flag = 0
        # 토큰 행을 도는 step 은 M 이 n_tok
        if nm in ('event_projection', 'preproc', 'proc_ev.1', 'proc_ev.4',
                  'proc_ev.res', 'pos_gather') \
           or nm.endswith(('ln_x',)) and 'cross' in nm:
            var |= V_M
        if '.in_proj.K' in nm or '.in_proj.V' in nm:
            if nm.startswith('cross'):
                var |= V_M
        # cross 의 attention 만 Lk 가 토큰에 의존 (latent 는 96+1 상수)
        if nm.startswith('cross'):
            if '.qk.' in nm:  var |= V_NOUT
            if '.sm.' in nm:  var |= V_C
            if '.av.' in nm:  var |= V_K
        # 라우팅
        if '.in_proj.V' in nm:  flag |= F_TR | F_HM
        if '.in_proj.Q' in nm or '.in_proj.K' in nm: flag |= F_HM
        if '.qk.' in nm or '.av.' in nm: flag |= F_BA
        st['VAR'], st['FLAG'] = var, flag


def annotate2(S, CFG):
    """GELU 뒤 재양자화 shift 와, 나중에 붙은 FLAG2 두 가지를 채웁니다.

    `pack_evt.py` 가 레이어 채널 뒤에 **한 칸**을 비워 GELU 곱수를 넣어 두었고
    (`PB + NOUT`), shift 는 여기서 붙입니다. `linear1` 은 뒤가 LayerNorm 이라
    int8 격자가 없어 **Q4.11 16비트를 그대로** 씁니다(raw16).
    """
    GSH, RAW16, REQ2 = CFG['gelu_shift'], set(CFG['gelu_raw16']), set(CFG['req2'])
    SH2, PBB2 = CFG['shifts'], CFG['pb_base']

    # 본체 GEMM 은 `gemm()` 헬퍼가 manifest 에서 상수를 채우지만 **꼬리는 dict 를
    # 직접 만들어** 그 경로를 안 탑니다. 그래서 `SHIFT=None, PB=None` 이 그대로
    # 남아 인코더가 0 으로 떨어뜨렸고, `embs.linear1` / `clf.linear_1` 이
    # **event_projection 채널 0 의 곱수에 시프트 0** 으로 돌았습니다.
    # 여기서 한 번 더 훑어 빠진 것을 채웁니다 (본체는 이미 차 있어 무해).
    for st in S:
        lay = st.get('layer')
        if st['kind'] in (K_GEMM, K_ARGMAX) and lay in PBB2:
            if st.get('PB') is None:
                st['PB'] = PBB2[lay]
            if st.get('SHIFT') is None:
                st['SHIFT'] = SH2.get(lay) or 0
    ISTEP = CFG['input_steps']
    for st in S:
        if st['kind'] == K_GEMM and st.get('CONS') == C_Q411:
            lay = st['layer']
            if lay in RAW16:
                st['FLAG'] = (st.get('FLAG') or 0) | F_RAW16
                st['GSH'] = 0
            else:
                st['GSH'] = GSH[lay]
        if st['kind'] == K_GEMM and st.get('layer') in REQ2:
            st['FLAG2'] = (st.get('FLAG2') or 0) | F2_REQ2
            st['GSH'] = GSH[st['layer']]
        # `layer_norm_2` 만 입력이 정수 코드(Q4.11) 입니다 — 나머지는 bf16
        if st['kind'] == K_LN and st['layer'].endswith('layer_norm_2'):
            st['FLAG2'] = (st.get('FLAG2') or 0) | F2_LNINT
        # proc_events 의 잔차만 두 피연산자가 int8 코드 (스케일이 서로 다름)
        if st['name'] == 'proc_ev.res':
            st['FLAG2'] = (st.get('FLAG2') or 0) | F2_RESQ
            # a = AIN(PRE) = preproc 출력 int8, b = AOUT(EV) = proc_ev.4 출력 int8
            # step 을 `round(step * 2^RSH)` 정수로 (RTL 의 RSH 와 한 벌).
            # bf16 상수로 두면 비율이 0.4 % 흔들려 결과가 절반 어긋납니다.
            st['PB']   = fx_step(ISTEP['backbone.proc_event_blocks.0.seq_init.1'])
            st['OSTR'] = fx_step(SEQ5_STEP)


def enc(st):
    """step dict → 256비트 정수"""
    kind = st['kind']
    cons = st.get('CONS') or 0
    act  = st.get('ACT') or 0
    var  = st.get('VAR') or 0
    flag = st.get('FLAG') or 0
    sh   = st.get('SHIFT') or 0
    gsh  = st.get('GSH') or 0
    fl2  = st.get('FLAG2') or 0
    w0 = ((kind & 0xF) | ((cons & 3) << 4) | ((act & 3) << 6)
          | ((var & 0xF) << 8) | ((flag & 0xF) << 12)
          | ((sh & 0x3F) << 16) | ((gsh & 0x3F) << 22)
          | ((fl2 & 0xF) << 28))
    ws = [w0, st.get('M') or 0, st.get('K') or 0, st.get('NOUT') or 0,
          st.get('AIN') or 0, st.get('BIN') or 0, st.get('AOUT') or 0,
          ((st.get('PB') or 0) & 0xFFFF) | (((st.get('OSTR') or 0) & 0xFFFF) << 16)]
    v = 0
    for i, w in enumerate(ws):
        assert 0 <= w < (1 << 32), (st['name'], i, w)
        v |= (w & 0xFFFFFFFF) << (32 * i)
    return v


def emit_steps(dst, S, tail):
    words = [enc(x) for x in (S + tail)]
    with open(os.path.join(dst, 'stepmem.bin'), 'wb') as fb, \
         open(os.path.join(dst, 'stepmem.hex'), 'w') as fh:
        for v in words:
            raw = v.to_bytes(32, 'little')
            fb.write(raw)
            fh.write(raw[::-1].hex() + '\n')
    return len(words)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tok', type=int, default=123, help='토큰 수 (최악 123)')
    ap.add_argument('--aw_a', type=int, default=14, help='A_Mem 주소폭')
    ap.add_argument('--dst', default=DATA)
    args = ap.parse_args()

    S, tail, arena, R, dims = build(args.tok)
    # **본체와 꼬리에 똑같이** 걸어야 합니다. 꼬리를 빠뜨리면 `OSTR` 이 0 으로
    # 나가고, 쓰기 주소가 `AOUT + mt*OSTR + n` 이라 **행타일 3개가 같은 워드에
    # 겹쳐 써집니다** (뒤 64행은 직전 블록의 잔재가 남습니다).
    for lst in (S, tail):
        annotate(lst)
        set_ostr(lst)
        annotate2(lst, load_consts())
    errs, total = selfcheck(S, tail, arena, R, dims, args.aw_a)

    print(f"[schedule_evt] n_tok={args.tok}  Lk={dims['Lk']}  "
          f"토큰타일 {dims['TT']}  latent타일 {dims['QT']}")
    print(f"\n{'영역':<6}{'base':>8}{'words':>8}{'KB':>8}  설명")
    for r in arena.report()[0]:
        print(f"  {r['name']:<6}{r['base']:>8}{r['words']:>8}"
              f"{r['words']*64/1024:>8.1f}  {r['note']}")
    print(f"  {'합계':<6}{'':>8}{total:>8}{total*64/1024:>8.1f}  "
          f"(A_Mem 512b x {1<<args.aw_a} = {(1<<args.aw_a)*64/1024:.0f} KB)")

    print(f'\n타임스텝당 step {len(S)}개, 마지막 한 번 {len(tail)}개')
    from collections import Counter
    for k, c in Counter(s['kind'] for s in S).most_common():
        print(f"  {KIND[k]:<6}{c:>4}")

    print(f'\n앞 12 step:')
    print(f"  {'#':>3} {'kind':<6}{'name':<22}{'M':>5}{'K':>5}{'NOUT':>6}"
          f"{'AIN':>7}{'BIN':>7}{'AOUT':>7}  소비자")
    for i, s in enumerate(S[:12]):
        print(f"  {i:>3} {KIND[s['kind']]:<6}{s['name']:<22}"
              f"{s.get('M',''):>5}{s.get('K',''):>5}{s.get('NOUT',''):>6}"
              f"{str(s.get('AIN','')):>7}{str(s.get('BIN','')):>7}"
              f"{str(s.get('AOUT','')):>7}  {CONS.get(s.get('CONS'),'-')}")

    print()
    if errs:
        print('자체 검증 실패:')
        for e in errs[:10]:
            print('  ✗', e)
    else:
        print('자체 검증 : 영역 겹침 없음 · A_Mem 상한 이내 · 생산 전 소비 없음 ✅')

    nstep = emit_steps(args.dst, S, tail)
    print(f'step 이미지 {nstep} 워드 → stepmem.bin (본체 {len(S)}, 끝 {len(tail)})')

    json.dump(dict(n_tok=args.tok, dims=dims, n_body=len(S), n_tail=len(tail),
                   regions=arena.report()[0], a_words=total,
                   steps=S, tail=tail),
              open(os.path.join(args.dst, 'schedule.json'), 'w'), indent=1)
    print(f"\n-> {os.path.join(args.dst, 'schedule.json')}")


if __name__ == '__main__':
    main()
