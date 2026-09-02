"""골든에서 **명령어별 기대 A_Mem 내용**을 뽑아 통합 TB 용 hex 로

    python3 golden_insts.py --sample 0 --tstep 0

## 왜 MAC 피연산자를 쓰는가

A_Mem 의 어떤 영역이 맞는지 확인하려면 "그 영역이 담아야 할 int8 코드" 가 필요합니다.
모듈 훅은 **실수값**을 주므로 scale 로 나눠 코드로 만들어야 하는데, 그 매핑 표
자체가 틀리기 쉽습니다.

대신 **`MAC_PROBE` 가 모든 GEMM 의 int8 피연산자 `a` 를 그대로 줍니다.** 어떤 GEMM 의
`a` 는 곧 그 GEMM 이 읽는 A_Mem 영역의 내용입니다. 나눗셈도 반올림도 없습니다.

    A_Mem[PIN] == linear_K160 의 a        (preproc 입력 = [projection 96 | pos 64])
    A_Mem[PRE] == proc_ev.1 의 a
    A_Mem[EV1] == proc_ev.4 의 a
    A_Mem[EV]  == cross_attention 의 K/V proj 가 읽는 a (layer_norm_x 출력)

즉 **어떤 명령어의 출력은 다음 명령어의 입력으로 검증**됩니다.

## 이름으로 짝지어야 합니다 (README §5 ③)

골든은 `event_projection`/`preproc` 을 `(T,…)` 텐서 전체에 **한 번** 적용하고 나서
타임스텝 루프를 돕니다. 하드웨어는 타임스텝마다 돌립니다. 같은 선형변환이라 결과는
같지만 **호출 순서가 다릅니다.** 그래서 순서가 아니라 reduce 폭(K)으로 식별합니다:

    linear_K144  event_projection   (T 전체, 1회)
    linear_K160  preproc            (T 전체, 1회)
    linear_K128  나머지 전부         (타임스텝마다)

앞의 둘은 유일해서 바로 잡히고, `linear_K128` 은 타임스텝 안에서의 순번으로 셉니다.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))
# sjm: 원래는 sgh 저장소를 가리켰습니다 (아래 두 줄).
#   RDS   = <sgh>/EventTransformer/real_dvs_script
#   QUANT = <sgh>/EventTransformer/quantization
# real_dvs_script 를 EvT_quant 로 복사해 두고 그쪽을 봅니다.
# EVT_ROOT 로 덮어쓸 수 있습니다.
EVT_ROOT = os.environ.get('EVT_ROOT', '/hai/home/sjm/EvT_quant')
RDS = os.path.join(EVT_ROOT, 'real_dvs_script')
QUANT = os.path.join(EVT_ROOT, 'quantization')
for p in (RDS, QUANT, os.path.dirname(QUANT)):
    if p not in sys.path:
        sys.path.insert(0, p)

N = 32


def lane_words(rows):
    """(n_row, nfeat) → 워드 리스트 (워드 k 의 레인 i = rows[mt*32+i][k])"""
    n_row, nfeat = rows.shape
    TT = (n_row + N - 1) // N
    out = np.zeros((TT * nfeat, N), dtype=np.int64)
    for mt in range(TT):
        lo, hi = mt * N, min((mt + 1) * N, n_row)
        out[mt * nfeat:(mt + 1) * nfeat, :hi - lo] = rows[lo:hi].T
    return out


def w_hex(path, words, bits=16):
    with open(path, 'w') as f:
        for w in words:
            f.write(''.join(f'{int(v) & ((1 << bits) - 1):0{bits//4}x}'
                            for v in w[::-1]) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--tstep', type=int, default=0, help='몇 번째 타임스텝')
    ap.add_argument('--single', action='store_true',
                    help='그 타임스텝 **하나만** 넣어 돌립니다 (통합 TB 와 동일)')
    ap.add_argument('--dst', default=os.path.join(DATA, 'golden'))
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    from golden import Preprocessed, load_int_model            # noqa: E402
    from quant_lib import hw_quant as HQ                       # noqa: E402

    pre = Preprocessed(os.path.join(RDS, 'data'))
    model, payload, _ = load_int_model('cpu')

    # ---- 하드웨어와 **같은 비선형/포맷**으로 맞춥니다 ----
    # 이걸 안 하면 골든은 정확한 GELU, RTL 은 PWL 이라 int8 출력이 곳곳에서
    # 1 LSB 씩 어긋나고, 그 차이가 뒤 레이어에서 증폭돼 **진짜 버그와 구분이
    # 안 됩니다.** 맞춰 두면 남은 불일치는 전부 RTL 버그입니다.
    #   · GELU  : 64세그먼트 PWL (`rtl/gelu/Gelu_Pwl.v` 와 비트 동일)
    #   · 잔차  : `x_input` 을 preproc 출력의 int8 격자로 (A_Mem 이 int8 이라)
    #   · gelu1 뒤는 int8 화 **안 함** — 하드웨어가 Q4.11 16b 를 그대로 저장
    from hw_format import build as hw_build                # noqa: E402
    hw_build(model, {}, use_pwl=True, gelu1_int8=False)

    pol, pix, label = pre.sample(args.sample)
    T = pol.shape[0]
    assert args.tstep < T, (args.tstep, T)
    t_sel = args.tstep

    # ---- proc_events 의 중간/최종값 (RES 와 LayerNorm 을 따로 짚기 위해) ----
    # MAC 피연산자만으로는 명령어 3~5 가 한 덩어리라 어디서 틀렸는지 안 보입니다.
    # seq_init 출력(ReLU 뒤 실수)과 블록 출력(잔차 뒤 실수)을 받아 각각
    # int8 코드 / bf16 비트로 만들어 둡니다.
    SEQ5_STEP = 0.019990921020507812
    # 이 블록은 **타임스텝마다** 불립니다 — 호출을 순서대로 모아 두고 뒤에서
    # `tstep` 번째를 고릅니다 (마지막 것만 남기면 t=0 을 못 봅니다).
    box = {'seq': [], 'ev': []}
    blk0 = model.get_submodule('backbone.proc_event_blocks.0')
    blk0.seq_init.register_forward_hook(
        lambda m, i, o: box['seq'].append(o.detach()))
    blk0.register_forward_hook(lambda m, i, o: box['ev'].append(o.detach()))
    # `proc_embs_block` 의 **입력**이 곧 누적된 latent_vectors 입니다 (A_Mem 의
    # LATV). 여기가 맞는지 보면 3블록 체인과 꼬리 단 중 어디가 틀렸는지 갈립니다.
    model.get_submodule('backbone.proc_embs_block').register_forward_pre_hook(
        lambda m, i: box.__setitem__('latv', i[0].detach()))
    # cross 블록의 잔차 스트림 — `layer_norm_att` 의 **입력**이 곧 z_att 입니다
    # (out_proj 출력 + z_input). A_Mem 의 ZATT 영역과 대응합니다.
    # `proc_embs_block` 은 `F.relu` 를 인라인으로 부르므로 hw_quant 가 forward 를
    # 통째로 갈아끼웠습니다: `z = F.relu(self._sites[0](linear1(z)))` → `.mean(0)`.
    # 그 사이트를 감싸면 **MEAN 의 입력**(= A_Mem 의 FFN)을 그대로 잡습니다.
    _comp = model.get_submodule('backbone.proc_embs_block')
    _s0 = _comp._sites[0]

    def _tap_s0(z):
        y = _s0(z)
        box['embs_site'] = y.detach()
        return y
    _comp._sites = (_tap_s0,)

    CROSS = 'backbone.proc_memory_blocks.0.cross_attention'
    model.get_submodule(CROSS + '.layer_norm_att').register_forward_pre_hook(
        lambda m, i: box.__setitem__('zatt', i[0].detach()))

    macs = []

    def probe(tag, a_int, b_int, acc, bias_int=None):
        macs.append(dict(tag=tag, a=a_int.detach().cpu().numpy(),
                         b=b_int.detach().cpu().numpy(),
                         acc=acc.detach().cpu().numpy()))
    HQ.MAC_PROBE = probe
    if args.single:
        # 통합 TB 는 타임스텝 **하나**를 돌립니다 (`n_time=1`). 골든도 같은
        # 조건으로 돌려야 latent 누적과 분류기 출력까지 그대로 대조됩니다.
        pol, pix = pol[args.tstep:args.tstep + 1], pix[args.tstep:args.tstep + 1]
        T = 1
    with torch.no_grad():
        _e, logits = model(pol, pix)
    HQ.MAC_PROBE = None
    pred = int(logits.float().argmax(-1)[0])

    tags = [m['tag'] for m in macs]
    print(f'[golden_insts] 샘플 {args.sample}  T={T}  label={label} pred={pred}')
    print(f'  MAC {len(macs)}개')

    # ---- K144 / K160 은 유일 : T 전체에 한 번 ----
    i144 = tags.index('linear_K144')
    i160 = tags.index('linear_K160')
    a160 = macs[i160]['a']                    # preproc 입력 = [proj 96 | pos 64]
    print(f'  linear_K144 #{i144}  a{macs[i144]["a"].shape}')
    print(f'  linear_K160 #{i160}  a{a160.shape}   ← PIN 의 기대값')

    # (T, Npad, B, 160) 또는 (T, Npad, 1, 160) 형태에서 타임스텝을 고릅니다
    a160 = np.squeeze(a160)
    if args.single:
        a160 = a160[None] if a160.ndim == 2 else a160
    assert a160.ndim == 3, a160.shape
    pin_t = a160[0 if args.single else args.tstep]        # (Npad, 160)

    # 유효 토큰만 (앞쪽 0 패딩이므로 뒤에서 n_tok 개)
    off = int(pre.samples[args.sample][0])
    n_tok = int(pre.index[off + args.tstep][1])
    pin = pin_t[-n_tok:].astype(np.int64)
    print(f'  타임스텝 {args.tstep} : 유효 토큰 {n_tok}  PIN {pin.shape}')

    # ---- 타임스텝 안의 linear_K128 순번 ----
    # 순서: proc_ev.1, proc_ev.4, in_proj.Q, in_proj.K, in_proj.V, ...
    k128 = [i for i, t in enumerate(tags) if t == 'linear_K128']
    per_t = len(k128) // T
    base = k128[(0 if args.single else args.tstep) * per_t]
    names = ['proc_ev.1', 'proc_ev.4', 'in_proj.Q', 'in_proj.K', 'in_proj.V']
    print(f'  linear_K128 {len(k128)}개 = 타임스텝당 {per_t}개')

    def bf16_of(x):
        # 음의 0(0x8000) 은 +0 으로 맞춥니다 — 값은 같은데 비트가 달라
        # 통합 TB 가 불일치로 셉니다. 하드웨어는 정수합을 bf16 으로 내리므로
        # 0 이 나오면 항상 +0 입니다.
        b = torch.tensor(x, dtype=torch.float32).to(torch.bfloat16)
        v = b.view(torch.int16).numpy().astype(np.int64) & 0xFFFF
        return np.where(v == 0x8000, 0, v)

    # 호출 t 의 텐서 (n, B, E) → (n_tok, E)
    def pick(lst):
        a = np.squeeze(lst[0 if args.single else args.tstep].numpy())
        assert a.ndim == 2, a.shape
        return a[-n_tok:]

    out = {'PIN': pin}
    print(f'  proc_event_blocks 호출 {len(box["seq"])}회 (= 타임스텝 수)')
    out['EV4'] = np.clip(np.round(pick(box['seq']) / SEQ5_STEP),
                         -128, 127).astype(np.int64)
    out['EV'] = bf16_of(pick(box['ev']))
    for k, nm in enumerate(names):
        a = np.squeeze(macs[base + k]['a']).astype(np.int64)
        if a.ndim == 3:
            a = a[0 if args.single else args.tstep]
        # 토큰 축 명령어는 유효분만, latent 축 명령어는 전부
        if a.shape[0] > 96:
            a = a[-n_tok:]
        out[nm] = a
        print(f'    {nm:<12} a{a.shape}  범위 {a.min()}~{a.max()}')

    # =====================================================================
    # attention : Q/K/V/CTX 는 **A_Mem 영역 모양이 다릅니다** (head-major, Vᵀ)
    # 여기서는 (row, feat) 2차원으로 펴 두고, 워드 변환은 `lane_words` 가 합니다.
    #   Q    (96, 128)   head h 의 열 h*32..h*32+31
    #   K    (n_tok, 128) 같은 방식 (bias_k 토큰은 예약 칸이라 제외)
    #   V    (n_tok, 128) 전치는 TB 가 함 (레인=head_dim)
    #   CTX  (96, 128)   out_proj 의 a
    # =====================================================================
    qk = [i for i, t in enumerate(tags) if t.startswith('attn_QK')]
    av = [i for i, t in enumerate(tags) if t.startswith('attn_AV')]
    print(f'  attention MAC : QK {len(qk)}개, AV {len(av)}개')
    if qk:
        # (B,H,Lq,hd) / (B,H,Lk,hd)
        q_i = np.squeeze(macs[qk[0]]['a']).astype(np.int64)
        k_i = np.squeeze(macs[qk[0]]['b']).astype(np.int64)
        v_i = np.squeeze(macs[av[0]]['b']).astype(np.int64)
        H, Lq, HD = q_i.shape
        Lk = k_i.shape[1]
        out['QI'] = np.concatenate([q_i[h] for h in range(H)], axis=1)
        out['KI'] = np.concatenate([k_i[h][:Lk - 1] for h in range(H)], axis=1)
        out['VI'] = np.concatenate([v_i[h][:Lk - 1] for h in range(H)], axis=1)
        # bias 토큰은 따로 (예약 칸 검증용)
        json.dump(dict(bias_k=[[int(x) for x in k_i[h][Lk - 1]] for h in range(H)],
                       bias_v=[[int(x) for x in v_i[h][Lk - 1]] for h in range(H)]),
                  open(os.path.join(args.dst, 'bias_tok.json'), 'w'), indent=1)
        # out_proj 의 a = attn·V 결과 (head 이어붙임)
        # 타임스텝 안 순서 : proc_ev.1, proc_ev.4, in_proj Q/K/V, **out_proj**, ...
        op = k128[k128.index(base) + 5]
        out['CTX'] = np.squeeze(macs[op]['a']).astype(np.int64)
        print(f"    QI{out['QI'].shape} KI{out['KI'].shape} VI{out['VI'].shape} "
              f"CTX{out['CTX'].shape}  Lk={Lk}")

    # ---- cross 블록의 FFN 체인 (지금까지 한 번도 안 본 구간) ----
    # 타임스텝 안 `linear_K128` 순서 : proc_ev 2개 뒤로 블록마다 7개
    #   [2+7b+0..2] in_proj Q/K/V   [+3] out_proj   [+4..6] linear1/2/3
    # 각 `a` 가 그 GEMM 이 읽는 A_Mem 내용입니다.
    #   B0LNA1 = linear1 의 a = layer_norm_att 출력  (LNA)
    #   B0LNA2 = linear2 의 a = layer_norm_2 출력    (LNA)
    #   B0FFN3 = linear3 의 a = gelu2 뒤 int8        (FFN)
    b0 = k128.index(base) + 2
    for nm2, off in (('B0LNA1', 4), ('B0LNA2', 5), ('B0FFN3', 6)):
        a = np.squeeze(macs[k128[b0 + off]]['a']).astype(np.int64)
        out[nm2] = a
        print(f'    {nm2:<7} a{a.shape}  범위 {a.min()}~{a.max()}')
    if 'zatt' in box:
        za = np.squeeze(box['zatt'].numpy())
        out['B0ZATT'] = bf16_of(za)
        print(f'    B0ZATT  a{za.shape}  범위 {za.min():.4g}~{za.max():.4g}')

    # ---- MEAN 의 입력 = embs.linear1 → ReLU 뒤 int8 (A_Mem 의 FFN) ----
    if 'embs_site' in box:
        mfj = json.load(open(os.path.join(
            QUANT, 'fpga_export',
            os.environ.get('EVT_EXPORT', 'DVS128_10'), 'manifest.json')))
        rscale = next(f['lsb'] for f in mfj['nonlinear_formats']
                     if f['site'] == 'backbone.proc_embs_block.relu.in')
        y = np.squeeze(box['embs_site'].numpy())
        out['TFFN'] = np.clip(np.round(np.maximum(y, 0.0) / rscale),
                              -128, 127).astype(np.int64)
        print(f"    TFFN    a{out['TFFN'].shape}  범위 {out['TFFN'].min()}~"
              f"{out['TFFN'].max()}  (scale {rscale:.6g})  열합[0]="
              f"{int(out['TFFN'][:, 0].sum())}")

    # ---- 누적 latent (LATV) — bf16 ----
    if 'latv' in box:
        lv = np.squeeze(box['latv'].numpy())
        assert lv.ndim == 2, lv.shape
        out['TLATV'] = bf16_of(lv)
        print(f'    TLATV a{lv.shape}  범위 {lv.min():.4g}~{lv.max():.4g}')

    # ---- 꼬리 단 (LN → linear1 → MEAN → clf) 의 MAC 피연산자 ----
    # 이 구간은 타임스텝 루프 밖이라 앞단 검증에 안 걸립니다. 마지막
    # `linear_K128` 세 개가 차례로 embs.linear1 / clf.linear_1 / clf.linear_2 이고
    # 각각의 `a` 가 곧 그 GEMM 이 읽는 A_Mem 내용입니다.
    #   TLNA  (96,128) embs.ln 출력        ← embs.linear1 이 읽음
    #   TMEAN ( 1,128) MEAN 출력(레인 0)   ← clf.linear_1 이 읽음
    #   TCLF1 ( 1,128) clf.linear_1 출력   ← clf.linear_2 가 읽음
    for nm2, ii in (('TLNA', -3), ('TMEAN', -2), ('TCLF1', -1)):
        a = np.squeeze(macs[k128[ii]]['a']).astype(np.int64)
        if a.ndim == 1:
            a = a[None, :]
        out[nm2] = a
        print(f'    {nm2:<6} a{a.shape}  범위 {a.min()}~{a.max()}')

    # ---- 분류기 누산기 10개 (정수) ----
    # 최종 클래스 하나만 대조하면 **채널별 상수를 틀려도 우연히 통과**합니다
    # (실제로 ARGMAX 가 채널 0 의 M 만 쓰는 버그가 그렇게 보드까지 갔습니다).
    # `argmax(acc[c]*M[c])` 의 `acc[c] = Σx·w + b_int[c]` 를 그대로 뽑아
    # 하드웨어의 `res_logits` 와 **비트 단위로** 맞춥니다.
    cls_acc = None
    for m in macs[::-1]:
        a = np.squeeze(m['acc'])
        if a.ndim == 1 and a.shape[0] == 10:
            cls_acc = a.astype(np.int64)
            break
    if cls_acc is None:
        print('  ** 분류기 누산기(10개)를 못 찾았습니다')
    else:
        with open(os.path.join(args.dst, 'clf_acc.hex'), 'w') as f:
            for v in cls_acc:
                f.write(f'{int(v) & 0xFFFFFFFF:08x}\n')
        print(f'  분류기 acc {list(map(int, cls_acc))}')

    json.dump(dict(pred=pred, label=label, n_tok=n_tok,
                   clf_acc=None if cls_acc is None else [int(x) for x in cls_acc],
                   logits=[float(x) for x in logits.float()[0]]),
              open(os.path.join(args.dst, 'expect.json'), 'w'), indent=1)

    # ---- hex 로 ----
    for nm, arr in out.items():
        wr = lane_words(arr)
        w_hex(os.path.join(args.dst, f'{nm}.hex'), wr)
    meta = dict(sample=args.sample, tstep=args.tstep, n_tok=n_tok,
                label=label, pred=pred,
                shapes={k: list(v.shape) for k, v in out.items()},
                words={k: (v.shape[0] + N - 1) // N * v.shape[1]
                       for k, v in out.items()})
    json.dump(meta, open(os.path.join(args.dst, 'meta.json'), 'w'), indent=1)
    print(f'\n-> {args.dst}')
    for k, v in meta['words'].items():
        print(f'    {k:<12} {v:>6} 워드')


if __name__ == '__main__':
    main()
