"""골든 tap 수집 — RTL 테스트벤치가 대조할 포맷 경계 전부

    python3 taps.py --sample 0 --steps 2     # 샘플 0 의 앞 2 타임스텝
    python3 taps.py --sample 0 --steps 0     # 전체 타임스텝

## 왜 tap 이 필요한가

`nonlinear_script` 에서 배운 것: 레이어 출력만 보면 유닛이 많을 때 **어디가 틀렸는지
못 짚습니다.** EvT 는 GEMM 22개 + attention MAC 6개라 훨씬 심합니다. 그래서 포맷이
바뀌는 지점마다 찍습니다.

## 세 가지 경로를 합칩니다

`quant_lib/hw_quant.py` 는 세 종류의 관측점을 이미 갖고 있습니다:

| | 무엇을 주나 | 이름 |
|---|---|---|
| 모듈 forward 훅 | Linear/LayerNorm/GELU 의 입출력 | **이름 기반** |
| `ATTN_TAP` | attention 의 q_in/k_in/v_in, q/k/v, out_proj 입출력 | **이름 기반** |
| `MAC_PROBE` | **모든 정수 MAC** 의 (a_int, b_int, acc) — attention 내부 포함 | 순서 기반 |

`MAC_PROBE` 가 결정적입니다. `attention_forward_hw` 는 `int8_linear` 를 직접 부르므로
모듈 훅으로는 안 보이는데, 여기로는 보입니다. 특히 **softmax 출력(uint8)** 이
`attn_AV` MAC 의 첫 피연산자로 그대로 나옵니다.

`MAC_PROBE` 는 태그가 `linear_K{reduce}` 처럼 일반형이라 이름이 없습니다. 대신 **호출
순서가 결정적**이므로 타임스텝 안에서 순번으로 짝지읍니다.

## 출력

    taps/<사이트이름>.bin      int8/int16/int32 — dtype 은 manifest 규칙을 따름
    taps/manifest.json        사이트별 dtype·shape·순서
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVT_DIR = os.path.dirname(SCRIPT_DIR)
QUANT_DIR = os.path.join(EVT_DIR, 'quantization')
for p in (EVT_DIR, QUANT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from golden import Preprocessed, load_int_model, DATASET     # noqa: E402


def _np(t):
    return t.detach().cpu().numpy()


def _dtype_for(a):
    """정수 텐서를 가장 좁은 안전한 폭으로. 값 범위로 판단합니다."""
    lo, hi = float(a.min()), float(a.max())
    if lo >= -128 and hi <= 127:
        return '<i1'
    if lo >= -32768 and hi <= 32767:
        return '<i2'
    return '<i4'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--dst', default=os.path.join(SCRIPT_DIR, 'taps'))
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--steps', type=int, default=2, help='0 = 전체')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    from quant_lib import hw_quant as HQ                     # noqa: E402

    pre = Preprocessed(args.data)
    model, payload, _ = load_int_model(args.device)

    pol, pix, label = pre.sample(args.sample)
    if args.steps:
        pol, pix = pol[:args.steps], pix[:args.steps]
    T = pol.shape[0]
    print(f'[taps] {DATASET}  샘플 {args.sample}  T={T}  label={label}')

    # ---- ① 모듈 forward 훅 ----
    store, handles = {}, []

    def mk(name):
        def hook(_m, inp, out):
            store.setdefault(name, []).append(
                dict(inp=_np(inp[0]) if torch.is_tensor(inp[0]) else None,
                     out=_np(out) if torch.is_tensor(out) else None))
        return hook

    WANT = ('Linear', 'LayerNorm', 'GELU', 'ReLU')
    for name, m in model.named_modules():
        if type(m).__name__ in WANT:
            handles.append(m.register_forward_hook(mk(name)))

    # ---- ② attention 내부 ----
    HQ.ATTN_TAP = {}

    # ---- ③ 모든 정수 MAC (호출 순서) ----
    macs = []

    def probe(tag, a_int, b_int, acc, bias_int=None):
        macs.append(dict(tag=tag, a=_np(a_int), b=_np(b_int), acc=_np(acc),
                         bias=None if bias_int is None else _np(bias_int)))
    HQ.MAC_PROBE = probe

    with torch.no_grad():
        _embs, logits = model(pol.to(args.device), pix.to(args.device))

    HQ.MAC_PROBE = None
    attn = HQ.ATTN_TAP
    HQ.ATTN_TAP = None
    for h in handles:
        h.remove()

    pred = int(logits.float().argmax(dim=-1)[0])
    print(f'  예측 {pred} / 정답 {label}   MAC {len(macs)}개   '
          f'훅 사이트 {len(store)}개   attention 블록 {len(attn)}개')

    # =========================================================================
    # 저장
    # =========================================================================
    sites = []

    def emit(name, arr, note=''):
        a = np.asarray(arr)
        if a.dtype.kind == 'f':
            # 정수 데이터패스라 값은 정수입니다. 정수가 아니면 그대로 두고 표시합니다.
            if np.allclose(a, np.round(a)):
                a = np.round(a).astype(np.int64)
                dt = _dtype_for(a)
            else:
                dt = '<f4'
        else:
            dt = _dtype_for(a.astype(np.int64))
        fn = name.replace('.', '_').replace('/', '_') + '.bin'
        a.astype(dt).tofile(os.path.join(args.dst, fn))
        sites.append(dict(name=name, file=fn, dtype=dt,
                          shape=list(a.shape), note=note))

    # 입력
    off, _T = int(pre.samples[args.sample][0]), int(pre.samples[args.sample][1])
    rows = pre.index[off:off + T]
    emit('input.n_tok', rows[:, 1], '타임스텝별 유효 토큰 수 (마스크)')
    emit('input.tokens_int8',
         np.concatenate([pre.tok8[int(r[0]):int(r[0]) + int(r[1])] for r in rows]),
         '유효 토큰만 이어붙임 (n, 144)')
    emit('input.pos_idx',
         np.concatenate([pre.pos[int(r[0]):int(r[0]) + int(r[1])] for r in rows]),
         '(y//6)*21 + (x//6)')
    emit('label', np.array([label]))
    emit('pred', np.array([pred]))
    emit('logits', _np(logits[0]), '분류 헤드 출력 (argmax 전)')

    # 모듈 사이트 — 타임스텝마다 한 번씩 불리므로 call 순으로 이어붙입니다
    for name, calls in store.items():
        for k, c in enumerate(calls):
            if c['out'] is not None:
                emit(f'{name}#{k}.out', c['out'])

    # attention 내부
    for blk, d in attn.items():
        for k, v in d.items():
            emit(f'{blk}.{k}', _np(v))

    # 정수 MAC — 순서 기반. RTL 은 이 순서 그대로 실행합니다.
    #
    # attention MAC 은 **피연산자도** 저장합니다. 둘 다 activation 이라 가중치처럼
    # 중복되지 않고, RTL attention 테스트벤치가 그대로 먹습니다:
    #   attn_QK^T : a = Q_int(int8),      b = K_int(int8)
    #   attn_AV   : a = attn_int(uint8),  b = V_int(int8)   ← a 가 softmax 출력
    # 특히 `attn_AV` 의 a 로 **마스크된 키가 정말 0 인지**를 직접 볼 수 있습니다.
    for i, m in enumerate(macs):
        emit(f'mac{i:03d}.{m["tag"]}.acc', m['acc'])
        if m['tag'].startswith('attn_'):
            emit(f'mac{i:03d}.{m["tag"]}.a', m['a'])
            emit(f'mac{i:03d}.{m["tag"]}.b', m['b'])

    mani = dict(dataset=DATASET, sample=args.sample, T=T, label=label, pred=pred,
                n_mac=len(macs),
                mac_order=[m['tag'] for m in macs],
                sites=sites)
    json.dump(mani, open(os.path.join(args.dst, 'manifest.json'), 'w'), indent=1)

    total = sum(os.path.getsize(os.path.join(args.dst, s['file'])) for s in sites)
    print(f'\n-> {args.dst}   사이트 {len(sites)}개, {total/1024:.1f} KiB')
    print(f'\n  MAC 호출 순서 (타임스텝 {T}개 분):')
    from collections import Counter
    for tag, c in Counter(m['tag'] for m in macs).most_common():
        print(f'    {tag:<24}{c:>5}회')


if __name__ == '__main__':
    main()
