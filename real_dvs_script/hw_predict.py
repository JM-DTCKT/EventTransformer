"""**하드웨어와 같은 포맷**으로 샘플별 예측을 찍습니다 (보드 대조용)

    python3 hw_predict.py --n 8

`golden.py` 의 예측(`pred_golden.int32.bin`)은 **원래 골든**입니다 — 정확한 GELU,
float softmax, fp32 잔차/LayerNorm 입력. 보드는 그 넷이 다 다릅니다:

    GELU        64세그먼트 PWL          (`gelu_pwl.v` 와 비트 동일)
    softmax     정수 exp/recip LUT
    잔차 x_input preproc 출력의 int8 격자 (A_Mem 이 int8)
    latent 초기값 bf16

그래서 보드가 틀린 샘플을 원래 골든과 비교하면 **엉뚱한 곳을 파게 됩니다.**
여기서 나온 예측이 보드가 맞춰야 할 값입니다.

(여전히 다른 것 하나 : LayerNorm 입력 격자. 골든은 Q16.-1(lsb 2.0)에 스냅하고
 하드웨어는 bf16 이라 **하드웨어가 더 정밀**합니다 — README §5 ⑥.)
"""
import argparse, os, sys, time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVT = os.path.dirname(SCRIPT_DIR)
for p in (EVT, os.path.join(EVT, 'quantization')):
    if p not in sys.path:
        sys.path.insert(0, p)
from golden import Preprocessed, load_int_model, DATASET      # noqa: E402
from hw_format import build as hw_build                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--first', type=int, default=0)
    args = ap.parse_args()

    pre = Preprocessed(args.data)
    model, _p, _a = load_int_model('cpu')
    hw_build(model, {}, use_pwl=True, gelu1_int8=False)

    ref = None
    f = os.path.join(args.data, 'pred_golden.int32.bin')
    if os.path.exists(f):
        ref = np.fromfile(f, dtype='<i4')

    print(f'[hw_predict] {DATASET}  샘플 {args.first}..{args.first+args.n-1}'
          f'  (하드웨어 포맷)')
    print(f"  {'샘플':>5} {'T':>3} {'라벨':>5} {'HW골든':>7} {'원골든':>7}  일치")
    n_ok = n_diff = 0
    t0 = time.time()
    with torch.no_grad():
        for s in range(args.first, args.first + args.n):
            pol, pix, lab = pre.sample(s)
            _e, logits = model(pol, pix)
            pr = int(logits.float().argmax(-1)[0])
            n_ok += int(pr == lab)
            r = int(ref[s]) if ref is not None else -1
            n_diff += int(r != pr)
            print(f'  {s:>5} {pol.shape[0]:>3} {lab:>5} {pr:>7} {r:>7}   '
                  f'{"O" if pr == lab else "X"}'
                  f'{"   <- 원골든과 다름" if r != pr else ""}', flush=True)

    n = args.n
    print(f'\n  하드웨어 포맷 정확도 : {n_ok}/{n}')
    print(f'  원래 골든과 예측이 갈린 샘플 : {n_diff}개')
    print(f'  {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
