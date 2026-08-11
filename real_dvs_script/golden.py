"""전처리 결과 → 골든 정수 데이터패스 (RTL 레퍼런스)

    python3 golden.py                 # 전체 264 샘플 정확도
    python3 golden.py --n 8           # 8개만

## 무엇을 검증하나

`preprocess.py` 가 만든 파일만으로 모델을 돌려 **`quantization/README.md` 의
DVS128_10 정수 정확도(97.3485 %)가 재현되는지** 봅니다. 재현되면 두 가지가 동시에
확정됩니다:

 1. 전처리가 학습/평가 때와 같다
 2. 활성 토큰만 저장하고 `n_tok` 로 마스크를 표현한 것이 원래 패딩 표현과 동등하다

②가 중요합니다. 원래 파이프라인은 `(T, B, Npad, 144)` 로 **앞쪽 0 패딩**을 하고
`mask = kv.sum(-1)==0` 으로 마스크를 만듭니다. 보드는 패딩을 아예 안 받고 `n_tok`
하나만 받으므로, 그 둘이 같은 결과를 내는지 확인해야 합니다.

## 정수 데이터패스는 새로 만들지 않습니다

`quantization/quant_lib/hw_quant.py` 의 `instantiate()` 가 fp32 모델의 모듈을
정수 연산으로 갈아끼웁니다. 이미 22개 GEMM 전부를 순수 정수식과 대조해 둔 코드라
(README 의 "100.00 % within 1 LSB"), 여기서 다시 쓰면 **레퍼런스가 두 벌**이 됩니다.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVT_DIR = os.path.dirname(SCRIPT_DIR)
QUANT_DIR = os.path.join(EVT_DIR, 'quantization')
for p in (EVT_DIR, QUANT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

DATASET = 'DVS128_10'
POS_GRID, DOWNSAMPLE, TOKEN_DIM = 21, 6, 144


class Preprocessed:
    """`preprocess.py` 출력을 읽어 모델이 먹는 형태로 되돌립니다."""

    def __init__(self, d):
        self.meta = json.load(open(os.path.join(d, 'meta.json')))
        n_tok = self.meta['n_token']
        self.tok8 = np.fromfile(os.path.join(d, 'tokens.int8.bin'),
                                dtype=np.int8).reshape(n_tok, TOKEN_DIM)
        self.tokf = np.fromfile(os.path.join(d, 'tokens.fp32.bin'),
                                dtype='<f4').reshape(n_tok, TOKEN_DIM)
        self.pos = np.fromfile(os.path.join(d, 'pos_idx.int16.bin'), dtype='<i2')
        self.index = np.fromfile(os.path.join(d, 'index.int32.bin'),
                                 dtype='<i4').reshape(-1, 4)
        self.samples = np.fromfile(os.path.join(d, 'samples.int32.bin'),
                                   dtype='<i4').reshape(-1, 4)
        self.labels = np.fromfile(os.path.join(d, 'labels.int32.bin'), dtype='<i4')

    def __len__(self):
        return len(self.samples)

    def sample(self, s, use_int8=False):
        """샘플 s → (polarity (T,1,Npad,144), pixels (T,1,Npad,2), label)

        원래 파이프라인과 같이 **앞쪽 0 패딩**으로 복원합니다. `Npad` 는 이 샘플의
        타임스텝 중 최대 토큰 수입니다 (`pad_list_of_sequences` 와 동일).
        """
        off, T = int(self.samples[s][0]), int(self.samples[s][1])
        rows = self.index[off:off + T]
        Npad = max(1, int(rows[:, 1].max()))

        pol = torch.zeros(T, 1, Npad, TOKEN_DIM)
        pix = torch.zeros(T, 1, Npad, 2, dtype=torch.long)
        src = self.tok8 if use_int8 else self.tokf
        for t in range(T):
            o, n = int(rows[t][0]), int(rows[t][1])
            if n == 0:
                continue
            x = torch.from_numpy(np.ascontiguousarray(src[o:o + n])).float()
            if use_int8:
                x = x * self.meta['input_step']
            pol[t, 0, -n:] = x                       # ← 앞쪽 패딩(pre_padding)
            idx = torch.from_numpy(np.ascontiguousarray(self.pos[o:o + n])).long()
            # 모델은 pixels//6 을 [y, x] 로 읽습니다. 인덱스를 되돌립니다.
            pix[t, 0, -n:, 1] = (idx // POS_GRID) * DOWNSAMPLE
            pix[t, 0, -n:, 0] = (idx % POS_GRID) * DOWNSAMPLE
        return pol, pix, int(self.labels[s])


def load_int_model(device):
    from quant_lib import data_utils as DU               # noqa: E402
    from quant_lib import hw_quant as HQ                 # noqa: E402
    ap_path = os.path.join(QUANT_DIR, DATASET, 'all_params.json')
    all_params = json.load(open(ap_path))
    ckpt = DU.find_best_checkpoint(os.path.join(QUANT_DIR, DATASET, 'weights'))
    model, _ap = DU.load_model(ckpt, ap_path, device=device)   # (model, all_params)
    payload = torch.load(os.path.join(QUANT_DIR, DATASET, 'quantized',
                                      'model_int8.pt'),
                         map_location=device, weights_only=False)
    model, _rt = HQ.instantiate(payload, model, device)        # (model, runtime)
    model.eval()
    return model, payload, all_params


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'data'))
    p.add_argument('--n', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--int8', action='store_true',
                   help='fp32 대신 int8 코드를 되돌려 입력 (보드와 동일)')
    args = p.parse_args()

    pre = Preprocessed(args.data)
    n = args.n or len(pre)
    print(f'[golden] {DATASET}  샘플 {n}/{len(pre)}  '
          f'{"int8 코드 복원" if args.int8 else "fp32"} 입력')

    model, payload, _ = load_int_model(args.device)

    correct = 0
    preds = np.zeros(n, dtype='<i4')
    with torch.no_grad():
        for s in tqdm(range(n), desc='golden', leave=False):
            pol, pix, lab = pre.sample(s, use_int8=args.int8)
            _embs, logits = model(pol.to(args.device), pix.to(args.device))
            pr = int(logits.float().argmax(dim=-1)[0])
            preds[s] = pr
            correct += int(pr == lab)

    acc = correct / n
    print(f'\n  정확도 : {correct}/{n} = {acc*100:.4f} %')
    print(f'  기준   : quantization/README.md 의 DVS128_10 정수 97.3485 %')
    ok = abs(acc - 0.973485) < 1e-6 if n == len(pre) else None
    if ok is True:
        print('  → 전처리·마스크 표현이 원 파이프라인과 동등함 ✅')
    elif ok is False:
        print('  → **불일치** — 전처리 또는 마스크 표현을 확인하세요')

    np.asarray(preds, dtype='<i4').tofile(os.path.join(args.data, 'pred_golden.int32.bin'))
    print(f'  예측 저장 : {os.path.join(args.data, "pred_golden.int32.bin")}')


if __name__ == '__main__':
    main()
