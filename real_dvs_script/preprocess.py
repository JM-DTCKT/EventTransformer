"""DVS128 테스트셋 → **결정론적** 보드 입력 텐서

    conda activate evt_new
    python3 preprocess.py                 # 전체 264 샘플
    python3 preprocess.py --n 8           # 브링업용 8개만

## 왜 별도 전처리가 필요한가

`nonlinear_script` 은 MNIST 라 이미지를 그대로 넣으면 됐습니다. EvT 입력은
**이벤트 스트림을 패치 토큰으로 바꾼 것**이라 전처리 자체가 파이프라인입니다:

    aedat → 12 ms sparse frame → chunk(24 ms = 2 frame) → bins 2 → 6x6 패치
          → 활성 패치만 추림(>= 3픽셀) → log(1+p) → 토큰 144차원

이걸 보드에서 할 수는 없으니(호스트에서 미리 해 둡니다), **학습/평가 때와 한 글자도
다르지 않아야** 합니다. 그래서 새로 구현하지 않고 `data_generation.py::Event_DataModule`
을 **validation 모드**로 그대로 돌립니다.

## 결정론

`data_generation.py` 는 `__getitem__` 안에서 `np.random` 으로 증강합니다(랜덤 시간창,
크롭, 플립, 토큰 드롭). `validation=True` 면 이 경로들이 전부 꺼지므로 재실행해도
같은 텐서가 나옵니다. `val_dataloader()` 가 validation 데이터셋을 줍니다.

실측(테스트셋 264 샘플):

    T (timestep)      min  3   중앙 20   max  20
    활성 토큰/스텝     min 16   중앙 43   max 123
    0토큰 스텝         없음

## 출력

    tokens.int8.bin    활성 토큰만 이어붙임, 토큰 하나 = 144 B
    pos_idx.int16.bin  토큰별 positional encoding 인덱스 (y//6)*21 + (x//6)
    index.int32.bin    샘플/타임스텝 경계 (아래 참조)
    labels.int32.bin   샘플별 정답

`index` 는 샘플 s 의 타임스텝 t 마다 4워드입니다:

    [tok_off, n_tok, 0, 0]     tok_off = tokens.bin 안의 시작 토큰 번호

이렇게 **활성 토큰만** 저장하면 패딩(전체의 60%)을 안 실어 보내도 됩니다. 마스크는
`n_tok` 하나로 표현됩니다 — 보드는 `j < n_tok` 만 유효 키로 봅니다.

## int8 인코딩

첫 Linear(`event_projection.seq_init.0`)의 입력 step 을 그대로 씁니다
(`fpga_export/<dataset>/manifest.json`). 즉 보드가 받는 것은 이미 정수 코드입니다:

    x_int = clamp(round(log(1+p) / step), -128, 127)

fp32 텐서도 함께 저장해 골든이 같은 것을 보게 합니다.
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

DATASET = 'DVS128_10'
POS_GRID = 21                 # 128 // downsample_pos_enc(6)
DOWNSAMPLE = 6
TOKEN_DIM = 144               # patch 6x6 x bins 2 x polarity 2


def load_cfg():
    ap = json.load(open(os.path.join(QUANT_DIR, DATASET, 'all_params.json')))
    mf = json.load(open(os.path.join(QUANT_DIR, 'fpga_export', DATASET,
                                     'manifest.json')))
    return ap, mf


def input_step(mf):
    """첫 Linear 의 입력 step — 보드가 받는 int8 코드의 LSB."""
    for L in mf['layers']:
        if L['name'].endswith('event_projection.seq_init.0'):
            return float(L['input']['step'])
    raise SystemExit('event_projection.seq_init.0 을 manifest 에서 못 찾음')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dst', default=os.path.join(SCRIPT_DIR, 'data'))
    p.add_argument('--n', type=int, default=0, help='샘플 수 (0=전체)')
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    ap, mf = load_cfg()
    step = input_step(mf)

    from quant_lib import data_utils as DU          # noqa: E402
    dm = DU.build_datamodule(ap, workers=args.workers, batch_size=1)
    loader = dm.val_dataloader()

    print(f'[preprocess] {DATASET}  data_folder={dm.data_folder}')
    print(f'  입력 step = {step:.10g}  (첫 Linear 의 입력 LSB)')

    tok_i8, pos_ix, index, labels = [], [], [], []
    fp32_chunks = []
    n_sample = n_step = n_token = 0
    stat_T, stat_N = [], []

    for polarity, pixels, label in loader:
        if polarity is None:
            continue
        # polarity (T, B=1, Npad, 144)   pixels (T, B=1, Npad, 2)
        pol = polarity[:, 0]                                  # (T, Npad, 144)
        pix = pixels[:, 0].long()                             # (T, Npad, 2)
        T = pol.shape[0]

        # 활성 토큰 = 패딩이 아닌 것. 모델의 `samples_mask = kv.sum(-1)==0` 과 같은 식.
        active = (pol.sum(-1) != 0)                           # (T, Npad)

        for t in range(T):
            sel = active[t]
            n = int(sel.sum())
            index.append([n_token, n, 0, 0])
            if n:
                x = pol[t][sel]                               # (n, 144) float
                q = pix[t][sel] // DOWNSAMPLE                 # (n, 2)
                # 모델은 pos_encoding[pixels[...,1], pixels[...,0], :] 로 읽습니다
                idx = q[:, 1] * POS_GRID + q[:, 0]
                assert int(idx.max()) < POS_GRID * POS_GRID, int(idx.max())

                xi = torch.clamp(torch.round(x.double() / step), -128, 127)
                tok_i8.append(xi.to(torch.int8).numpy())
                pos_ix.append(idx.to(torch.int16).numpy())
                fp32_chunks.append(x.float().numpy())
                n_token += n
            stat_N.append(n)
        stat_T.append(T)
        labels.append(int(label[0]))
        n_sample += 1
        n_step += T
        if args.n and n_sample >= args.n:
            break
        if n_sample % 50 == 0:
            print(f'  {n_sample} 샘플', end='\r')

    tok = np.concatenate(tok_i8) if tok_i8 else np.zeros((0, TOKEN_DIM), np.int8)
    pos = np.concatenate(pos_ix) if pos_ix else np.zeros((0,), np.int16)
    idx = np.asarray(index, dtype='<i4')
    lab = np.asarray(labels, dtype='<i4')
    f32 = np.concatenate(fp32_chunks) if fp32_chunks else np.zeros((0, TOKEN_DIM), np.float32)

    # 샘플 경계 = 타임스텝 인덱스의 시작 위치
    starts, c = [], 0
    for T in stat_T:
        starts.append([c, T, 0, 0])
        c += T
    sam = np.asarray(starts, dtype='<i4')

    tok.astype(np.int8).tofile(os.path.join(args.dst, 'tokens.int8.bin'))
    pos.astype('<i2').tofile(os.path.join(args.dst, 'pos_idx.int16.bin'))
    idx.tofile(os.path.join(args.dst, 'index.int32.bin'))
    sam.tofile(os.path.join(args.dst, 'samples.int32.bin'))
    lab.tofile(os.path.join(args.dst, 'labels.int32.bin'))
    f32.astype('<f4').tofile(os.path.join(args.dst, 'tokens.fp32.bin'))

    sT, sN = np.asarray(stat_T), np.asarray(stat_N)
    meta = dict(
        dataset=DATASET, n_sample=n_sample, n_step=n_step, n_token=n_token,
        token_dim=TOKEN_DIM, pos_grid=POS_GRID, downsample=DOWNSAMPLE,
        input_step=step,
        T_max=int(sT.max()), T_min=int(sT.min()),
        N_max=int(sN.max()), N_min=int(sN.min()), N_mean=float(sN.mean()),
        files=dict(
            tokens='tokens.int8.bin  (n_token, 144) int8',
            tokens_fp32='tokens.fp32.bin  (n_token, 144) float32 — 골든 대조용',
            pos_idx='pos_idx.int16.bin  (n_token,) int16 = (y//6)*21 + (x//6)',
            index='index.int32.bin  (n_step, 4) = [tok_off, n_tok, 0, 0]',
            samples='samples.int32.bin  (n_sample, 4) = [step_off, T, 0, 0]',
            labels='labels.int32.bin  (n_sample,) int32'))
    json.dump(meta, open(os.path.join(args.dst, 'meta.json'), 'w'), indent=2)

    print(f'\n-> {args.dst}\n')
    print(f"{'':<14}{'값':>10}")
    print(f"{'샘플':<14}{n_sample:>10,}")
    print(f"{'타임스텝':<14}{n_step:>10,}   T {sT.min()}~{sT.max()} (평균 {sT.mean():.1f})")
    print(f"{'활성 토큰':<14}{n_token:>10,}   /스텝 {sN.min()}~{sN.max()} (평균 {sN.mean():.1f})")
    print()
    for f in ('tokens.int8.bin', 'tokens.fp32.bin', 'pos_idx.int16.bin',
              'index.int32.bin', 'samples.int32.bin', 'labels.int32.bin'):
        print(f'  {f:<20}{os.path.getsize(os.path.join(args.dst, f)):>12,} B')

    # 패딩을 안 실어 보내서 아낀 양
    padded = n_step * int(sN.max()) * TOKEN_DIM
    print(f'\n  패딩 포함 저장 시 {padded:,} B → 활성만 {n_token*TOKEN_DIM:,} B '
          f'({100*(1-n_token*TOKEN_DIM/padded):.0f}% 절약)')


if __name__ == '__main__':
    main()
