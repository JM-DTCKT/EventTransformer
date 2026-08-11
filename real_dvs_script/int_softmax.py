"""attention softmax 를 **하드웨어와 같은 정수 LUT 파이프라인**으로 교체

    python3 int_softmax.py            # 264 샘플 정확도 재측정
    python3 int_softmax.py --n 32

## 왜 필요한가

`quant_lib/hw_quant.py` 의 attention softmax 는 이렇게 되어 있습니다:

    scores   = site_scores(raw * inv_sqrt_d)        # Q6.9 격자로 스냅
    attn     = site_attn(torch.softmax(scores))     # ← float softmax 후 Q1.14 스냅
    attn_int = clamp(round(attn / step), 0, 127)

즉 **float softmax 를 격자에 스냅한 시뮬레이터**입니다. 반면 `hw_flow.md` §2.4 와
RTL(`Softmax_Attn.v`)은 **exp LUT + 역수 LUT 정수 파이프라인**입니다:

    d[j] = clamp(score[j] - max, -16, 0)            Q6.9
    e[j] = EXP_LUT[-d[j]]                           Q1.14
    S    = Σ e[j]                                   ≥ 2^14
    r    = RECIP_LUT[normalize(S)]                  127·2^22 / S
    p[j] = sat_u8( (e[j]·r + 2^(21+sh)) >> (22+sh) )

둘은 1 LSB 이내로 일치하지만 **비트 단위로 같지는 않습니다.** 실제로 실데이터
1,696개 중 1개가 어긋났습니다(`tb_softmax_attn`).

RTL 레퍼런스는 비트 단위로 같아야 합니다. 안 그러면 보드에서 1 LSB 차이가 났을 때
"버그인가 알고리즘 차이인가" 를 매번 다시 따져야 합니다. 그래서 골든 쪽을 하드웨어에
맞춥니다 — 하드웨어를 골든에 맞출 수는 없습니다(LUT 파이프라인이 §2.4 의 스펙이므로).

## 이 파일이 하는 일

`hw_quant.attention_forward_hw` 를 monkey-patch 해서 softmax 부분만 정수 LUT 으로
바꾸고, 264 샘플 정확도를 다시 잽니다. **원본 파일은 건드리지 않습니다.**

LUT 은 `nonlinear_script/formats.py` 것을 그대로 씁니다 — `fpga_nl` 에서 보드로
검증된 바로 그 테이블입니다.
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
NL_DIR = os.path.join(EVT_DIR, 'nonlinear_script')
for p in (EVT_DIR, QUANT_DIR, NL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from golden import Preprocessed, load_int_model, DATASET      # noqa: E402


def build_luts():
    import formats as F                                        # noqa: E402
    return (torch.as_tensor(F.build_exp_lut()),
            torch.as_tensor(F.build_recip_lut()),
            F.SMAX_RANGE, F.RECIP_N)


EXP_LUT, RCP_LUT, SMAX_RANGE, RECIP_N = build_luts()


def softmax_u8_int(score_code):
    """Q6.9 코드 (..., Lk) → uint8 확률. `Softmax_Attn.v` 와 **비트 단위로 동일**."""
    s = score_code.long()
    d = s - s.max(dim=-1, keepdim=True).values               # ≤ 0
    idx = torch.clamp(-d, 0, SMAX_RANGE)
    oor = idx >= SMAX_RANGE
    e = torch.where(oor, torch.zeros_like(idx),
                    EXP_LUT.to(idx.device)[torch.clamp(idx, max=SMAX_RANGE - 1)])

    S = e.sum(dim=-1, keepdim=True)                           # ≥ 2^14
    # 선행 1 위치 → [2^14, 2^15) 로 정규화
    blen = torch.zeros_like(S)
    v = S.clone()
    while bool((v > 0).any()):
        blen += (v > 0).long()
        v = v >> 1
    sh = torch.clamp(blen - 15, min=0)
    Sn = S >> sh
    r = RCP_LUT.to(S.device)[Sn - (1 << 14)]

    num = e * r + (1 << (RECIP_N - 1)) * (1 << sh)
    p = num >> (RECIP_N + sh)
    return torch.clamp(p, 0, 127)


def install(model):
    """골든의 softmax 를 **하드웨어와 같은 정수 알고리즘**으로 바꿉니다.

    `pwl_gelu` 의 GELU 교체와 짝입니다 — 통합 TB 가 1 LSB 차이를 전부 버그로
    볼 수 있게 하려면 비선형이 골든과 하드웨어에서 같아야 합니다.
    반환값은 두 방식의 코드 차이 통계(`diff`/`total`/`blocks`) 입니다.
    """
    import quant_lib.hw_quant as HQ                            # noqa: E402

    # ---- softmax 교체 ----
    # `attention_forward_hw` 안에서 scores 는 **마스킹 전** 값입니다. 마스크는
    # 그 직후 `masked_fill(-inf)` 로 걸리므로, site_attn 만 가로채면 어느 키가
    # 마스크됐는지 알 수 없습니다. 그래서 함수 자체를 감싸 `key_padding_mask` 를
    # 잡아 둡니다. (float softmax 출력이 0 인지로 추정하면 안 됩니다 — 지수가
    # 아주 작을 때도 0 으로 언더플로할 수 있습니다.)
    orig_fwd = HQ.attention_forward_hw
    cur = dict(mask=None)
    stats = dict(diff=0, total=0, calls=0)

    def fwd(q_in, k_in, v_in, mha, key_padding_mask, ctx):
        cur['mask'] = key_padding_mask
        return orig_fwd(q_in, k_in, v_in, mha, key_padding_mask, ctx)
    HQ.attention_forward_hw = fwd
    # AttentionBlock.forward 가 모듈 로컬로 이미 바인딩했을 수 있으므로 확인
    import quant_lib.hw_quant as _hq
    _hq.attention_forward_hw = fwd

    LSB_S = 2.0 ** -9                                   # scores 는 Q6.9

    def wrap(ctx):
        site_scores, site_attn = ctx['site_scores'], ctx['site_attn']
        box = {}

        def new_scores(x):
            y = site_scores(x)
            box['s'] = y
            return y

        def new_attn(attn_float):
            sc = box['s']                               # (B, H, Lq, Lk) 마스킹 전
            code = torch.round(sc.double() / LSB_S).long()
            m = cur['mask']
            if m is not None:
                # bias_k 자리 1칸이 뒤에 붙어 있습니다 (절대 마스킹 안 됨)
                mm = m
                if mm.shape[-1] == code.shape[-1] - 1:
                    mm = torch.cat([mm, mm.new_zeros(mm.shape[0], 1)], dim=1)
                code = code.masked_fill(mm.view(mm.shape[0], 1, 1, -1), -32768)
            p = softmax_u8_int(code)
            ref = torch.round(site_attn(attn_float).double() / ctx['softmax'])
            stats['diff'] += int((ref.long() != p).sum())
            stats['total'] += int(p.numel())
            stats['calls'] += 1
            return (p.double() * ctx['softmax']).to(attn_float.dtype)

        ctx['site_scores'] = new_scores
        ctx['site_attn'] = new_attn

    n_blk = 0
    for _name, mod in model.named_modules():
        if hasattr(mod, '_hw_ctx'):
            wrap(mod._hw_ctx); n_blk += 1
    assert n_blk == 3, f'attention 블록 {n_blk}개 (3이어야 함)'
    stats['blocks'] = n_blk
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    import quant_lib.hw_quant as HQ                            # noqa: E402

    pre = Preprocessed(args.data)
    n = args.n or len(pre)
    model, payload, _ = load_int_model(args.device)

    stats = install(model)
    n_blk = stats['blocks']
    print(f'[int_softmax] {DATASET}  샘플 {n}/{len(pre)}  attention 블록 {n_blk}개 교체')
    if n_blk == 0:
        print('  ** ctx 를 못 찾았습니다'); return

    correct = 0
    with torch.no_grad():
        for s in tqdm(range(n), desc='int-softmax', leave=False):
            pol, pix, lab = pre.sample(s)
            _e, logits = model(pol.to(args.device), pix.to(args.device))
            correct += int(int(logits.float().argmax(-1)[0]) == lab)

    print(f'\n  정확도(정수 softmax) : {correct}/{n} = {correct/n*100:.4f} %')
    print(f'  기준(골든 float softmax) : 97.3485 %')
    if stats['total']:
        print(f'  두 방식의 코드 차이 : {stats["diff"]}/{stats["total"]} '
              f'({stats["diff"]/stats["total"]*100:.4f} %)')


if __name__ == '__main__':
    main()
