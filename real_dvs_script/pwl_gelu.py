"""GELU 를 **64세그먼트 PWL 근사**로 바꾸고 정확도 영향을 잽니다

    python3 pwl_gelu.py            # 264 샘플
    python3 pwl_gelu.py --n 32

## 왜 재야 하는가

`04_basic_rtl/GELU` 는 전수 LUT(16,384 x 16b = 8 BRAM36, 32레인이면 **256 BRAM**)을
**base/delta 64쌍 = 2 Kb** 로 줄인 PWL 근사입니다. Q4.11 전 코드에서 정확한 GELU 대비
**최대 1 LSB**(65,536 중 96.6%는 정확)지만, 1 LSB 가 네트워크 정확도에 얼마나
번지는지는 **재 봐야 압니다** — GELU 는 타임스텝마다 6곳(블록 3 x 2)에서 불립니다.

`int_softmax.py` 와 같은 방식으로 골든의 GELU 사이트만 갈아끼웁니다.
**원본 `quantization/` 은 건드리지 않습니다.**

## PWL 식 (`GELU/verilog/gelu_pwl.v`)

    a = |code|,  seg = a >> 7,  frac = a & 127        (Q4.11, 세그먼트 폭 2^-4)
    R14 = (a >= 8192) ? 0 : base14[seg] + ((delta14[seg]*frac) >> 7)
    R   = (R14 + 2^2) >> 3                            (Q14 → Q4.11, 반올림)
    y   = (code < 0) ? -R : code - R
"""
import argparse, os, sys
import numpy as np, torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVT = os.path.dirname(SCRIPT_DIR)
for p in (EVT, os.path.join(EVT, 'quantization')):
    if p not in sys.path:
        sys.path.insert(0, p)
from golden import Preprocessed, load_int_model, DATASET   # noqa: E402

LUT = np.load('/hai/home/sgh/04_basic_rtl/GELU/sw/pwl_model.npz')
BASE = torch.as_tensor(LUT['base'].astype(np.int64))
DELTA = torch.as_tensor(LUT['delta'].astype(np.int64))
QF, FR, FRACBITS, NSEG, AMAX = 11, 14, 7, 64, 4 << 11
RSH = FR - QF


def gelu_pwl_code(code):
    """Q4.11 코드 → Q4.11 코드. `gelu_pwl.v` 와 **비트 단위로 동일**."""
    c = code.long()
    a = c.abs()
    ge4 = a >= AMAX
    seg = torch.clamp(a >> FRACBITS, 0, NSEG - 1)
    frac = a & ((1 << FRACBITS) - 1)
    interp = (DELTA.to(c.device)[seg] * frac) >> FRACBITS
    r14 = torch.where(ge4, torch.zeros_like(a), BASE.to(c.device)[seg] + interp)
    r = (r14 + (1 << (RSH - 1))) >> RSH
    y = torch.where(c < 0, -r, c - r)
    return torch.clamp(y, -32768, 32767)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--n', type=int, default=0)
    args = ap.parse_args()

    pre = Preprocessed(args.data)
    n = args.n or len(pre)
    model, payload, _ = load_int_model('cpu')

    import torch.nn as nn, types                                  # noqa: E402
    LSB = 2.0 ** -QF

    def wrap_pair(pair):
        """(site_in, site_out) → PWL 로 계산하는 (site_in, 새 site_out)

        골든은 `s_out(F.gelu(s_in(z)))` 로 씁니다. `s_in` 이 Q4.11 격자로 스냅한
        값을 기억해 두었다가, `s_out` 자리에서 **그 코드로 PWL 을 다시 계산**합니다.
        (인자로 들어오는 float GELU 결과는 버립니다.)
        """
        s_in, s_out = pair
        box = {}

        def new_in(z):
            y = s_in(z)
            box['z'] = y
            return y

        def new_out(_ignored):
            code = torch.round(box['z'].double() / LSB).long()
            return (gelu_pwl_code(code).double() * LSB).to(box['z'].dtype)

        return (new_in, new_out)

    n_site = 0
    # ---- ① MLPBlock 안의 nn.GELU (Sequential 의 원소라 모듈로 잡힘) ----
    for _name, m in model.named_modules():
        if isinstance(m, nn.GELU):
            m._sites = wrap_pair(m._sites)
            s_in, s_out = m._sites

            def fwd(self, x, s_in=s_in, s_out=s_out):
                return s_out(s_in(x))

            m.forward = types.MethodType(fwd, m)
            n_site += 1

    # ---- ② AttentionBlock 안의 GELU ----
    # `EvT.py:61,65` 는 `torch.nn.GELU()(z)` 로 **매 forward 새 인스턴스**를
    # 만듭니다. 서브모듈이 아니라 `named_modules()` 에 안 잡히므로 ①로는 못 잡습니다.
    # `hw_quant._patch_attention_blocks` 가 forward 를 통째로 갈아끼우면서
    # `ctx['gelu1'] / ctx['gelu2']` 로 처리하므로 거기를 바꿉니다.
    for _name, mod in model.named_modules():
        if hasattr(mod, '_hw_ctx'):
            c = mod._hw_ctx
            for k in ('gelu1', 'gelu2'):
                if k in c:
                    c[k] = wrap_pair(c[k])
                    n_site += 1

    print(f'[pwl_gelu] {DATASET}  샘플 {n}/{len(pre)}  GELU 사이트 {n_site}개 교체')
    if n_site != 8:
        print(f'  ** 사이트가 8개가 아닙니다 ({n_site}) — MLP 2 + attention 3블록 x 2 = 8')
        return

    correct = 0
    with torch.no_grad():
        import time
        t0 = time.time()
        for s in range(n):
            pol, pix, lab = pre.sample(s)
            _e, logits = model(pol, pix)
            correct += int(int(logits.float().argmax(-1)[0]) == lab)
            if (s + 1) % 20 == 0 or s + 1 == n:
                el = time.time() - t0
                print(f'  {s+1:>4}/{n}  정답 {correct}  '
                      f'{el:.0f}s 경과, 남은 예상 {el/(s+1)*(n-s-1):.0f}s',
                      flush=True)

    acc = correct / n
    print(f'\n  정확도(PWL GELU) : {correct}/{n} = {acc*100:.4f} %')
    print(f'  기준(정확한 GELU) : 97.3485 %  (257/264)')
    print(f'  차이 : {correct-257:+d} 샘플 = {(acc-0.973485)*100:+.4f} %p'
          f'   (1 샘플 = 0.379 %p)')


if __name__ == '__main__':
    main()
