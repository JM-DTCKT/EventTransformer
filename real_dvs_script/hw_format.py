"""하드웨어가 **추가로 넣는 양자화 두 곳**의 정확도 비용을 잽니다

    python3 hw_format.py --calib 32          # 보정만
    python3 hw_format.py                     # 264 샘플 정확도

## 왜 재는가

`fpga_dvs128_10` 의 A_Mem 은 워드 하나가 32레인 x 16b 이고, 레인 값은 **int8 코드**
아니면 **bf16** 입니다. 골든의 중간값이 Q4.11 실수인 자리가 두 곳 있는데, 거기서
하드웨어가 int8 로 한 번 더 떨어뜨립니다:

    ① linear1 → GELU → **int8** → layer_norm_2        (골든은 Q4.11 그대로 LN 입력)
    ② proc_events 의 잔차 `x + x_input` 에서 `x_input`
       = preproc 의 GELU 출력. 골든은 Q4.11, 하드웨어는 **proc_ev.1 이 읽는 int8**

①은 LayerNorm 이 스케일 불변이라 step 을 자유롭게 고를 수 있습니다 — 보정으로
`max|gelu1_out| / 127` 을 씁니다. ②는 이미 있는 step(0.0384197) 을 그대로 씁니다.

## 대안(둘 다 비쌈)

①을 피하려면 LayerNorm 이 Q4.11 int16 을 받게 해야 하고(레인마다 int→bf16 변환),
②를 피하려면 preproc 을 **두 번** 돌려 bf16 사본을 따로 저장해야 합니다
(타임스텝당 GEMM 1개 추가 + A_Mem 영역 1개 추가).

정확도가 안 떨어지면 둘 다 안 만드는 것이 맞습니다. `int_softmax.py`/`pwl_gelu.py`
와 같은 판단 절차입니다.
"""
import argparse, os, sys, time, types
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVT = os.path.dirname(SCRIPT_DIR)
for p in (EVT, os.path.join(EVT, 'quantization')):
    if p not in sys.path:
        sys.path.insert(0, p)
from golden import Preprocessed, load_int_model, DATASET      # noqa: E402
from pwl_gelu import gelu_pwl_code, QF                        # noqa: E402

LSB = 2.0 ** -QF
# proc_ev.1 이 읽는 입력 step (manifest layers[].input.step) — ②가 쓰는 격자
PRE_STEP = 0.03841966390609741


def q8(x, step):
    return torch.clamp(torch.round(x / step), -128, 127) * step


def build(model, gelu1_steps, use_pwl=True, gelu1_int8=True):
    """골든 모델에 ①② 를 심습니다. `gelu1_steps` 가 None 이면 보정 모드."""
    cal = {}

    def wrap_gelu(pair, key, extra_q):
        """`s_out(F.gelu(s_in(z)))` 자리를 PWL + (선택)int8 로 바꿉니다."""
        s_in, s_out = pair
        box = {}

        def new_in(z):
            y = s_in(z)
            box['z'] = y
            return y

        def new_out(_ignored):
            code = torch.round(box['z'].double() / LSB).long()
            y = (gelu_pwl_code(code).double() * LSB).to(box['z'].dtype) \
                if use_pwl else s_out(_ignored)
            if extra_q:
                if gelu1_steps is None:                 # 보정 : 범위만 모읍니다
                    cal[key] = max(cal.get(key, 0.0), float(y.abs().max()))
                else:
                    y = q8(y, gelu1_steps[key])
            return y

        return (new_in, new_out)

    n_g = 0
    for _n, m in model.named_modules():                  # MLPBlock 안의 nn.GELU
        if isinstance(m, nn.GELU):
            m._sites = wrap_gelu(m._sites, _n, False)
            si, so = m._sites
            m.forward = types.MethodType(lambda self, x, si=si, so=so: so(si(x)), m)
            n_g += 1
    for nm, mod in model.named_modules():                # AttentionBlock 안의 GELU
        if hasattr(mod, '_hw_ctx'):
            c = mod._hw_ctx
            # gelu1 뒤에만 ①이 붙습니다 (gelu2 뒤는 linear3 의 int8 입력이라 원래 있음)
            c['gelu1'] = wrap_gelu(c['gelu1'], nm, gelu1_int8)
            c['gelu2'] = wrap_gelu(c['gelu2'], nm + '#2', False)
            n_g += 2
    assert n_g == 8, f'GELU 사이트 {n_g}개 (8이어야 함)'

    # ---- ④ softmax 도 하드웨어와 같은 정수 알고리즘 ----
    from int_softmax import install as install_softmax          # noqa: E402
    install_softmax(model)

    # ---- ③ latent 초기값은 하드웨어에서 bf16 ----
    # Z / LATV 는 잔차 스트림이라 A_Mem 에 **bf16** 으로 삽니다. 초기값
    # (`memory_vertical`) 도 마찬가지라 골든도 같은 격자에 올려야 합니다.
    # 안 하면 `layer_norm_1` 출력이 0.4 % 자리에서 1 LSB 씩 어긋납니다.
    bb = model.backbone if hasattr(model, 'backbone') else None
    for nm2, prm in model.named_parameters():
        if nm2.endswith('memory_vertical'):
            prm.data = prm.data.to(torch.bfloat16).float()

    # ---- ② proc_events 의 잔차 ----
    n_r = 0
    from models.EvT import MLPBlock                       # noqa: E402
    for nm, mod in model.named_modules():
        if isinstance(mod, MLPBlock) and getattr(mod, 'add_x_input', False):
            def fwd(self, x_input, mask=None, pos_embs=None, **a):
                x = self.seq_init(x_input)
                return x + q8(x_input, PRE_STEP)          # ← 하드웨어는 int8 사본
            mod.forward = types.MethodType(fwd, mod)
            n_r += 1
    assert n_r == 1, f'add_x_input MLPBlock {n_r}개 (1이어야 함)'
    return cal


def run(model, pre, n, tag):
    correct, t0 = 0, time.time()
    with torch.no_grad():
        for s in range(n):
            pol, pix, lab = pre.sample(s)
            _e, logits = model(pol, pix)
            correct += int(int(logits.float().argmax(-1)[0]) == lab)
            if (s + 1) % 20 == 0 or s + 1 == n:
                el = time.time() - t0
                print(f'  [{tag}] {s+1:>4}/{n}  정답 {correct}  {el:.0f}s '
                      f'남은 {el/(s+1)*(n-s-1):.0f}s', flush=True)
    return correct


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--calib', type=int, default=24)
    ap.add_argument('--steps', default=os.path.join(SCRIPT_DIR, 'data',
                                                    'hw_format_steps.json'))
    args = ap.parse_args()
    pre = Preprocessed(args.data)
    n = args.n or len(pre)

    # ---- 보정 : gelu1 출력 범위 ----
    import json
    if os.path.exists(args.steps):
        steps = json.load(open(args.steps))
        print(f'[hw_format] 보정값 재사용 {args.steps}')
    else:
        model, _p, _a = load_int_model('cpu')
        cal = build(model, None)
        print(f'[hw_format] 보정 {args.calib} 샘플 …', flush=True)
        run(model, pre, args.calib, 'calib')
        steps = {k: v / 127.0 for k, v in cal.items()}
        json.dump(steps, open(args.steps, 'w'), indent=1)
    for k, v in steps.items():
        print(f'    gelu1 step  {k:<58} {v:.6g}')

    model, _p, _a = load_int_model('cpu')
    build(model, steps)
    print(f'\n[hw_format] {DATASET}  샘플 {n}/{len(pre)}  ①gelu1→int8 ②잔차 int8',
          flush=True)
    correct = run(model, pre, n, 'hw')
    acc = correct / n
    print(f'\n  정확도(하드웨어 포맷) : {correct}/{n} = {acc*100:.4f} %')
    print(f'  기준(PWL GELU 만)     : 257/264 = 97.3485 %')
    print(f'  차이 : {correct-257:+d} 샘플 = {(acc-0.973485)*100:+.4f} %p')


if __name__ == '__main__':
    main()
