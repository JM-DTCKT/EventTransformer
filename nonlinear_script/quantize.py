"""fp32 FFN → 정수/고정소수 데이터패스. RTL 과 **한 줄씩 대응**하는 골든 모델.

    python quantize.py

`linear_script/quantize.py` 가 int8 경로에서 한 것과 같은 역할이고, 대상이
`hw_flow.md` 의 16비트 포맷과 비선형 유닛으로 넓어졌습니다.

## 데이터패스 (한 이미지가 흘러가는 길)

    x:int8 ─stem──→ acc:INT32 ─bf16(acc·s_x·s_w)─→ x_bf : bf16      §2.6 소비자
    ┌──────────────────── block × 2 ──────────────────────────────┐
    │ LayerNorm  bf16 → (mu, var, rsqrt: bf16) → xhat:Q4.11        │  §2.6
    │            → affine(gamma Q1.14, beta Q4.11) → out:Q4.11     │
    │            → requant → a:int8                                │
    │ fc1        acc:INT32 → requant → Q4.11                       │  §2.3 소비자
    │ GELU       Q4.11 → LUT → Q4.11 → requant → int8              │  §2.3
    │ fc2        acc:INT32 → bf16                                  │  §2.7 소비자
    │ residual   x_bf = bf16(x_bf + h_bf)                          │  §2.7
    └──────────────────────────────────────────────────────────────┘
    LayerNorm → int8 ─head─→ acc:INT32 → requant → Q6.9            §2.4 소비자
    softmax   Q6.9 → exp LUT → Σ → ×1/Σ → uint8                    §2.4

## 누산기 정밀도 — 명시적 선택

LayerNorm 의 `mean` 과 `mean(centered²)` 는 **fp32 로 누산하고 마지막에 한 번만
bf16 으로 반올림**합니다 (`quant_lib/hw_quant.py` 와 동일). bf16 으로 매 단계
반올림하면 128개 합에서 오차가 누적돼 분산이 망가집니다. RTL 에는 fp32 가산기가
하나 필요하고, 두 패스가 공유합니다.

`rsqrt` 는 입력을 먼저 bf16 으로 못박아 **LUT 으로 정확히 구현 가능**하게 했습니다:
bf16 입력은 가수 8비트 + 지수 홀짝 → 256 엔트리면 전수입니다.
"""

import argparse
import math
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data as data_mod                     # noqa: E402
import formats as F                         # noqa: E402
from model import FFNNet                    # noqa: E402

CKPT = os.path.join(SCRIPT_DIR, 'ffn_fp32.pt')
OUT = os.path.join(SCRIPT_DIR, 'ffn_int8.pt')
EPS = 1e-5                                  # nn.LayerNorm 기본값


# =============================================================================
# 캘리브레이션 — fp32 모델에서 각 사이트의 동적 범위를 관측
# =============================================================================
@torch.no_grad()
def calibrate(model, loader, device, n_batches):
    obs = {}
    handles = []

    def watch(key):
        def hook(_m, args, _out=None):
            v = float(args[0].detach().abs().max())
            obs[key] = max(obs.get(key, 0.0), v)
        return hook

    def watch_out(key):
        def hook(_m, _args, out):
            v = float(out.detach().abs().max())
            obs[key] = max(obs.get(key, 0.0), v)
        return hook

    for name, m in model.linears:
        handles.append(m.register_forward_pre_hook(watch(f'{name}.in')))
        handles.append(m.register_forward_hook(watch_out(f'{name}.out')))
    for name, m in model.layernorms:
        handles.append(m.register_forward_pre_hook(watch(f'{name}.in')))
        handles.append(m.register_forward_hook(watch_out(f'{name}.out')))
    for i, b in enumerate(model.blocks):
        handles.append(b.gelu.register_forward_pre_hook(watch(f'b{i}.gelu.in')))
        handles.append(b.gelu.register_forward_hook(watch_out(f'b{i}.gelu.out')))

    model.eval().to(device)
    for j, (x, _) in enumerate(loader):
        if j >= n_batches:
            break
        model(x.to(device))
    for h in handles:
        h.remove()
    return obs


# =============================================================================
# 상수 만들기
# =============================================================================
@torch.no_grad()
def build_payload(model, obs):
    lin = dict(model.linears)
    lnm = dict(model.layernorms)
    nb = model.dims['n_block']

    # ---- int8 activation step: 그 Linear 입력의 관측 범위에서 ----
    s_x = {n: max(obs[f'{n}.in'], 1e-12) / F.INT8_QMAX for n, _ in model.linears}

    layers, notes = {}, []

    def pack_linear(name, consumer, lsb_out=None):
        m = lin[name]
        W, b = m.weight.data, m.bias.data
        s_w = F.weight_scale(W)
        w_int = F.quant_weight(W, s_w)
        step = s_x[name] * s_w                       # 누산기 LSB (채널별)
        b_int = F.fold_bias(b, step)
        e = dict(name=name, consumer=consumer, shape=tuple(W.shape),
                 w_int=w_int.to(torch.int8), b_int=b_int,
                 s_x=s_x[name], s_w=s_w)
        if consumer == 'bf16':
            e['step'] = step                          # bf16(acc · step[c])
            e['M'], e['shift'] = None, None
        else:
            M, sh = F.make_requant(step / lsb_out)
            e['M'], e['shift'], e['lsb_out'] = M, sh, lsb_out
        layers[name] = e
        return e

    # ---- stem → bf16 (LayerNorm 소비) ----
    pack_linear('stem', 'bf16')

    # ---- 블록 ----
    for i in range(nb):
        # LayerNorm: gamma Q1.14, beta Q4.11.  출력 Q4.11 → 다음 GEMM int8
        ln = lnm[f'b{i}.norm']
        g, be = ln.weight.data, ln.bias.data
        if float(g.abs().max()) >= 2.0:
            notes.append(f'b{i}.norm gamma max {float(g.abs().max()):.3f} ≥ 2 → Q1.14 포화')
        M_ln, sh_ln = F.make_requant(torch.tensor([2.0 ** -F.N_LNOUT / s_x[f'b{i}.fc1']]))
        layers[f'b{i}.norm'] = dict(
            name=f'b{i}.norm', kind='layernorm',
            gamma=F.q_enc(g, F.N_GAMMA), beta=F.q_enc(be, F.N_BETA),
            M=M_ln, shift=sh_ln, eps=EPS)

        pack_linear(f'b{i}.fc1', 'q4_11', lsb_out=2.0 ** -F.N_GELU)
        # GELU 출력 Q4.11 → fc2 입력 int8 (스칼라 M 하나)
        M_g, sh_g = F.make_requant(torch.tensor([2.0 ** -F.N_GELU / s_x[f'b{i}.fc2']]))
        layers[f'b{i}.gelu'] = dict(name=f'b{i}.gelu', kind='gelu', M=M_g, shift=sh_g)
        pack_linear(f'b{i}.fc2', 'bf16')

    # ---- 마지막 LayerNorm + head → Q6.9 → softmax ----
    ln = lnm['norm_f']
    g, be = ln.weight.data, ln.bias.data
    if float(g.abs().max()) >= 2.0:
        notes.append(f'norm_f gamma max {float(g.abs().max()):.3f} ≥ 2 → Q1.14 포화')
    M_ln, sh_ln = F.make_requant(torch.tensor([2.0 ** -F.N_LNOUT / s_x['head']]))
    layers['norm_f'] = dict(name='norm_f', kind='layernorm',
                            gamma=F.q_enc(g, F.N_GAMMA), beta=F.q_enc(be, F.N_BETA),
                            M=M_ln, shift=sh_ln, eps=EPS)
    pack_linear('head', 'q6_9', lsb_out=2.0 ** -F.N_SMAX)

    return dict(layers=layers, dims=model.dims, s_x=s_x, obs=obs, notes=notes,
                gelu_lut=F.build_gelu_lut(), exp_lut=F.build_exp_lut(),
                recip_lut=F.build_recip_lut())


# =============================================================================
# 정수 추론 — RTL 이 하는 연산만
# =============================================================================
def _gemm(x_int, w_int):
    """정확한 정수 GEMM. |acc| < 784·127² < 2^24 라 fp64 로 반올림이 없습니다."""
    return torch.round(x_int.double() @ w_int.double().t()).long()


def _linear_acc(e, x_int):
    return _gemm(x_int, e['w_int'].long()) + e['b_int']


def _to_bf16(e, acc):
    """Requant_Bf16 : out = bf16(acc · step_bf16[c])

    **상수는 bf16 으로 저장됩니다.** 하드웨어가 들고 있는 것이 bf16 이므로 골든도
    full precision 이 아니라 bf16 으로 반올림한 step 을 써야 합니다 — 안 그러면
    출력이 1 ULP 씩 어긋납니다(실측 30%).

    곱은 정확합니다: acc 는 2^24 미만, step 가수는 8비트라 곱이 float64 안에서
    반올림 없이 떨어지고, **마지막에 bf16 으로 한 번만** 내립니다
    (`Requant_Bf16.v` 가 정수곱 → 정규화 → RNE 한 번으로 하는 것과 동일).
    """
    step_bf = F.bf16(e['step'].float()).double()
    return F.bf16(acc.double() * step_bf)


def _rsqrt_bf16(v):
    """입력을 먼저 bf16 으로 못박아 LUT 으로 구현 가능하게 (가수 8b + 지수 홀짝)."""
    return F.bf16(torch.rsqrt(F.bf16(v)))


def _fp32_sum(x):
    """k = 0,1,…,E-1 **순차** fp32 누산.

    `torch.sum` / `torch.mean` 은 내부적으로 pairwise·벡터화 순서라 **하드웨어와
    어긋납니다.** RTL 은 fp32 가산기 하나를 k 순서로 돌리므로 골든도 같은 순서여야
    비트 단위로 맞습니다. 이미지 32장(레인)은 병렬이라 루프는 k 축만 돕니다.
    """
    acc = torch.zeros(x.shape[:-1] + (1,), dtype=torch.float32)
    for k in range(x.shape[-1]):
        acc = acc + x[..., k:k + 1].float()                 # IEEE fp32 덧셈
    return acc


def _layernorm(e, x_bf, tap=None):
    """bf16 앞단 + Qm.n 뒷단 (hw_flow.md §2.6). 반환: int8 코드.

    E 가 2의 거듭제곱이라 `/E` 는 지수 감소 하나이고 오차가 없습니다.
    """
    x = F.bf16(x_bf)
    E = x.shape[-1]
    mu = F.bf16(_fp32_sum(x) / E)                           # fp32 누산 → bf16 1회
    ctr = F.bf16(x - mu)
    var = F.bf16(_fp32_sum(ctr * ctr) / E)                  # bf16² 는 fp32 에서 정확
    rstd = _rsqrt_bf16(var + e['eps'])
    xhat = F.q_enc(F.bf16(ctr * rstd), F.N_XHAT)            # Q4.11
    y = F.ln_affine(xhat, e['gamma'], e['beta'])            # Q4.11
    out = F.requant(y, e['M'][0], e['shift'], -128, 127)    # → int8
    if tap is not None:
        tap.update({f"{e['name']}.xhat": xhat, f"{e['name']}.out": y,
                    f"{e['name']}.int8": out})
    return out


@torch.no_grad()
def int_forward(pl, x_float, tap=None):
    """x_float (B,784) in [0,1] → uint8 확률 (B,10)."""
    L = pl['layers']
    nb = pl['dims']['n_block']

    e = L['stem']
    x_int = torch.clamp(torch.round(x_float.double() / e['s_x']), -128, 127).long()
    if tap is not None:
        tap['stem.in'] = x_int
    acc = _linear_acc(e, x_int)
    x_bf = _to_bf16(e, acc)                                  # residual 스트림 시작
    if tap is not None:
        tap['stem.acc'], tap['stem.bf16'] = acc, x_bf

    for i in range(nb):
        a = _layernorm(L[f'b{i}.norm'], x_bf, tap)

        e1 = L[f'b{i}.fc1']
        acc1 = _linear_acc(e1, a)
        q = F.requant(acc1, e1['M'], e1['shift'], F.I16_MIN, F.I16_MAX)   # Q4.11
        g = F.gelu_q(q, pl['gelu_lut'])                                   # Q4.11
        eg = L[f'b{i}.gelu']
        h = F.requant(g, eg['M'][0], eg['shift'], -128, 127)              # int8
        if tap is not None:
            tap.update({f'b{i}.fc1.q411': q, f'b{i}.gelu.q411': g, f'b{i}.gelu.int8': h})

        e2 = L[f'b{i}.fc2']
        h_bf = _to_bf16(e2, _linear_acc(e2, h))
        x_bf = F.bf16(x_bf + h_bf)                                        # residual
        if tap is not None:
            tap[f'b{i}.res.bf16'] = x_bf

    a = _layernorm(L['norm_f'], x_bf, tap)
    eh = L['head']
    z = F.requant(_linear_acc(eh, a), eh['M'], eh['shift'], F.I16_MIN, F.I16_MAX)  # Q6.9
    p = F.softmax_u8(z, pl['exp_lut'], pl['recip_lut'])                   # uint8
    if tap is not None:
        tap['head.q69'], tap['softmax.u8'] = z, p
    return p


@torch.no_grad()
def evaluate_int(pl, loader, device):
    correct = total = 0
    for x, y in loader:
        p = int_forward(pl, x.reshape(x.shape[0], -1).to(device))
        correct += (p.argmax(dim=-1) == y.to(device)).sum().item()
        total += y.numel()
    return correct / total, total


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--calib_batches', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = args.device
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = FFNNet(**ck['dims'])
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    train_loader, test_loader = data_mod.loaders(args.batch_size, workers=0)

    fp32_acc, n = data_mod.accuracy(model, test_loader, device)
    print(f'fp32 accuracy : {fp32_acc*100:.2f}%  (n={n})\n')

    obs = calibrate(model, train_loader, device, args.calib_batches)
    pl = build_payload(model.cpu(), obs)

    # ---- 포맷이 실제로 담기는지 (클리핑 점검) ----
    print(f"{'사이트':<16}{'관측 max|x|':>13}  포맷 / 여유")
    checks = [('GELU 입력', [f'b{i}.gelu.in' for i in range(pl['dims']['n_block'])], 16.0, 'Q4.11'),
              ('GELU 출력', [f'b{i}.gelu.out' for i in range(pl['dims']['n_block'])], 16.0, 'Q4.11'),
              ('LN 출력', [f'b{i}.norm.out' for i in range(pl['dims']['n_block'])] + ['norm_f.out'], 16.0, 'Q4.11'),
              ('LN 입력(bf16)', [f'b{i}.norm.in' for i in range(pl['dims']['n_block'])] + ['norm_f.in'], float('inf'), 'bf16'),
              ('head 출력', ['head.out'], 64.0, 'Q6.9')]
    for label, keys, lim, fmt in checks:
        v = max(obs[k] for k in keys if k in obs)
        mark = 'OK' if v < lim else '** 클리핑 **'
        room = f'{lim/v:.2f}x 여유' if math.isfinite(lim) else '지수 범위'
        print(f'{label:<16}{v:>13.4f}  {fmt}  {room}  {mark}')
    for nn_ in pl['notes']:
        print(f'  [주의] {nn_}')

    print(f"\n{'레이어':<12}{'형상':>14}{'소비자':>8}{'sh':>4}")
    for k, e in pl['layers'].items():
        if e.get('kind') in ('layernorm', 'gelu'):
            print(f"{k:<12}{'-':>14}{e['kind']:>8}{e['shift']:>4}")
        else:
            print(f"{k:<12}{str(e['shape']):>14}{e['consumer']:>8}"
                  f"{('-' if e['shift'] is None else e['shift']):>4}")

    int_acc, n = evaluate_int(pl, test_loader, 'cpu')
    print(f'\nint8+Qm.n accuracy : {int_acc*100:.2f}%  (n={n})')
    print(f'accuracy change    : {(int_acc-fp32_acc)*100:+.3f} pp   '
          f'(1 test sample = {100.0/n:.3f} pp)')

    torch.save(dict(pl, fp32_accuracy=fp32_acc, int_accuracy=int_acc), OUT)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
