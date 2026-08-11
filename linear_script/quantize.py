"""fp32 MLP -> 정수 데이터패스. RTL 과 **한 줄씩 대응**하도록 쓴 골든 모델.

    python quantize.py

각 Linear 는 RTL 세 블록에 그대로 매핑됩니다:

    acc[c]  = Σ_k x_int[k]*w_int[c,k] + b_int[c]      Mac_OS/Linear_Top  (+ b 는 requant 프리애더)
    out[c]  = sat((acc[c]*M[c] + 2^(sh-1)) >> sh)     Requant/Requant_Int
    ReLU                                              requant 의 UNSIGNED_OUT 로 융합

상수 만드는 법 (전부 오프라인):
    s_w[c] = max|W[c,:]| / 127                weight scale, 출력 채널별
    s_x    = max|x| / 127                     activation scale, 캘리브레이션
    b_int  = round(b / (s_x*s_w[c]))          INT32 accumulator addend
    ratio  = s_x*s_w[c] / lsb_out             lsb_out = 다음 층 입력 step (마지막은 1)
    sh     = |M| < 2^31 을 만족하는 최대값     M[c] = round(ratio[c] * 2^sh)

은닉층은 소비자가 ReLU 이므로 **ReLU 를 requant 에 융합**합니다
(하한을 -128 대신 0 으로 클램프 → [0,127]). `04_basic_rtl/ReLU` §3 에서 두 경로가
비트 단위로 같음을 증명했고, 여기서도 분리 경로와 대조해 확인합니다.

마지막 층은 **argmax(acc*M)** 로 끝냅니다. softmax 는 단조증가라 순서를 안 바꾸고,
`s_w[c]` 가 클래스마다 달라 `M[c]` 는 반드시 곱해야 합니다.
"""

import argparse
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data as data_mod                     # noqa: E402
from model import MLP                       # noqa: E402

CKPT = os.path.join(SCRIPT_DIR, 'mlp_fp32.pt')
OUT = os.path.join(SCRIPT_DIR, 'mlp_int8.pt')
INT8_QMAX, INT32_MAX = 127, 2 ** 31 - 1


# =============================================================================
# 상수 만들기
# =============================================================================
def make_requant(ratio, m_bits=32):
    """ratio[c] -> (M[c] int64 tensor, sh int).  |M| < 2^(m_bits-1) 를 만족하는
    최대 sh 를 고른다 (곱수의 유효비트를 꽉 채워 반올림 오차를 없앰)."""
    ratio = ratio.double()
    lim = float(2 ** (m_bits - 1) - 1)
    sh = int(torch.floor(torch.log2(torch.tensor(lim / float(ratio.abs().max())))).item())
    while True:
        M = torch.round(ratio * (2.0 ** sh))
        if float(M.abs().max()) <= lim:
            break
        sh -= 1
    assert 0 <= sh < 64, sh
    return M.long(), sh


def _gemm(x_int, w_int):
    """x_int @ w_int^T 를 **정확한 정수**로. CUDA 는 int64 matmul 을 지원하지 않아
    fp64 로 돌리는데, |acc| < 127*127*784 + bias < 2^24 이고 fp64 가수는 52비트라
    반올림이 전혀 없습니다 (RTL 의 INT32 누산기와 비트 동일)."""
    return torch.round(x_int.double() @ w_int.double().t()).long()


def requantize(acc, M, sh, lo, hi):
    """RTL 과 같은 식: sat((acc*M + 2^(sh-1)) >> sh).  round-half-up + 산술 시프트."""
    prod = acc.long() * M.long()
    r = (1 << (sh - 1)) if sh > 0 else 0
    return torch.clamp((prod + r) >> sh, lo, hi)


@torch.no_grad()
def calibrate(model, loader, device, n_batches):
    """각 Linear 입력의 max|x| 를 관측. augmentation 이 없어 결정론적."""
    model.eval().to(device)
    obs = {name: 0.0 for name, _ in model.linears}
    handles = []
    for name, m in model.linears:
        def hook(_m, args, _n=name):
            obs[_n] = max(obs[_n], float(args[0].detach().abs().max()))
        handles.append(m.register_forward_pre_hook(hook))
    seen = 0
    for x, _ in loader:
        if seen >= n_batches:
            break
        model(x.to(device))
        seen += 1
    for h in handles:
        h.remove()
    return obs, seen


@torch.no_grad()
def build_payload(model, act_max):
    """fp32 모델 -> 정수 상수. RTL 이 읽을 것 전부."""
    layers, names = [], [n for n, _ in model.linears]
    for i, (name, m) in enumerate(model.linears):
        last = (i == len(names) - 1)
        W, b = m.weight.data.double(), m.bias.data.double()

        s_w = (W.abs().amax(dim=1) / INT8_QMAX).clamp(min=1e-12)          # (E_out,)
        w_int = torch.clamp(torch.round(W / s_w[:, None]), -128, 127)
        s_x = max(act_max[name], 1e-12) / INT8_QMAX
        step = s_x * s_w                                                   # 누산기 LSB
        b_int = torch.clamp(torch.round(b / step), -INT32_MAX, INT32_MAX)

        # 소비자: 은닉층 -> 다음 Linear 의 int8 step, 마지막 -> argmax (lsb=1)
        lsb_out = 1.0 if last else max(act_max[names[i + 1]], 1e-12) / INT8_QMAX
        M, sh = make_requant(step / lsb_out)

        layers.append(dict(
            name=name, shape=tuple(W.shape),
            w_int=w_int.to(torch.int8), b_int=b_int.long(), M=M, shift=sh,
            s_x=s_x, s_w=s_w, out_step=(None if last else lsb_out),
            relu=(not last),          # 은닉층은 ReLU 를 requant 에 융합
        ))
    return dict(layers=layers, dims=model.dims)


# =============================================================================
# 정수 추론 — RTL 이 하는 것과 같은 연산만
# =============================================================================
@torch.no_grad()
def int_forward(payload, x_float, keep_trace=False):
    """x_float (B,784) in [0,1] -> 클래스 인덱스.  trace 는 골든 벡터용."""
    trace = []
    L = payload['layers']
    x_int = torch.clamp(torch.round(x_float.double() / L[0]['s_x']), -128, 127)

    for i, e in enumerate(L):
        acc = _gemm(x_int, e['w_int']) + e['b_int']                        # INT32
        last = (i == len(L) - 1)
        if last:
            scored = acc * e['M']                                          # argmax 용
            out = None
        else:
            lo, hi = (0, 127) if e['relu'] else (-128, 127)                # ReLU 융합
            out = requantize(acc, e['M'], e['shift'], lo, hi)
            scored = None
        if keep_trace:
            trace.append(dict(x_int=x_int.clone(), acc=acc.clone(),
                              out=(None if out is None else out.clone()),
                              scored=(None if scored is None else scored.clone())))
        if last:
            return scored.argmax(dim=-1), trace
        x_int = out
    raise RuntimeError('unreachable')


@torch.no_grad()
def int_forward_split_relu(payload, x_float):
    """검증용: requant(signed) 후 별도 ReLU. 융합 경로와 같아야 한다."""
    L = payload['layers']
    x_int = torch.clamp(torch.round(x_float.double() / L[0]['s_x']), -128, 127)
    for i, e in enumerate(L):
        acc = _gemm(x_int, e['w_int']) + e['b_int']
        if i == len(L) - 1:
            return (acc * e['M']).argmax(dim=-1)
        out = requantize(acc, e['M'], e['shift'], -128, 127)               # signed 포화
        x_int = torch.clamp(out, min=0)                                    # 그 다음 ReLU
    raise RuntimeError('unreachable')


@torch.no_grad()
def evaluate_int(payload, loader, device):
    correct = total = 0
    for x, y in loader:
        pred, _ = int_forward(payload, x.reshape(x.shape[0], -1).to(device))
        correct += (pred == y.to(device)).sum().item()
        total += y.numel()
    return correct / total, total


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--calib_batches', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = MLP(ck['dims'])
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    train_loader, test_loader = data_mod.loaders(args.batch_size)

    fp32_acc, n_test = data_mod.accuracy(model, test_loader, device)
    print(f'fp32 accuracy      : {fp32_acc * 100:.2f}%  (n={n_test})')

    act_max, seen = calibrate(model, train_loader, device, args.calib_batches)
    print(f'calibration        : {seen} train batches')
    for k, v in act_max.items():
        print(f'    {k:>6s}.in  max|x| = {v:10.5f}   step = {v / 127:.8f}')

    payload = build_payload(model.cpu(), act_max)
    for e in payload['layers']:
        print(f"    {e['name']:>6s}  {str(e['shape']):>12s}  sh={e['shift']:2d}  "
              f"|M|max={int(e['M'].abs().max()):>10d}  |b_int|max={int(e['b_int'].abs().max()):>6d}"
              f"{'   ReLU fused' if e['relu'] else '   -> argmax'}")

    payload = {k: (v if k != 'layers' else
                   [{kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in e.items()}
                    for e in v]) for k, v in payload.items()}
    int_acc, n = evaluate_int(payload, test_loader, device)
    print(f'\nint8 accuracy      : {int_acc * 100:.2f}%  (n={n})')
    print(f'accuracy change    : {(int_acc - fp32_acc) * 100:+.3f} pp   '
          f'(1 test sample = {100.0 / n:.3f} pp)')

    # ReLU 융합이 분리 경로와 정말 같은지 (RTL 의 tb_relu_fused 와 같은 주장)
    same = tot = 0
    for x, _ in test_loader:
        xf = x.reshape(x.shape[0], -1).to(device)
        a, _ = int_forward(payload, xf)
        b = int_forward_split_relu(payload, xf)
        same += (a == b).sum().item(); tot += a.numel()
    print(f'ReLU 융합 == 분리   : {same}/{tot} 일치 ({same / tot * 100:.4f}%)')

    torch.save({k: (v if k != 'layers' else
                    [{kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in e.items()}
                     for e in v]) for k, v in payload.items()} |
               dict(fp32_accuracy=fp32_acc, int8_accuracy=int_acc, n_test=n,
                    act_max=act_max), OUT)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
