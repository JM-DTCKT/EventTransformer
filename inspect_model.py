import json
import sys
import os

sys.path.append('/hai/home/sgh/01_assignment/EventTransformer')

from models.EvT import EvNetBackbone, CLFBlock
from torch import nn
import torch

# ── 1. config 로드 ──────────────────────────────────────────
with open('/hai/home/sgh/01_assignment/EventTransformer/pretrained_models/ASL_DVS_dwn/all_params.json') as f:
    params = json.load(f)

backbone_params = params['backbone_params']
clf_params      = params['clf_params']

# ── 2. 모델 생성 ─────────────────────────────────────────────
backbone = EvNetBackbone(**backbone_params)
clf      = CLFBlock(ipt_dim=backbone_params['embed_dim'], **clf_params)

model = nn.ModuleDict({'backbone': backbone, 'clf': clf})

# ── 3. 레이어 구조 출력 ──────────────────────────────────────
print("=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(model)

# ── 4. 파라미터 수 계산 ──────────────────────────────────────
def count_params(module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

print("\n" + "=" * 60)
print("PARAMETER COUNT")
print("=" * 60)

for name, module in model.items():
    total, trainable = count_params(module)
    print(f"\n[{name}]")
    print(f"  Total      : {total:>10,}")
    print(f"  Trainable  : {trainable:>10,}")
    print(f"  Frozen     : {total - trainable:>10,}")

    # 서브모듈별 파라미터 수
    print(f"\n  --- Sub-modules ---")
    for sub_name, sub_module in module.named_children():
        t, tr = count_params(sub_module)
        print(f"  {sub_name:<30} total={t:>8,}  trainable={tr:>8,}")

total_all, trainable_all = count_params(model)
print("\n" + "=" * 60)
print(f"  TOTAL ALL      : {total_all:>10,}")
print(f"  TOTAL TRAINABLE: {trainable_all:>10,}")
print("=" * 60)

# ── 5. 각 레이어별 shape 상세 출력 ───────────────────────────
print("\n" + "=" * 60)
print("LAYER-WISE PARAMETER SHAPES")
print("=" * 60)
for name, param in model.named_parameters():
    print(f"  {name:<55} {str(list(param.shape)):<20}  {param.numel():>8,}")