# EvT integer datapath (ZCU102)

fp32 checkpoint → INT8 weights/activations, INT32 accumulators and biases,
16-bit Qm.n scales, GELU Q4.11, softmax input Q6.9, bf16→Q4.11 LayerNorm, INT8
positional encoding. Nothing here fake-quantizes: every number below is what
the board computes.

`hw_flow.md` documents the datapath unit by unit and why each format was
chosen. This file is how to run it.

## Configuration

There is exactly one, frozen in `quant_lib/hw_quant.FinalConfig`:

| | |
|---|---|
| weights | INT8, per-output-channel scale (one shared exponent per layer) |
| activations | INT8, per-tensor scale |
| accumulators | INT32; bias folded to an INT32 accumulator addend |
| scales | 16-bit Qm.n codes — no fp32 scale anywhere |
| GELU | Q4.11 in and out → one LUT for the whole network |
| softmax | input Q6.9 → one exp LUT; output Q1.14 → uint8, step 1/127 |
| LayerNorm | bf16 front end (input → mean → variance → rsqrt), then Q4.11 `xhat` / output; `gamma` Q1.14, `beta` Q4.11 |
| positional encoding | INT8 table sharing the consumer's input scale (no K-split) |
| classifier head | `argmax(acc·M)` — no log_softmax, no output grid, no shift |
| requantizer | `out = sat((acc·M + (1<<(sh-1))) >> sh)`, `M` int32, `sh` uint6 |

## Pipeline

```bash
conda activate evt_new
cd EventTransformer/quantization

python quantize.py                       # all datasets; --datasets DVS128_10 for one
python export_fpga.py --dataset DVS128_10
python audit.py      --dataset DVS128_10
python plot_accuracy.py
python test_quant.py
```

`quantize.py` does the whole model-side flow per dataset:

1. sanity-check the reference attention forward against `nn.MultiheadAttention`
2. calibrate activation scales on train batches (`quant_lib/calibration.py`)
3. calibrate the non-linear requantization sites on the integer datapath
4. pack every parameter into integer constants
5. save `<dataset>/quantized/model_int8.pt`, reload the model **from that file**,
   and evaluate
6. evaluate the untouched fp32 checkpoint for the baseline

Results land in `results.csv`; `plot_accuracy.py` turns them into
`accuracy_drop.png`.

## Results

fp32 checkpoint → integer datapath, full test sets, `--seed 0`
(`accuracy_drop.png`):

| dataset | fp32 | integer | change | n | 1 sample | parameter bytes |
|---|---|---|---|---|---|---|
| ASL_DVS | 99.9653 % | 99.9504 % | −0.015 pp | 20,157 | 0.005 pp | 2,174 KiB → 565 KiB (3.76×) |
| DVS128_10 | 98.1061 % | 97.3485 % | −0.758 pp | 264 | 0.379 pp | 1,927 KiB → 516 KiB (3.73×) |
| DVS128_11 | 95.8333 % | 96.1806 % | **+0.347 pp** | 288 | 0.347 pp | 1,927 KiB → 516 KiB (3.73×) |

Read the last column before the third: the two DVS128 test sets are so small
that **one sample is 0.38 pp / 0.35 pp**. DVS128_10 loses exactly two samples
and DVS128_11 gains exactly one — neither is a measurement of anything finer
than that. ASL_DVS is the only run with real resolution, and it loses 3 samples
out of 20,157.

### Reproducibility

The event pipeline augments inside `__getitem__` with `np.random` (random time
window, crop, flip, token drop) and PyTorch does not seed numpy in dataloader
workers, so **which batches calibration sees is part of the result**.
`data_utils.seed_dataloader` pins it; re-running `quantize.py --datasets X`
twice now gives the same number, and `--seed` changes it deliberately.

Before that was fixed, repeated runs of DVS128_10 landed on −0.379 pp and
−0.758 pp — one test sample apart. Treat sub-sample differences on the DVS128
sets as calibration noise, not as signal.

### Integer formula vs. the simulator

`export_fpga.py` re-computes each layer with the pure-integer expression
(`sat((acc·M + (1<<(sh-1))) >> sh)`) and compares against the datapath the
accuracy above came from. It covers **all 22 GEMMs**, including the six inside
the attention blocks that no module hook can see: **100.00 % within 1 LSB** on
all three datasets, worst case 1 LSB of the consumer's format. The six layers
feeding the bf16 LayerNorm match **exactly (0.00 ulp)**.

`audit.py` confirms every MAC operand is an exact INT8 integer, every
accumulator an exact integer, and every non-linear input lands on its declared
grid, with no saturating site.

## Files

| file | what it does |
|---|---|
| `quantize.py` | the pipeline above — calibrate, pack, save, evaluate |
| `export_fpga.py` | packed `.pt` → RTL constants (`M`, shift, W, B `.bin` + `manifest.json`), and checks the integer formula against the simulator |
| `audit.py` | instruments a real forward: are all MAC operands INT8 integers, all accumulators exact, every non-linear input on its declared grid, anything saturating |
| `plot_accuracy.py` | `results.csv` → `accuracy_drop.png` |
| `test_quant.py` | unit checks for the Qm.n primitives and INT32 bias folding |
| `quant_lib/hw_quant.py` | the datapath: pack a model into integer constants and run it |
| `quant_lib/calibration.py` | activation-scale calibration + the reference attention forward |
| `quant_lib/fixed_point.py` | Qm.n primitives (format choice, pack/unpack) |
| `quant_lib/data_utils.py` | checkpoint / dataloader / accuracy helpers |
| `hw_flow.md` | what the hardware computes, unit by unit |

## Per-dataset layout

```
<dataset>/
  all_params.json                model + data config
  weights/                       fp32 training checkpoints (input)
  quantized/model_int8.pt        packed integer checkpoint (output)
  datapath_audit.csv             per-site audit detail
fpga_export/<dataset>/           .bin constants + manifest.json + verification.csv
```
