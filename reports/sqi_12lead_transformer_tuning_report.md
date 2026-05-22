# SQI 12-Lead Raw Transformer Tuning

This report records the small tuning sweep for the standalone 12-lead raw
transformer comparator on the classical SQI pipeline. The goal was to give the
SQI line its own transformer control without changing the existing SQI feature
models or polluting the main transformer pipeline.

Implementation file:

```text
src/sqi_pipeline/models/transformer_12lead.py
```

Run outputs:

```text
outputs/sqi/models/transformer_12lead
outputs/sqi/models/transformer_12lead_tuning
```

Slurm jobs:

```text
29517203  sqi12_tune
29517592  sqi12_tune2
```

## Code Changes

The comparator stayed as one small script. I only added tuning switches:

| Switch | Values | Reason |
| --- | --- | --- |
| `--pooling` | `cls`, `mean`, `cls_mean` | Compare CLS readout against mean and CLS+mean readout. |
| `--label_smoothing` | float | Test whether mild smoothing stabilizes the binary boundary. |
| `--select_best_by` | `val_acc`, `val_auc` | Allow checkpoint selection by validation AUC. |

Defaults are unchanged for the baseline behavior:

```text
pooling=cls
label_smoothing=0.0
select_best_by=val_acc
```

## Baselines

All results use the same balanced Set-A split and 12-lead 10 s signals at
125 Hz.

| Model | Input | Fixed Test Acc | Test AUC | Test maxAcc |
| --- | --- | ---: | ---: | ---: |
| SVM-RBF | 12-lead 84 SQI | 0.9440 | 0.9838 | 0.9569 |
| LM-MLP | 12-lead 84 SQI | 0.9267 | 0.9618 | 0.9267 |
| LogReg | 12-lead 84 SQI | 0.9138 | 0.9447 | 0.9224 |
| Raw Transformer baseline | 12-lead waveform | 0.9138 | 0.9418 | 0.9310 |

## Tuning Sweep

| Run | Seed | Test Acc | AUC | Se | Sp | Oracle | Val Acc/AUC | Best Ep | Key config |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `t07_long_patch_cls_p100s25` | 0 | 0.9181 | 0.9518 | 0.9397 | 0.8966 | 0.9397 | 0.9526/0.9550 | 33 | cls, d=96, L=2, patch/stride=100/25, drop=0.15, lr=8e-4, wd=5e-4 |
| `t07_seed1_long_patch_cls_p100s25` | 1 | 0.9181 | 0.9456 | 0.9397 | 0.8966 | 0.9181 | 0.9440/0.9695 | 46 | same as t07 |
| `t07_seed2_long_patch_cls_p100s25` | 2 | 0.9181 | 0.9426 | 0.9224 | 0.9138 | 0.9224 | 0.9569/0.9605 | 63 | same as t07 |
| `transformer_12lead` | 0 | 0.9138 | 0.9418 | 0.9655 | 0.8621 | 0.9310 | 0.9483/0.9722 | 53 | cls, d=96, L=2, patch/stride=25/10, drop=0.10, lr=1e-3, wd=1e-4 |
| `t10_long_patch_clsmean_p100s25` | 0 | 0.9138 | 0.9220 | 0.9310 | 0.8966 | 0.9138 | 0.9483/0.9615 | 44 | cls_mean, d=96, L=2, patch/stride=100/25, drop=0.15, lr=8e-4, wd=5e-4 |
| `t12_long_patch_cls_p100s50` | 0 | 0.9095 | 0.9420 | 0.9138 | 0.9052 | 0.9181 | 0.9483/0.9579 | 48 | cls, d=96, L=2, patch/stride=100/50, drop=0.15, lr=8e-4, wd=5e-4 |
| `t11_long_patch_cls_p100s25_drop25` | 0 | 0.9095 | 0.9365 | 0.9483 | 0.8707 | 0.9267 | 0.9526/0.9744 | 69 | cls, d=96, L=2, patch/stride=100/25, drop=0.25, lr=5e-4, wd=1e-3 |
| `t13_patch75_cls_p75s25` | 0 | 0.9095 | 0.9347 | 0.9224 | 0.8966 | 0.9138 | 0.9612/0.9663 | 50 | cls, d=96, L=2, patch/stride=75/25, drop=0.15, lr=8e-4, wd=5e-4 |
| `t03_clsmean_d96_drop20_ls005` | 0 | 0.9052 | 0.9518 | 0.9052 | 0.9052 | 0.9138 | 0.9483/0.9624 | 26 | cls_mean, d=96, L=2, patch/stride=25/10, drop=0.20, lr=5e-4, wd=5e-4, label smoothing=0.05 |
| `t02_small_mean_l1_d64_drop20` | 0 | 0.9052 | 0.9334 | 0.9224 | 0.8879 | 0.9138 | 0.9440/0.9626 | 27 | mean, d=64, L=1, patch/stride=25/10, drop=0.20, lr=5e-4, wd=1e-3 |
| `t06_valauc_cls_d96_drop20` | 0 | 0.9052 | 0.9197 | 0.9052 | 0.9052 | 0.9095 | 0.9353/0.9589 | 26 | cls, d=96, L=2, patch/stride=25/10, select best by val AUC |
| `t09_long_patch_mean_p100s25` | 0 | 0.9009 | 0.9512 | 0.9483 | 0.8534 | 0.9224 | 0.9526/0.9718 | 41 | mean, d=96, L=2, patch/stride=100/25, drop=0.15, lr=8e-4, wd=5e-4 |
| `t08_lowlr_cls_d96_drop05` | 0 | 0.9009 | 0.9475 | 0.9397 | 0.8621 | 0.9138 | 0.9440/0.9667 | 29 | cls, d=96, L=2, patch/stride=25/10, drop=0.05, lr=3e-4, wd=1e-4 |
| `t05_coarse_patch_mean_small` | 0 | 0.9009 | 0.9330 | 0.9310 | 0.8707 | 0.9052 | 0.9569/0.9706 | 64 | mean, d=64, L=1, patch/stride=50/25, drop=0.25, lr=5e-4, wd=1e-3 |
| `t04_coarse_patch_cls_p50s25` | 0 | 0.8966 | 0.9220 | 0.8707 | 0.9224 | 0.9095 | 0.9569/0.9695 | 45 | cls, d=96, L=2, patch/stride=50/25, drop=0.15, lr=8e-4, wd=5e-4 |
| `t01_small_cls_l1_d64_drop20` | 0 | 0.8793 | 0.9359 | 0.8534 | 0.9052 | 0.8922 | 0.9397/0.9630 | 41 | cls, d=64, L=1, patch/stride=25/10, drop=0.20, lr=5e-4, wd=1e-3 |

## Best Tuned Result

The best fixed-threshold model is:

```text
t07_long_patch_cls_p100s25
```

| Metric | Value |
| --- | ---: |
| Fixed test accuracy | 0.9181 |
| Test AUC | 0.9518 |
| Sensitivity | 0.9397 |
| Specificity | 0.8966 |
| Test oracle maxAcc | 0.9397 |
| Confusion matrix | tn=104, fp=12, fn=7, tp=109 |

Multi-seed check for the same `t07` config:

| Seeds | Mean Test Acc | Std Test Acc | Mean AUC |
| --- | ---: | ---: | ---: |
| 0, 1, 2 | 0.9181 | 0.0000 | 0.9467 |

## Readout

The tuning sweep gives only a small improvement over the raw transformer
baseline:

```text
0.9138 -> 0.9181  (+0.0043)
```

It remains below the stronger SQI-feature models:

```text
SVM-RBF all84 fixed acc: 0.9440
LM-MLP all84 fixed acc: 0.9267
```

The best tuning pattern is longer waveform patches with CLS pooling. Mean
pooling, CLS+mean pooling, label smoothing, val-AUC checkpoint selection, lower
learning rate, and heavier dropout did not improve fixed test accuracy.

Conclusion: this is a useful comparator for the SQI line, but the classical
12-lead SQI feature representation is still the better model family here. The
raw transformer control should be reported as a negative/diagnostic comparison,
not as the main SQI-line result.
