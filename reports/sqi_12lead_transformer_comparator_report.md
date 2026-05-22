# SQI-Line 12-Lead Raw Transformer Comparator

This is a small standalone comparator for the classical SQI line. It does not
modify the existing SQI feature models or the current transformer-line code.

Implementation:

```text
src/sqi_pipeline/models/transformer_12lead.py
```

Local outputs:

```text
outputs/sqi/models/transformer_12lead
```

The output directory contains:

```text
best_model.pt
history.csv
metrics.json
predictions.csv
summary.md
```

## Setup

Input data:

```text
outputs/sqi/splits/split_seta_seed0_balanced.csv
outputs/sqi/resampled_125/*.npz
```

Task:

```text
acceptable(+1) vs unacceptable(-1)
```

Data size:

| Split | Acceptable | Unacceptable | Total |
| --- | ---: | ---: | ---: |
| train | 541 | 541 | 1082 |
| val | 116 | 116 | 232 |
| test | 116 | 116 | 232 |

Model:

- 12-lead raw waveform input, shape `1250 x 12`
- per-lead normalization from train split only
- Conv1d patch embedding
- learnable CLS token
- 2-layer Transformer encoder
- binary classifier

Training:

| Parameter | Value |
| --- | ---: |
| seed | 0 |
| epochs max | 80 |
| patience | 14 |
| batch size | 64 |
| lr | 0.001 |
| weight decay | 0.0001 |
| d_model | 96 |
| heads | 4 |
| layers | 2 |
| dropout | 0.10 |
| best epoch | 53 |
| train time | 21.8 s |

## Result

| Evaluation | Acc | Se | Sp | AUC | Threshold | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed threshold | 0.9138 | 0.9655 | 0.8621 | 0.9418 | 0.5000 | tn=100, fp=16, fn=4, tp=112 |
| val-selected threshold | 0.8922 | 0.9828 | 0.8017 | 0.9418 | 0.0915 | tn=93, fp=23, fn=2, tp=114 |
| test oracle maxAcc | 0.9310 | 0.9655 | 0.8966 | 0.9418 | 0.8890 | tn=104, fp=12, fn=4, tp=112 |

Use the fixed-threshold row as the clean model comparison. The test oracle
maxAcc row is included only because the SQI tables also report maxAcc-style
thresholds.

## SQI-Line Comparison

| Model | Input | Fixed Test Acc | Test AUC | Test maxAcc |
| --- | --- | ---: | ---: | ---: |
| SVM-RBF | 12-lead 84 SQI | 0.9440 | 0.9838 | 0.9569 |
| LM-MLP | 12-lead 84 SQI | 0.9267 | 0.9618 | 0.9267 |
| LogReg | 12-lead 84 SQI | 0.9138 | 0.9447 | 0.9224 |
| Raw Transformer | 12-lead waveform | 0.9138 | 0.9418 | 0.9310 |

## Readout

On the classical SQI-line binary task, the small raw 12-lead transformer does
not beat the best 12-lead SQI feature model. It matches LogReg at fixed
threshold and trails SVM-RBF.

This is a useful negative control: the classical SQI dataset is already well
served by engineered 12-lead SQI summaries, while the newer E3.9 transformer
line is useful for a different claim: local waveform evidence can beat global
summary SQI on synthetic morphology-damage benchmarks.
