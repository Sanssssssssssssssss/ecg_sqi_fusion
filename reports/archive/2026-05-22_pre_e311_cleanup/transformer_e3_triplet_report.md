# E3 Triplet Baseline

Date: 2026-05-10

## Purpose

E3 is the simple triplet baseline that E3.5 keeps as its structural template.
Each training clean segment is augmented into good/medium/bad variants using
the same clean ECG and same sampled noise, with labels determined by global SNR
bins.

## Best Run

Artifact:

`outputs/transformer_e3_triplet_k1`

Model:

`outputs/transformer_e3_triplet_k1/models/e3_triplet_tune09`

Result:

| Metric | Value |
| --- | ---: |
| test accuracy | 0.9405 |
| best validation accuracy | 0.9590 |
| medium test recall | 0.9192 |
| best epoch | 18 |

Test confusion matrix, rows=true and cols=pred:

```text
[[917, 34, 2],
 [54, 876, 23],
 [7, 50, 895]]
```

## Recipe

- epochs: `26`
- batch size: `32`
- dropout: `0.05`
- input: raw ECG
- pooling: decoder mean
- no ordinal head
- no SNR regression head
- schedule: `e_cls=0`, `e_denoise=18`, `e_level=4`, `e_uncert=4`
- `lambda_cls=22`, `lambda_den=25`, `lambda_lvl=1`
- label smoothing: `0.02`
- medium class weight: `1.15`
- bad denoise max weight: `0.02`
- best checkpoint selected by validation accuracy

## Readout

E3 improved the earlier transformer benchmark by making the training data more
structured, but its class labels are still recoverable from measured global SNR.
The follow-up E3.5 experiment keeps the same compact triplet structure while
replacing SNR-bin labels with morphology-damage labels.
