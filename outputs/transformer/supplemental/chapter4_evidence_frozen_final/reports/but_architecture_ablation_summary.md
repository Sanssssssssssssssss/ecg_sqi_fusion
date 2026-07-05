# BUT Matched Architecture Ablation Retraining

- Seeds: `20260876, 20260877, 20260878`
- Data: v116 official split; train balanced, val/test original-only.
- Verdict: **partial support: the full Conformer is best and pooled-H is clearly worse, but no-hires is only modestly worse**.
- Figure: `outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M7_but_architecture_ablation.png`

## Summary

| model | n_seeds | acc_mean | macro_f1_mean | good_recall_mean | medium_recall_mean | boundary_error_rate_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Full Conformer | 3 | 0.9441 | 0.9537 | 0.9481 | 0.9246 | 0.0593 |
| No hi-res cross attention | 3 | 0.9371 | 0.9472 | 0.9380 | 0.9209 | 0.0663 |
| Mean-H fusion | 3 | 0.9414 | 0.9501 | 0.9478 | 0.9161 | 0.0613 |
| Pooled-H only | 3 | 0.9275 | 0.9394 | 0.9234 | 0.9193 | 0.0764 |

## Seed-level rows

| seed | model | acc | macro_f1 | good_recall | medium_recall | boundary_error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 20260876 | Full Conformer | 0.9389 | 0.9511 | 0.9316 | 0.9367 | 0.0659 |
| 20260877 | Full Conformer | 0.9454 | 0.9532 | 0.9506 | 0.9241 | 0.0570 |
| 20260878 | Full Conformer | 0.9481 | 0.9567 | 0.9620 | 0.9130 | 0.0552 |
| 20260876 | No hi-res cross attention | 0.9264 | 0.9406 | 0.9193 | 0.9193 | 0.0789 |
| 20260877 | No hi-res cross attention | 0.9400 | 0.9473 | 0.9468 | 0.9146 | 0.0617 |
| 20260878 | No hi-res cross attention | 0.9448 | 0.9537 | 0.9478 | 0.9288 | 0.0582 |
| 20260876 | Mean-H fusion | 0.9400 | 0.9486 | 0.9288 | 0.9446 | 0.0623 |
| 20260877 | Mean-H fusion | 0.9432 | 0.9527 | 0.9639 | 0.8940 | 0.0605 |
| 20260878 | Mean-H fusion | 0.9410 | 0.9488 | 0.9506 | 0.9098 | 0.0611 |
| 20260876 | Pooled-H only | 0.9259 | 0.9390 | 0.9060 | 0.9430 | 0.0783 |
| 20260877 | Pooled-H only | 0.9248 | 0.9379 | 0.9117 | 0.9288 | 0.0795 |
| 20260878 | Pooled-H only | 0.9319 | 0.9413 | 0.9525 | 0.8861 | 0.0712 |
