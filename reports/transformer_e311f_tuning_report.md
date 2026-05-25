# E3.11f Visual Transformer Tuning

Dataset: `e311f_lite_e310_morph`.
Goal: improve the best E3.11-style visual version without changing data or model structure.

## Baselines

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| D1 reference | 0.9465 | 0.9153 | 0.9465 | 0.9777 | 0.013/-0.710/-1.042 | `[[616, 53, 4], [22, 637, 14], [2, 13, 658]]` |
| E3.10 M2 best | 0.9402 | 0.9491 | 0.9135 | 0.9580 | 0.584/1.190/1.689 | `[[746, 35, 5], [48, 718, 20], [7, 26, 753]]` |
| E3.11 current best | 0.9000 | 0.9017 | 0.8629 | 0.9353 | -2.043/0.954/1.995 | `[[697, 65, 11], [61, 667, 45], [5, 45, 723]]` |
| E3.11f baseline | 0.9329 | 0.9174 | 0.9187 | 0.9626 | -0.378/0.777/1.117 | `[[711, 61, 3], [51, 712, 12], [7, 22, 746]]` |

## Round 1

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| R1 cls-only SNR 0.05 | 0.9376 | 0.9277 | 0.9265 | 0.9587 | -9.978/-6.767/-5.962 | `[[719, 54, 2], [52, 718, 5], [11, 21, 743]]` |
| R1 cls-only SNR 0.10 | 0.9359 | 0.9303 | 0.9161 | 0.9613 | -9.293/-6.129/-5.340 | `[[721, 50, 4], [56, 710, 9], [9, 21, 745]]` |
| R1 delayed light denoise | 0.9333 | 0.9342 | 0.9084 | 0.9574 | -2.152/-0.519/0.211 | `[[724, 48, 3], [62, 704, 9], [10, 23, 742]]` |
| R1 noise-type aux | 0.9320 | 0.9135 | 0.9213 | 0.9613 | -0.472/0.704/1.286 | `[[708, 63, 4], [54, 714, 7], [7, 23, 745]]` |

Round-1 best warm-start for round 2: `e311f_lite_e310_morph_r1_cls_only_snr005`

## Round 2

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| R2 low-lr continue | 0.9303 | 0.9174 | 0.9148 | 0.9587 | -10.680/-7.489/-6.409 | `[[711, 60, 4], [55, 709, 11], [7, 25, 743]]` |
| R2 label smoothing | 0.9372 | 0.9290 | 0.9213 | 0.9613 | -7.994/-5.076/-3.532 | `[[720, 52, 3], [50, 714, 11], [7, 23, 745]]` |
| R2 good/medium weights | 0.9290 | 0.9148 | 0.9135 | 0.9587 | -10.812/-7.598/-6.519 | `[[709, 62, 4], [56, 708, 11], [7, 25, 743]]` |

## Round 3

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| R3 lr 2.5e-5 | 0.9342 | 0.9342 | 0.9084 | 0.9600 | -9.471/-6.291/-5.564 | `[[724, 47, 4], [63, 704, 8], [10, 21, 744]]` |
| R3 lr 4e-5 | 0.9376 | 0.9290 | 0.9226 | 0.9613 | -10.912/-7.594/-6.633 | `[[720, 53, 2], [52, 715, 8], [8, 22, 745]]` |
| R3 dropout 0.15 | 0.9338 | 0.9342 | 0.9032 | 0.9639 | -10.062/-6.732/-6.108 | `[[724, 48, 3], [67, 700, 8], [10, 18, 747]]` |
| R3 weight decay 0.05 | 0.9346 | 0.9342 | 0.9110 | 0.9587 | -9.765/-6.553/-5.757 | `[[724, 47, 4], [60, 706, 9], [10, 22, 743]]` |
| R3 batch size 64 | 0.9329 | 0.9187 | 0.9213 | 0.9587 | -8.922/-5.885/-5.082 | `[[712, 61, 2], [53, 714, 8], [10, 22, 743]]` |
| R3 SNR lambda 0.02 | 0.9351 | 0.9226 | 0.9226 | 0.9600 | -10.066/-6.863/-6.075 | `[[715, 56, 4], [51, 715, 9], [8, 23, 744]]` |
| R3 SNR lambda 0.075 | 0.9368 | 0.9290 | 0.9187 | 0.9626 | -9.517/-6.328/-5.538 | `[[720, 52, 3], [54, 712, 9], [9, 20, 746]]` |

## Round 4

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| R4 seed 1 best recipe | 0.9312 | 0.9226 | 0.9123 | 0.9587 | -9.521/-6.371/-5.630 | `[[715, 56, 4], [53, 707, 15], [9, 23, 743]]` |
| R4 seed 2 best recipe | 0.9342 | 0.9265 | 0.9135 | 0.9626 | -8.701/-5.673/-5.037 | `[[718, 54, 3], [57, 708, 10], [8, 21, 746]]` |
| R4 seed 3 best recipe | 0.9368 | 0.9252 | 0.9226 | 0.9626 | -8.915/-5.964/-5.113 | `[[717, 56, 2], [46, 715, 14], [9, 20, 746]]` |
| R4 dropout 0.05 | 0.9320 | 0.9252 | 0.9110 | 0.9600 | -10.001/-6.938/-6.144 | `[[717, 53, 5], [60, 706, 9], [10, 21, 744]]` |
| R4 label smoothing 0.01 | 0.9351 | 0.9329 | 0.9123 | 0.9600 | -8.569/-5.528/-4.413 | `[[723, 47, 5], [61, 707, 7], [10, 21, 744]]` |
| R4 good weight 1.08 | 0.9363 | 0.9316 | 0.9161 | 0.9613 | -9.668/-6.443/-5.644 | `[[722, 50, 3], [57, 710, 8], [9, 21, 745]]` |
| R4 medium weight 1.08 | 0.9338 | 0.9329 | 0.9097 | 0.9587 | -10.011/-6.799/-6.016 | `[[723, 47, 5], [61, 705, 9], [10, 22, 743]]` |
| R4 select by val loss | 0.9222 | 0.9303 | 0.8916 | 0.9445 | -7.507/-4.713/-3.735 | `[[721, 50, 4], [72, 691, 12], [15, 28, 732]]` |

Best E3.11f tuning result: `R1 cls-only SNR 0.05` = `0.9376`
Best completed result including references: `D1 reference` = `0.9465`

## Interpretation

- Best E3.11f visual tuning so far is `R1 cls-only SNR 0.05` at `0.9376`.
- The tuning target for replacing E3.10 as the visual benchmark is `>=0.94` test accuracy.
- Round 2 did not improve the first-round best: low-LR continuation dropped to `0.9303`, label smoothing was nearly tied at `0.9372`, and good/medium weighting dropped to `0.9290`.
- Round 3 did not improve either: LR, dropout, weight decay, batch size, and SNR-head weight all stayed at or below `0.9376`.
- Round 4 also stayed below target: seeds 1/2/3 reached `0.9312/0.9342/0.9368`, and light boundary tuning moved good/medium recall around without increasing total accuracy.
- The current best recipe family is: D1 warm-start, `cls_pool=cls`, raw input, `snr_head`, `lambda_snr` near `0.05`, `lr` near `3e-5`, no rank/local/SQI-teacher/noise-type head, and no denoise/level losses.
- Compared with the references, E3.11f R1 is much better than the current E3.11 result (`0.9000`) but remains below E3.10 M2 (`0.9402`) and D1 (`0.9465`).
- For cls-only rows, denoise outputs are not trained; use accuracy/recall as the classification evidence and treat denoise SNR values as non-decision diagnostics.
- Final recommendation: keep E3.10 M2 as the safer `>=0.94` visual benchmark and keep E3.11f as a diagnostic result showing the limit of this cleaner visual-label variant under simple transformer tuning.

## Image Folders

- Images only: `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/<variant>/<class>/<noise_kind>/`
- No HTML/CSV/JSON is required for the visual check.
