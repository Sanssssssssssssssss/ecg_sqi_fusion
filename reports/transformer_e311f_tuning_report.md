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

Best E3.11f tuning result: `R1 cls-only SNR 0.05` = `0.9376`
Best completed result including references: `D1 reference` = `0.9465`

## Interpretation

- Best E3.11f visual tuning is `R1 cls-only SNR 0.05` at `0.9376`, improving the E3.11f baseline `0.9329` but not passing the `0.94` target.
- Round 2 did not improve the first-round best: low-LR continuation dropped to `0.9303`, label smoothing was nearly tied at `0.9372`, and good/medium weighting dropped to `0.9290`.
- The best E3.11f recipe is: D1 warm-start, `cls_pool=cls`, raw input, `snr_head`, `lambda_snr=0.05`, `lr=3e-5`, `epochs=24`, `dropout=0.10`, `weight_decay=0.03`, no rank/local/SQI-teacher/noise-type head, and no denoise/level losses.
- Compared with the references, E3.11f R1 is much better than the current E3.11 result (`0.9000`) but remains below E3.10 M2 (`0.9402`) and D1 (`0.9465`).
- For cls-only rows, denoise outputs are not trained; use accuracy/recall as the classification evidence and treat denoise SNR values as non-decision diagnostics.
- Recommendation: keep E3.10 M2 as the safer visual benchmark if the requirement is `>=0.94`; keep E3.11f R1 as the best E3.11-style diagnostic result.

## Image Folders

- Images only: `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/<variant>/<class>/<noise_kind>/`
- No HTML/CSV/JSON is required for the visual check.
