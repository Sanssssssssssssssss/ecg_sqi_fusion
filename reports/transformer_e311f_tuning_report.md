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
| R1 cls-only SNR 0.05 | pending |  |  |  |  |  |
| R1 cls-only SNR 0.10 | pending |  |  |  |  |  |
| R1 delayed light denoise | pending |  |  |  |  |  |
| R1 noise-type aux | pending |  |  |  |  |  |

Round-1 best warm-start for round 2: `pending`

## Round 2

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| R2 low-lr continue | pending |  |  |  |  |  |
| R2 label smoothing | pending |  |  |  |  |  |
| R2 good/medium weights | pending |  |  |  |  |  |
| R2 low-lr continue | pending |  |  |  |  |  |
| R2 label smoothing | pending |  |  |  |  |  |
| R2 good/medium weights | pending |  |  |  |  |  |
| R2 low-lr continue | pending |  |  |  |  |  |
| R2 label smoothing | pending |  |  |  |  |  |
| R2 good/medium weights | pending |  |  |  |  |  |
| R2 low-lr continue | pending |  |  |  |  |  |
| R2 label smoothing | pending |  |  |  |  |  |
| R2 good/medium weights | pending |  |  |  |  |  |

Best completed result: `D1 reference` = `0.9465`

## Image Folders

- Images only: `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/<variant>/<class>/<noise_kind>/`
- No HTML/CSV/JSON is required for the visual check.
