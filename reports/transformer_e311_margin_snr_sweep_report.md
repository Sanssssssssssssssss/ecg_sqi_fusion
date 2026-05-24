# E3.11 SNR/Morphology Learnability Sweep

Goal: diagnose whether E3.11 is hard because the SNR gap is too wide, the morphology margin is too strict, or clean-reference morphology labels are weakly visible in raw noisy ECG.

## Baselines

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| D1 reference | 0.9465 | 0.9153 | 0.9465 | 0.9777 | `[[616, 53, 4], [22, 637, 14], [2, 13, 658]]` |
| E3.10 M2 best | 0.9402 | 0.9491 | 0.9135 | 0.9580 | `[[746, 35, 5], [48, 718, 20], [7, 26, 753]]` |
| E3.11 M1 best | 0.9000 | 0.9017 | 0.8629 | 0.9353 | `[[697, 65, 11], [61, 667, 45], [5, 45, 723]]` |

## Data Audit

| Variant | Design | Train Groups | Val Groups | Test Groups | SNR Oracle | SNR Mean Gap | Smooth Mean G/M/B | Global Mean G/M/B | Obs Margin p10/p50 | SQI-SVM | SQI-MLP | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| E3.11b | wide SNR + E3.10 morphology | 3985 | 796 | 796 | 0.6667 | 9.8042 | 0.0655/0.3225/0.5597 | 0.2012/0.3989/0.6224 | 0.2640/0.3671 | 0.6189 | 0.5879 | pass |
| E3.11c | wide SNR + relaxed morphology | 3999 | 799 | 800 | 0.6667 | 9.8193 | 0.0671/0.3112/0.5736 | 0.2009/0.4004/0.6227 | 0.2644/0.3717 | 0.5975 | 0.5763 | pass |
| E3.11d | wide SNR-primary + good guard | 4000 | 800 | 800 | 0.6667 | 10.0651 | 0.0466/0.2732/0.4414 | 0.1977/0.3981/0.6298 | 0.2127/0.3739 | 0.5463 | 0.5258 | pass |
| E3.11e | wide SNR-only visual | 4000 | 800 | 800 | 0.6667 | 10.0760 | 0.0466/0.2798/0.4336 | 0.1977/0.3981/0.6305 | 0.2135/0.3799 | 0.5225 | 0.5262 | pass |
| E3.11f | lite SNR + E3.10 morphology | 3799 | 764 | 775 | 0.6552 | 5.4172 | 0.0741/0.3087/0.4009 | 0.2447/0.3532/0.4563 | 0.3010/0.3617 | 0.5265 | 0.5157 | pass |
| E3.11g | lite SNR-primary | 3999 | 798 | 800 | 0.6667 | 5.3632 | 0.0568/0.2400/0.3110 | 0.2426/0.3447/0.4495 | 0.1010/0.3367 | 0.4408 | 0.4275 | pass |

## Transformer Runs

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E3.11b main | 0.8966 | 0.8869 | 0.8744 | 0.9284 |  | `[[706, 81, 9], [61, 696, 39], [13, 44, 739]]` |
| E3.11c main | 0.8862 | 0.8950 | 0.8450 | 0.9187 |  | `[[716, 76, 8], [65, 676, 59], [8, 57, 735]]` |
| E3.11d main | 0.8296 | 0.8888 | 0.7438 | 0.8562 |  | `[[711, 73, 16], [80, 595, 125], [24, 91, 685]]` |
| E3.11e main | 0.8350 | 0.8950 | 0.7488 | 0.8612 |  | `[[716, 60, 24], [94, 599, 107], [18, 93, 689]]` |
| E3.11f main | 0.9329 | 0.9174 | 0.9187 | 0.9626 |  | `[[711, 61, 3], [51, 712, 12], [7, 22, 746]]` |
| E3.11g main | 0.7346 | 0.8550 | 0.6450 | 0.7037 |  | `[[684, 68, 48], [90, 516, 194], [45, 192, 563]]` |
| E3.11b denoise-aware | 0.8957 | 0.8907 | 0.8719 | 0.9246 |  | `[[709, 82, 5], [64, 694, 38], [11, 49, 736]]` |
| E3.11d denoise-aware | 0.8375 | 0.8988 | 0.7612 | 0.8525 |  | `[[719, 63, 18], [75, 609, 116], [23, 95, 682]]` |

Best new run so far: `E3.11f main` = `0.9329`

## Figures

Combined visual galleries:

- Class x noise examples: `outputs/transformer_e311_margin_snr_sweep/visual_gallery/e311_margin_snr_class_noise_examples_gallery.png`
- Counterfactual triplets: `outputs/transformer_e311_margin_snr_sweep/visual_gallery/e311_margin_snr_counterfactual_triplets_gallery.png`
- Audit overview: `outputs/transformer_e311_margin_snr_sweep/visual_gallery/e311_margin_snr_audit_overview.png`

Individual real ECG case folders:

- Folder root: `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders`
- Content: PNG images only for visual inspection.
- Layout: `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/<variant>/<good|medium|bad>/<em|ma|mix>/`

| Variant | Real ECG Case Folder |
| --- | --- |
| E3.11b | `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/e311b_snr_gap_e310_morph` |
| E3.11c | `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/e311c_snr_gap_relaxed_morph` |
| E3.11d | `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/e311d_snr_primary_good_guard` |
| E3.11e | `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/e311e_snr_only_visual` |
| E3.11f | `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/e311f_lite_e310_morph` |
| E3.11g | `outputs/transformer_e311_margin_snr_sweep/real_ecg_case_folders/by_variant/e311g_lite_snr_primary` |

## Reading Guide

- If E3.11b beats current E3.11 clearly, the strict E3.11 morphology margin was the main issue.
- If E3.11d/e beat E3.11b, SNR/visual dirtiness is learnable while morphology labels are less visible.
- If lite variants beat wide variants, the 3-6 dB bad range is too severe for this benchmark.
- If SQI baselines are high, use that variant only as a visual diagnostic, not as the main non-shortcut benchmark.
