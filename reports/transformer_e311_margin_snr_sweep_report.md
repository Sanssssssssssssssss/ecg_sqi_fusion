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
| E3.11d main | pending |  |  |  |  | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/models/e311d_snr_primary_good_guard_m1_d1warm_snr005/test_report.json` |
| E3.11e main | pending |  |  |  |  | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311e_snr_only_visual/models/e311e_snr_only_visual_m1_d1warm_snr005/test_report.json` |
| E3.11f main | pending |  |  |  |  | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311f_lite_e310_morph/models/e311f_lite_e310_morph_m1_d1warm_snr005/test_report.json` |
| E3.11g main | pending |  |  |  |  | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311g_lite_snr_primary/models/e311g_lite_snr_primary_m1_d1warm_snr005/test_report.json` |
| E3.11b denoise-aware | pending |  |  |  |  | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311b_snr_gap_e310_morph/models/e311b_snr_gap_e310_morph_m2_d1warm_snr005_denoise/test_report.json` |
| E3.11d denoise-aware | pending |  |  |  |  | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/models/e311d_snr_primary_good_guard_m2_d1warm_snr005_denoise/test_report.json` |

Best new run so far: `E3.11b main` = `0.8966`

## Figures

Combined visual galleries:

- Class x noise examples: `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/visual_gallery/e311_margin_snr_class_noise_examples_gallery.png`
- Counterfactual triplets: `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/visual_gallery/e311_margin_snr_counterfactual_triplets_gallery.png`
- Audit overview: `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/visual_gallery/e311_margin_snr_audit_overview.png`

| Variant | Triplets | Class x Noise Examples |
| --- | --- | --- |
| E3.11b | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311b_snr_gap_e310_morph/figs_label_samples/e311b_snr_gap_e310_morph_counterfactual_triplets.png` | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311b_snr_gap_e310_morph/figs_label_samples/e311b_snr_gap_e310_morph_class_noise_examples.png` |
| E3.11c | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311c_snr_gap_relaxed_morph/figs_label_samples/e311c_snr_gap_relaxed_morph_counterfactual_triplets.png` | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311c_snr_gap_relaxed_morph/figs_label_samples/e311c_snr_gap_relaxed_morph_class_noise_examples.png` |
| E3.11d | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/figs_label_samples/e311d_snr_primary_good_guard_counterfactual_triplets.png` | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/figs_label_samples/e311d_snr_primary_good_guard_class_noise_examples.png` |
| E3.11e | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311e_snr_only_visual/figs_label_samples/e311e_snr_only_visual_counterfactual_triplets.png` | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311e_snr_only_visual/figs_label_samples/e311e_snr_only_visual_class_noise_examples.png` |
| E3.11f | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311f_lite_e310_morph/figs_label_samples/e311f_lite_e310_morph_counterfactual_triplets.png` | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311f_lite_e310_morph/figs_label_samples/e311f_lite_e310_morph_class_noise_examples.png` |
| E3.11g | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311g_lite_snr_primary/figs_label_samples/e311g_lite_snr_primary_counterfactual_triplets.png` | `/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep/e311g_lite_snr_primary/figs_label_samples/e311g_lite_snr_primary_class_noise_examples.png` |

## Reading Guide

- If E3.11b beats current E3.11 clearly, the strict E3.11 morphology margin was the main issue.
- If E3.11d/e beat E3.11b, SNR/visual dirtiness is learnable while morphology labels are less visible.
- If lite variants beat wide variants, the 3-6 dB bad range is too severe for this benchmark.
- If SQI baselines are high, use that variant only as a visual diagnostic, not as the main non-shortcut benchmark.
