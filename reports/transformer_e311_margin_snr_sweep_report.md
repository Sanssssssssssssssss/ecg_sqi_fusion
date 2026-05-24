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
| E3.11b | wide SNR + E3.10 morphology | pending |  |  |  |  |  |  |  |  |  | pending |
| E3.11c | wide SNR + relaxed morphology | pending |  |  |  |  |  |  |  |  |  | pending |
| E3.11d | wide SNR-primary + good guard | pending |  |  |  |  |  |  |  |  |  | pending |
| E3.11e | wide SNR-only visual | pending |  |  |  |  |  |  |  |  |  | pending |
| E3.11f | lite SNR + E3.10 morphology | pending |  |  |  |  |  |  |  |  |  | pending |
| E3.11g | lite SNR-primary | pending |  |  |  |  |  |  |  |  |  | pending |

## Transformer Runs

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E3.11b main | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311b_snr_gap_e310_morph/models/e311b_snr_gap_e310_morph_m1_d1warm_snr005/test_report.json` |
| E3.11c main | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311c_snr_gap_relaxed_morph/models/e311c_snr_gap_relaxed_morph_m1_d1warm_snr005/test_report.json` |
| E3.11d main | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/models/e311d_snr_primary_good_guard_m1_d1warm_snr005/test_report.json` |
| E3.11e main | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311e_snr_only_visual/models/e311e_snr_only_visual_m1_d1warm_snr005/test_report.json` |
| E3.11f main | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311f_lite_e310_morph/models/e311f_lite_e310_morph_m1_d1warm_snr005/test_report.json` |
| E3.11g main | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311g_lite_snr_primary/models/e311g_lite_snr_primary_m1_d1warm_snr005/test_report.json` |
| E3.11b denoise-aware | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311b_snr_gap_e310_morph/models/e311b_snr_gap_e310_morph_m2_d1warm_snr005_denoise/test_report.json` |
| E3.11d denoise-aware | pending |  |  |  |  | `outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/models/e311d_snr_primary_good_guard_m2_d1warm_snr005_denoise/test_report.json` |

## Figures

| Variant | Triplets | Class x Noise Examples |
| --- | --- | --- |
| E3.11b | `outputs/transformer_e311_margin_snr_sweep/e311b_snr_gap_e310_morph/figs_label_samples/e311b_snr_gap_e310_morph_counterfactual_triplets.png` | `outputs/transformer_e311_margin_snr_sweep/e311b_snr_gap_e310_morph/figs_label_samples/e311b_snr_gap_e310_morph_class_noise_examples.png` |
| E3.11c | `outputs/transformer_e311_margin_snr_sweep/e311c_snr_gap_relaxed_morph/figs_label_samples/e311c_snr_gap_relaxed_morph_counterfactual_triplets.png` | `outputs/transformer_e311_margin_snr_sweep/e311c_snr_gap_relaxed_morph/figs_label_samples/e311c_snr_gap_relaxed_morph_class_noise_examples.png` |
| E3.11d | `outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/figs_label_samples/e311d_snr_primary_good_guard_counterfactual_triplets.png` | `outputs/transformer_e311_margin_snr_sweep/e311d_snr_primary_good_guard/figs_label_samples/e311d_snr_primary_good_guard_class_noise_examples.png` |
| E3.11e | `outputs/transformer_e311_margin_snr_sweep/e311e_snr_only_visual/figs_label_samples/e311e_snr_only_visual_counterfactual_triplets.png` | `outputs/transformer_e311_margin_snr_sweep/e311e_snr_only_visual/figs_label_samples/e311e_snr_only_visual_class_noise_examples.png` |
| E3.11f | `outputs/transformer_e311_margin_snr_sweep/e311f_lite_e310_morph/figs_label_samples/e311f_lite_e310_morph_counterfactual_triplets.png` | `outputs/transformer_e311_margin_snr_sweep/e311f_lite_e310_morph/figs_label_samples/e311f_lite_e310_morph_class_noise_examples.png` |
| E3.11g | `outputs/transformer_e311_margin_snr_sweep/e311g_lite_snr_primary/figs_label_samples/e311g_lite_snr_primary_counterfactual_triplets.png` | `outputs/transformer_e311_margin_snr_sweep/e311g_lite_snr_primary/figs_label_samples/e311g_lite_snr_primary_class_noise_examples.png` |

## Reading Guide

- If E3.11b beats current E3.11 clearly, the strict E3.11 morphology margin was the main issue.
- If E3.11d/e beat E3.11b, SNR/visual dirtiness is learnable while morphology labels are less visible.
- If lite variants beat wide variants, the 3-6 dB bad range is too severe for this benchmark.
- If SQI baselines are high, use that variant only as a visual diagnostic, not as the main non-shortcut benchmark.
