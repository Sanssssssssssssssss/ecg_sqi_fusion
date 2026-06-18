# Original Bad-Outlier Filter Diagnostic

Report-only diagnostic. Original BUT metrics are not used for node selection or promotion.

## Counts

- N7200_gm_trim_bad trim-bad target: {'bad': 4084, 'good': 7200, 'medium': 7200} = 18484 kept.
- Before trimming N7200_gm_trim_bad bad outliers: total 19685, bad 5285.
- Original test after dropping only bad/outlier_low_confidence: 8185 windows (3640 good, 4426 medium, 119 bad core/near-boundary).

## Top Filtered Original Candidates

- Top acc: `nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good122_mid174_seed20260721` `calibrated` acc=0.7759, macro-F1=0.5202, recalls G/M/B=0.750/0.818/0.000.
- Best macro/guardrail: `nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good128_mid168_837d9498a6ae` `calibrated` acc=0.7278, macro-F1=0.7635, recalls G/M/B=0.750/0.709/0.723.

## Figures

- `filtered_original_top_acc_confusion.png`
- `filtered_original_top_acc_pca_errors.png`
- `filtered_original_top_acc_error_waveforms.png`
- `filtered_original_best_macro_confusion.png`
- `denoise_clean_noisy_samples.png`
- `denoise_noise_residual_samples.png`
