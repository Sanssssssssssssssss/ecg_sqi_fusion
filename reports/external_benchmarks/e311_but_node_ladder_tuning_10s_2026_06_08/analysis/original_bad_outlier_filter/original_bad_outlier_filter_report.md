# Original Bad-Outlier Filter Diagnostic

Report-only diagnostic. Original BUT metrics are not used for node selection or promotion.

## Counts

- N6800 trim-bad target: {'bad': 4084, 'good': 6800, 'medium': 6800} = 17684 kept.
- Before trimming N6800 bad outliers: total 18885, bad 5285.
- Original test after dropping only bad/outlier_low_confidence: 8185 windows (3640 good, 4426 medium, 119 bad core/near-boundary).

## Top Filtered Original Candidates

- Top acc: `nl_n6800_gm_trim_bad_scan_109_sc_overlap_narrow_oscillato_1ece5cbe0b5c` `calibrated` acc=0.8109, macro-F1=0.5443, recalls G/M/B=0.842/0.807/0.000.
- Best macro/guardrail: `nl_n6800_gm_trim_bad_scan_093_sc_overlap_compact_pca_core_f60533bc4e32` `calibrated` acc=0.7701, macro-F1=0.7199, recalls G/M/B=0.734/0.808/0.462.

## Figures

- `filtered_original_top_acc_confusion.png`
- `filtered_original_top_acc_pca_errors.png`
- `filtered_original_top_acc_error_waveforms.png`
- `filtered_original_best_macro_confusion.png`
- `denoise_clean_noisy_samples.png`
- `denoise_noise_residual_samples.png`
