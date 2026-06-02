# E3.11f Mature Denoise Baseline

This directory is the fixed baseline package for the current best visual denoise setup.

## Definition

- Dataset: `med6p25_badgap7_badcm0p75`
- Good SNR: `11.5-13.0 dB`
- Medium SNR: `5.5-7.0 dB`
- Bad SNR: `-1.5-0.0 dB`
- Denoiser: `residual_unet`
- Loss family: `morph_soft_guard`
- Inference residual scale: `0.955`
- Source checkpoint: `E:\GPTProject2\ecg\outputs\experiment\e311_morph_denoise_gap5_7_grid\artifact\round11\r11_med6p25_badgap5_badc1p25_residual_unet_morph_soft_guard_ca39bf907_scale0p93_lc50p0_ld2p5_ep2\ckpt_best_denoise.pt`
- Classifier checkpoint: `E:\GPTProject2\ecg\outputs\experiment\e311_denoise_strong_grid\artifact\models\promote_strong_noise_clean_multiscale_deriv_multiscale_patch_ld5p0_ep8_ld12p5_ep14\ckpt_best_val.pt`

## Classification

- Noisy-input accuracy: `0.9332`
- Denoised-input accuracy: `0.9614`
- Denoised recall good/medium/bad: `0.9441` / `0.9496` / `0.9905`

## Denoise Metrics

- Denoise score: `2.8193`
- MSE ratio: `0.1029`
- SNR gain: `+7.69 dB`
- Critical-region MAE reduction: `0.6145`
- QRS timing delta: `-2.88 ms`
- QRS amplitude error delta: `-0.0730`
- SSIM delta: `0.2417`
- Patch-correlation delta: `0.0980`

## Files

- `metrics/test_report.json`
- `metrics/denoise_metrics.json`
- `metrics/sample_metrics.csv`
- `figures/confusion_matrices.png`
- `figures/metrics_by_class.png`
- `figures/sample_distributions.png`
- `figures/error_spectrum_by_class.png`
- `figures/training_curves_source_round11.png`
- `visuals/main/denoise_compact_test.png`
- `visuals/main/denoise_hard_test.png`
- `visuals/extended/*.png`

## Promotion Status

This is a strong experiment baseline, not yet a mainline promotion. The next publishable step is to replace fixed/oracle-style scale choices with a learned quality/SNR gate and validate across all sweep families plus the original strict E3.11f data.
