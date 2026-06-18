# BUT 10s Clean-Label Core Atlas

## Executive read

- CleanBUT-Core is a diagnostic and generator-target subset; it does not replace the original BUT 10s P1 test.
- Selection uses only atlas signal features plus labels: train-fitted PCA margin and kNN label purity.
- Original label-overlap regions are retained as ambiguous boundary material for analysis, not used as clean target.

## Clean subset counts

| clean_tier | bad | good | medium |
| --- | --- | --- | --- |
| ambiguous_boundary | 4087 | 13779 | 8658 |
| clean_core_strict | 719 | 1920 | 1137 |
| clean_core_train_target | 479 | 1344 | 833 |

## Split counts

| split | clean_tier | class_name | n |
| --- | --- | --- | --- |
| test | ambiguous_boundary | bad | 411 |
| test | ambiguous_boundary | good | 3602 |
| test | ambiguous_boundary | medium | 3895 |
| test | clean_core_strict | good | 17 |
| test | clean_core_strict | medium | 297 |
| test | clean_core_train_target | good | 21 |
| test | clean_core_train_target | medium | 234 |
| train | ambiguous_boundary | bad | 3593 |
| train | ambiguous_boundary | good | 9325 |
| train | ambiguous_boundary | medium | 4682 |
| train | clean_core_strict | bad | 719 |
| train | clean_core_strict | good | 1865 |
| train | clean_core_strict | medium | 824 |
| train | clean_core_train_target | bad | 479 |
| train | clean_core_train_target | good | 1244 |
| train | clean_core_train_target | medium | 591 |
| val | ambiguous_boundary | bad | 83 |
| val | ambiguous_boundary | good | 852 |

## Top clean target features

sample_entropy_proxy, higuchi_fd_proxy, sqi_bSQI, non_qrs_diff_p95, template_corr, diff_abs_median, zero_crossing_rate, band_30_45, hjorth_mobility, sqi_pSQI, wavelet_e4, diff_zero_crossing_rate, flatline_ratio, band_5_15, non_qrs_rms_ratio

## Ambiguous boundary summary

| class_name | ambiguous_type | n |
| --- | --- | --- |
| bad | medium_bad_boundary | 2766 |
| bad | bad_outlier | 1321 |
| good | good_medium_boundary | 7904 |
| good | isolated_good | 3321 |
| good | good_medium_negative_margin | 1537 |
| good | good_medium_low_purity | 1017 |
| medium | good_medium_boundary | 5106 |
| medium | isolated_medium | 2125 |
| medium | good_medium_low_purity | 1397 |
| medium | good_medium_negative_margin | 21 |
| medium | medium_bad_boundary | 9 |

## Synthetic distance leaderboard

| variant_id | source_root | variant_dir | n_common_features | clean_target_distance | clean_good_distance | clean_medium_distance | clean_bad_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper_denoise_interval__ma_only__cinc0p00__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_denoise_interval__ma_only__cinc0p00__cw1p00_1p40_1p70 | 31 | 0.5073 | 0.6298 | 0.2698 | 0.6223 |
| paper_denoise_interval__ma_only__cinc0p00__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_denoise_interval__ma_only__cinc0p00__cw1p00_1p55_1p70 | 31 | 0.5073 | 0.6298 | 0.2698 | 0.6223 |
| paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70 | 31 | 0.5088 | 0.6309 | 0.2744 | 0.6209 |
| paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70 | 31 | 0.5088 | 0.6309 | 0.2744 | 0.6209 |
| paper_fig2_like__ma_only__cinc0p00__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_fig2_like__ma_only__cinc0p00__cw1p00_1p40_1p70 | 31 | 0.5099 | 0.6308 | 0.2697 | 0.6292 |
| paper_fig2_like__ma_only__cinc0p00__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_fig2_like__ma_only__cinc0p00__cw1p00_1p55_1p70 | 31 | 0.5099 | 0.6308 | 0.2697 | 0.6292 |
| paper_fig2_like__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_fig2_like__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70 | 31 | 0.511 | 0.6275 | 0.276 | 0.6295 |
| paper_fig2_like__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_fig2_like__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70 | 31 | 0.511 | 0.6275 | 0.276 | 0.6295 |
| paper_table_strict__ma_only__cinc0p00__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_table_strict__ma_only__cinc0p00__cw1p00_1p55_1p70 | 31 | 0.5149 | 0.6298 | 0.2742 | 0.6407 |
| paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70 | 31 | 0.5163 | 0.6309 | 0.2776 | 0.6404 |
| paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70 | 31 | 0.5163 | 0.6309 | 0.2776 | 0.6404 |
| paper_denoise_interval__em_ma_ma_heavy__cinc0p10__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_denoise_interval__em_ma_ma_heavy__cinc0p10__cw1p00_1p40_1p70 | 31 | 0.518 | 0.6247 | 0.291 | 0.6383 |
| paper_denoise_interval__em_ma_ma_heavy__cinc0p10__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_denoise_interval__em_ma_ma_heavy__cinc0p10__cw1p00_1p55_1p70 | 31 | 0.518 | 0.6247 | 0.291 | 0.6383 |
| paper_fig2_like__em_ma_bw_light__cinc0p00__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_fig2_like__em_ma_bw_light__cinc0p00__cw1p00_1p40_1p70 | 31 | 0.5182 | 0.6184 | 0.294 | 0.6422 |
| paper_fig2_like__em_ma_bw_light__cinc0p00__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_fig2_like__em_ma_bw_light__cinc0p00__cw1p00_1p55_1p70 | 31 | 0.5182 | 0.6184 | 0.294 | 0.6422 |
| mg_qrs_confound_m_strong_detail_softbad__bw_badstrong__cw1p00_1p45_1p75 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05\synthetic_variants\mg_qrs_confound_m_strong_detail_softbad__bw_badstrong__cw1p00_1p45_1p75 | 31 | 0.5183 | 0.6241 | 0.2999 | 0.631 |
| mg_qrs_confound_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05\synthetic_variants\mg_qrs_confound_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70 | 31 | 0.5184 | 0.6241 | 0.2968 | 0.6344 |
| mg_qrs_confound_m_guard_softbad__bw_badstrong__cw1p00_1p45_1p75 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05\synthetic_variants\mg_qrs_confound_m_guard_softbad__bw_badstrong__cw1p00_1p45_1p75 | 31 | 0.5185 | 0.6241 | 0.2968 | 0.6347 |
| mg_visible_unusable_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05\synthetic_variants\mg_visible_unusable_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78 | 31 | 0.5185 | 0.6241 | 0.2999 | 0.6317 |
| paper_fig2_like__em_ma_ma_heavy__cinc0p10__cw1p00_1p40_1p70 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05 | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_sensors2025_noise_synthesis_but_10s_2026_06_05\scan_variants\paper_fig2_like__em_ma_ma_heavy__cinc0p10__cw1p00_1p40_1p70 | 31 | 0.5187 | 0.6245 | 0.281 | 0.6505 |

## Rule thresholds

```json
{
  "rules": {
    "clean_core_strict": {
      "knn_label_purity_min": 0.9,
      "pca_margin_quantile_within_class_train": 0.85
    },
    "clean_core_train_target": {
      "knn_label_purity_min": 0.85,
      "pca_margin_quantile_within_class_train": 0.75
    }
  },
  "class_thresholds": {
    "good": {
      "train_n": 12434,
      "strict_margin_min": 3.5676378037836303,
      "target_margin_min": 3.192209455354154,
      "own_distance_median_train": 4.370313785141294,
      "own_distance_q90_train": 6.829745101044882
    },
    "medium": {
      "train_n": 6097,
      "strict_margin_min": 3.2303989944829365,
      "target_margin_min": 2.9144131427514965,
      "own_distance_median_train": 3.9131827836014192,
      "own_distance_q90_train": 6.189229924022143
    },
    "bad": {
      "train_n": 4791,
      "strict_margin_min": 11.163180140725098,
      "target_margin_min": 11.082112510037152,
      "own_distance_median_train": 0.6454308958955289,
      "own_distance_q90_train": 1.1048350603258883
    }
  }
}
```

## Files

- Output root: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_clean_label_core_10s_2026_06_06`
- Report root: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_clean_label_core_10s_2026_06_06`
- Figures: `figures/clean_pca_class.png`, `figures/clean_vs_ambiguous_pca.png`, `figures/clean_feature_ridges.png`