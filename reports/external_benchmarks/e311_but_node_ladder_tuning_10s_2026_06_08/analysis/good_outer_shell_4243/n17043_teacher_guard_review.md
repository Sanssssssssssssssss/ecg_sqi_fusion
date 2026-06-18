# N17043 Good Outer Shell Teacher/Guard Review

Analysis-only. Clean/SemiClean/node diagnostic remains the selection source; original BUT remains report-only.

## Main Takeaway

- `N12800` feature rule is conservative: it protects medium/bad but leaves a large good outer shell as medium.
- A stronger but still feature-based teacher recovers most of that good outer shell; the earlier grid-search best reached all-target acc `0.940387` on the `N17043` target.
- The remaining bottleneck is medium protection: the best teacher-style methods still leave about 1.3k medium rows predicted as good.
- Next experiment should add a targeted medium hard-negative block matching those false-good medium rows, rather than adding generic good outer rows again.

## Reproduced Comparison

| method | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_to_medium | medium_to_good | bad_to_nonbad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N12800 feature rule on N17043 | 31755 | 0.870162 | 0.898695 | 0.804084 | 0.926233 | 1.000000 | 3339 | 784 | 0 |
| HGB teacher + bad core override | 31755 | 0.935097 | 0.946401 | 0.971543 | 0.851712 | 1.000000 | 485 | 1576 | 0 |
| HGB teacher + medium guard | 31755 | 0.939411 | 0.950315 | 0.965558 | 0.874200 | 1.000000 | 587 | 1337 | 0 |

## Earlier Grid-Search Best

- `hgb_l2_leaf80_guard` with medium guard: all-target acc `0.940387`, macro-F1 `0.951039`.
- Recall good/medium/bad: `0.968726/0.872036/1.000000`.
- Confusion: `[[16510, 533, 0], [1360, 9268, 0], [0, 0, 4084]]`.
- Guard flips `299` rows: true-good lost `94`, true-medium rescued `205`.

## remaining_medium_to_good_vs_correct_medium

| feature | a_median | b_median | delta | abs_delta_over_iqr |
| --- | --- | --- | --- | --- |
| pc2 | 11.081825 | 0.416041 | 10.665784 | 2.105787 |
| band_15_30 | 0.055738 | 0.242077 | -0.186339 | 2.048177 |
| qrs_band_ratio | 0.191472 | 0.510248 | -0.318777 | 1.891419 |
| non_qrs_diff_p95 | 0.024932 | 0.097709 | -0.072777 | 0.970929 |
| band_30_45 | 0.006384 | 0.026692 | -0.020308 | 0.964377 |
| pc4 | 0.548702 | -1.228967 | 1.777669 | 0.934669 |
| baseline_step | 1.180505 | 0.632179 | 0.548326 | 0.934296 |
| sqi_bSQI | 0.818182 | 0.909091 | -0.090909 | 0.779221 |
| pc1 | -3.682689 | -0.448339 | -3.234350 | 0.742319 |
| pc3 | -0.371743 | 2.134255 | -2.505998 | 0.734139 |

## remaining_good_to_medium_vs_correct_good

| feature | a_median | b_median | delta | abs_delta_over_iqr |
| --- | --- | --- | --- | --- |
| pc3 | 1.817280 | -0.764079 | 2.581358 | 0.756216 |
| flatline_ratio | 0.123299 | 0.302642 | -0.179343 | 0.736842 |
| pc1 | -1.190844 | -4.285268 | 3.094424 | 0.710204 |
| pc4 | -1.104853 | 0.231638 | -1.336492 | 0.702705 |
| low_amp_ratio | 0.174400 | 0.312000 | -0.137600 | 0.690763 |
| medium_detail_unreliable_score | 0.273749 | 0.485970 | -0.212221 | 0.605561 |
| qrs_visibility | 0.303856 | 0.494120 | -0.190263 | 0.588302 |
| baseline_step | 0.685912 | 0.358731 | 0.327182 | 0.557486 |
| sqi_kSQI | 13.418005 | 22.477509 | -9.059505 | 0.519246 |
| amplitude_entropy | 0.723098 | 0.628295 | 0.094803 | 0.517433 |

## Figures

- `reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_outer_shell_4243\n17043_teacher_guard_confusion_comparison.png`
- `reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_outer_shell_4243\n17043_teacher_guard_pca_errors.png`
- `reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_outer_shell_4243\n17043_teacher_guard_error_waveforms.png`

## Files

- `reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_outer_shell_4243\n17043_teacher_guard_comparison_metrics.csv`
- `reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_outer_shell_4243\n17043_teacher_guard_error_rows.csv`
- `reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_outer_shell_4243\n17043_teacher_guard_remaining_error_feature_gaps.csv`
- `reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_outer_shell_4243\n17043_teacher_guard_search_top8.csv`
