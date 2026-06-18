# CleanBUT Bad-Core Targeted Synthetic Grid

This is a generator-target scan. CleanBUT-Core is used only as a target/diagnostic subset; original BUT 10s P1 remains the benchmark.

## Current Best CPU Fit

- Best variant: `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626`
- Weighted score: `0.2922`
- Bad 64D distance: `0.6852` vs prior baseline `~0.748`
- Medium 64D distance: `0.2689` vs prior baseline `~0.311`

## Figures

- `figures/top_rules_64d_overlay.png`: CleanBUT-Core background with top targeted PTB rules.
- `figures/best_rule_64d_overlay.png`: best current candidate in the same PCA space.
- `figures/bad_core_centroid_shift.png`: class centroid gaps in CleanBUT 64D PCA.
- `figures/classwise_distance_bars.png`: class-wise distance leaderboard.

## Top No-Training Candidates

| rank | variant_id | family | score | class_worst_64d_KS | good_64d_KS | medium_64d_KS | bad_64d_KS | bad_distance_improvement_vs_baseline | medium_regression_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626 | bad_1530_locked_lowpc2_core_refine | 0.2922 | 0.6852 | 0.3400 | 0.2689 | 0.6852 | 0.0840 | -0.1347 |
| 2 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d8774356567f | bad_1530_locked_lowpc2_core_refine | 0.2922 | 0.6852 | 0.3400 | 0.2689 | 0.6852 | 0.0840 | -0.1347 |
| 3 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d7274ab2f04d | bad_1530_locked_lowpc2_core_refine | 0.2931 | 0.6852 | 0.3487 | 0.2689 | 0.6852 | 0.0840 | -0.1347 |
| 4 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_adf610ec036d | bad_1530_locked_lowpc2_core_refine | 0.2931 | 0.6852 | 0.3487 | 0.2689 | 0.6852 | 0.0840 | -0.1347 |
| 5 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626 | bad_1530_locked_lowpc2_core_refine | 0.2983 | 0.6885 | 0.3400 | 0.2737 | 0.6885 | 0.0795 | -0.1194 |
| 6 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d8774356567f | bad_1530_locked_lowpc2_core_refine | 0.2983 | 0.6885 | 0.3400 | 0.2737 | 0.6885 | 0.0795 | -0.1194 |
| 7 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_238aae411c7e | bad_1530_locked_lowpc2_core_refine | 0.2985 | 0.7111 | 0.3844 | 0.2800 | 0.7111 | 0.0493 | -0.0990 |
| 8 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_6bb3561abb9a | bad_1530_locked_lowpc2_core_refine | 0.2985 | 0.7111 | 0.3844 | 0.2800 | 0.7111 | 0.0493 | -0.0990 |
| 9 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626 | bad_1530_locked_lowpc2_core_refine | 0.2987 | 0.6847 | 0.3400 | 0.2800 | 0.6847 | 0.0846 | -0.0990 |
| 10 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d8774356567f | bad_1530_locked_lowpc2_core_refine | 0.2987 | 0.6847 | 0.3400 | 0.2800 | 0.6847 | 0.0846 | -0.0990 |
| 11 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_238aae411c7e | bad_1530_locked_lowpc2_core_refine | 0.2988 | 0.7182 | 0.3844 | 0.2737 | 0.7182 | 0.0398 | -0.1194 |
| 12 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_6bb3561abb9a | bad_1530_locked_lowpc2_core_refine | 0.2988 | 0.7182 | 0.3844 | 0.2737 | 0.7182 | 0.0398 | -0.1194 |
| 13 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d7274ab2f04d | bad_1530_locked_lowpc2_core_refine | 0.2992 | 0.6885 | 0.3487 | 0.2737 | 0.6885 | 0.0795 | -0.1194 |
| 14 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_adf610ec036d | bad_1530_locked_lowpc2_core_refine | 0.2992 | 0.6885 | 0.3487 | 0.2737 | 0.6885 | 0.0795 | -0.1194 |
| 15 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d7274ab2f04d | bad_1530_locked_lowpc2_core_refine | 0.2996 | 0.6847 | 0.3487 | 0.2800 | 0.6847 | 0.0846 | -0.0990 |
| 16 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_adf610ec036d | bad_1530_locked_lowpc2_core_refine | 0.2996 | 0.6847 | 0.3487 | 0.2800 | 0.6847 | 0.0846 | -0.0990 |
| 17 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_238aae411c7e | bad_1530_locked_lowpc2_core_refine | 0.3014 | 0.7143 | 0.3844 | 0.2689 | 0.7143 | 0.0451 | -0.1347 |
| 18 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_6bb3561abb9a | bad_1530_locked_lowpc2_core_refine | 0.3014 | 0.7143 | 0.3844 | 0.2689 | 0.7143 | 0.0451 | -0.1347 |
| 19 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_e587f1dc57a3 | bad_1530_locked_lowpc2_core_refine | 0.3022 | 0.7111 | 0.3961 | 0.2800 | 0.7111 | 0.0493 | -0.0990 |
| 20 | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_7f5875fa6ab6 | bad_1530_locked_lowpc2_core_refine | 0.3022 | 0.7111 | 0.3961 | 0.2800 | 0.7111 | 0.0493 | -0.0990 |

## Training Results

| mode | variant_id | acc | macro_f1 | good_recall | medium_recall | bad_recall | balanced_macro | clean_diag_macro | score | bad_64d_KS | medium_64d_KS | good_64d_KS | domain_separability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quick | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626 | 0.6141 | 0.4999 | 0.5841 | 0.6821 | 0.1484 | 0.4464 | None | 0.2922 | 0.6852 | 0.2689 | 0.3400 | 1.0000 |
| quick | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d8774356567f | 0.6301 | 0.4173 | 0.4343 | 0.8491 | 0.0049 | 0.3608 | None | 0.2922 | 0.6852 | 0.2689 | 0.3400 | 1.0000 |
| quick | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d7274ab2f04d | 0.6462 | 0.4314 | 0.5044 | 0.8229 | 0.0000 | 0.3648 | None | 0.2931 | 0.6852 | 0.2689 | 0.3487 | 1.0000 |
| quick | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_adf610ec036d | 0.6201 | 0.4195 | 0.5349 | 0.7476 | 0.0024 | 0.3585 | None | 0.2931 | 0.6852 | 0.2689 | 0.3487 | 1.0000 |
| quick | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_238aae411c7e | 0.6378 | 0.4249 | 0.4890 | 0.8195 | 0.0000 | 0.3579 | None | 0.2985 | 0.7111 | 0.2800 | 0.3844 | 1.0000 |
| quick | cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_6bb3561abb9a | 0.6469 | 0.5624 | 0.3929 | 0.8891 | 0.2895 | 0.5117 | None | 0.2985 | 0.7111 | 0.2800 | 0.3844 | 1.0000 |
| quick | cc_bad_1530_spike_core_quiet_core_tight_sec0p00_cw1p00_1p_8e39b9c324f4 | 0.7662 | 0.6594 | 0.6415 | 0.8974 | 0.4574 | 0.6640 | None | 0.4525 | 0.6007 | 0.2828 | 0.3420 | 1.0000 |
| quick | cc_bad_narrow_oscillatory_core_quiet_core_tight_sec0p00_c_1b585bae6df3 | 0.6780 | 0.4604 | 0.6407 | 0.7716 | 0.0000 | 0.3804 | None | 0.4432 | 0.6023 | 0.2828 | 0.3420 | 1.0000 |

## Notes

- Selection uses CleanBUT train-target core features and does not inspect BUT test predictions.
- Synthetic `sqi_iSQI` remains a single-lead detector-agreement proxy.
- `all` defaults to CPU distribution fitting only; pass `--run_training` for quick/full training after visual review.
