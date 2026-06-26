# GM Mechanism Repair Suite

- Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Folds: `3`
- Epochs: `12`
- Seed: `20260951`
- Suite: `all`

## Clean Test Summary

| ('candidate', '') | ('acc', 'mean') | ('acc', 'std') | ('acc', 'max') | ('macro_f1', 'mean') | ('macro_f1', 'std') | ('macro_f1', 'max') | ('good_recall', 'mean') | ('good_recall', 'std') | ('good_recall', 'max') | ('medium_recall', 'mean') | ('medium_recall', 'std') | ('medium_recall', 'max') | ('bad_recall', 'mean') | ('bad_recall', 'std') | ('bad_recall', 'max') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E24_e29e30_baseline | 0.917177 | 0.00639727 | 0.920994 | 0.898747 | 0.0104583 | 0.906794 | 0.849749 | 0.0548223 | 0.900592 | 0.855553 | 0.0240094 | 0.870884 | 0.981994 | 0.00061081 | 0.982656 |
| E29_e24_finalscore_pairrank_mediumguard | 0.903761 | 0.02265 | 0.919235 | 0.88293 | 0.025463 | 0.900584 | 0.822191 | 0.0644721 | 0.883333 | 0.841516 | 0.0264702 | 0.86747 | 0.976139 | 0.00164872 | 0.977205 |
| E30_e29_reliable_detached_factor_fusion | 0.917508 | 0.0035146 | 0.921242 | 0.900581 | 0.00642743 | 0.907715 | 0.869858 | 0.0240141 | 0.896429 | 0.852103 | 0.0341035 | 0.873484 | 0.976008 | 0.00330148 | 0.978442 |

## Factor Contract Variance

| feature | encoded_std | encoded_var |
| --- | --- | --- |
| amplitude_entropy | 0.0227606 | 0.000518052 |
| baseline_step | 1 | 1 |
| contact_loss_win_ratio | 0.0039267 | 1.54397e-05 |
| detector_agreement | 0.323125 | 0.104412 |
| flatline_ratio | 0.0689058 | 0.00474807 |
| non_qrs_diff_p95 | 1 | 1 |
| non_qrs_rms_ratio | 0.208689 | 0.0435524 |
| qrs_band_ratio | 0.154338 | 0.0238203 |
| qrs_visibility | 0.238238 | 0.0567571 |
| sqi_basSQI | 0.121057 | 0.0146548 |
| template_corr | 0.328269 | 0.10776 |

## Pairrank Audit

- candidate pair groups: `863`
- total groups: `11705`

## Local Event Diagnostics

| candidate | fold | seed_index | qrs_event_dice | peak_count_error |
| --- | --- | --- | --- | --- |
| E24_e29e30_baseline | 0 | 0 | 0.369924 | 80.5543 |
| E24_e29e30_baseline | 1 | 0 | 0.35275 | 85.0865 |
| E24_e29e30_baseline | 2 | 0 | 0.355162 | 88.9751 |
| E29_e24_finalscore_pairrank_mediumguard | 0 | 0 | 0.35225 | 86.436 |
| E29_e24_finalscore_pairrank_mediumguard | 1 | 0 | 0.315122 | 112.687 |
| E29_e24_finalscore_pairrank_mediumguard | 2 | 0 | 0.356967 | 86.8981 |
| E30_e29_reliable_detached_factor_fusion | 0 | 0 | 0.353755 | 85.4445 |
| E30_e29_reliable_detached_factor_fusion | 1 | 0 | 0.356231 | 82.872 |
| E30_e29_reliable_detached_factor_fusion | 2 | 0 | 0.355034 | 89.4123 |

## Expert Usage

No expert usage rows.

## Decoded Factor Recovery

| candidate | feature | corr_all | mae |
| --- | --- | --- | --- |
| E24_e29e30_baseline | amplitude_entropy | 0.334634 | 0.01803 |
| E24_e29e30_baseline | baseline_step | 0.972493 | 0.01096 |
| E24_e29e30_baseline | contact_loss_win_ratio | 0.267122 | 0.00581259 |
| E24_e29e30_baseline | detector_agreement | 0.709902 | 0.258602 |
| E24_e29e30_baseline | flatline_ratio | 0.855953 | 0.0218034 |
| E24_e29e30_baseline | non_qrs_diff_p95 | 0.901763 | 0.192944 |
| E24_e29e30_baseline | non_qrs_rms_ratio | 0.968681 | 0.040306 |
| E24_e29e30_baseline | qrs_band_ratio | 0.95543 | 0.483532 |
| E24_e29e30_baseline | qrs_visibility | 0.964617 | 0.0844331 |
| E24_e29e30_baseline | sqi_basSQI | 0.967722 | 0.024894 |
| E24_e29e30_baseline | template_corr | 0.947434 | 0.0748143 |
| E29_e24_finalscore_pairrank_mediumguard | amplitude_entropy | 0.279673 | 0.0187352 |
| E29_e24_finalscore_pairrank_mediumguard | baseline_step | 0.955164 | 0.0137367 |
| E29_e24_finalscore_pairrank_mediumguard | contact_loss_win_ratio | 0.188302 | 0.0108854 |
| E29_e24_finalscore_pairrank_mediumguard | detector_agreement | 0.63482 | 0.264303 |
| E29_e24_finalscore_pairrank_mediumguard | flatline_ratio | 0.679306 | 0.0299461 |
| E29_e24_finalscore_pairrank_mediumguard | non_qrs_diff_p95 | 0.850082 | 0.23057 |
| E29_e24_finalscore_pairrank_mediumguard | non_qrs_rms_ratio | 0.932481 | 0.0570671 |
| E29_e24_finalscore_pairrank_mediumguard | qrs_band_ratio | 0.939299 | 0.556909 |
| E29_e24_finalscore_pairrank_mediumguard | qrs_visibility | 0.961627 | 0.0904142 |
| E29_e24_finalscore_pairrank_mediumguard | sqi_basSQI | 0.94527 | 0.0294301 |
| E29_e24_finalscore_pairrank_mediumguard | template_corr | 0.941589 | 0.0784123 |
| E30_e29_reliable_detached_factor_fusion | amplitude_entropy | 0.334709 | 0.0179874 |
| E30_e29_reliable_detached_factor_fusion | baseline_step | 0.97463 | 0.00999231 |
| E30_e29_reliable_detached_factor_fusion | contact_loss_win_ratio | 0.285242 | 0.00579951 |
| E30_e29_reliable_detached_factor_fusion | detector_agreement | 0.716076 | 0.258966 |
| E30_e29_reliable_detached_factor_fusion | flatline_ratio | 0.848086 | 0.021215 |
| E30_e29_reliable_detached_factor_fusion | non_qrs_diff_p95 | 0.90061 | 0.185383 |
| E30_e29_reliable_detached_factor_fusion | non_qrs_rms_ratio | 0.962574 | 0.0414753 |
| E30_e29_reliable_detached_factor_fusion | qrs_band_ratio | 0.954715 | 0.481737 |
| E30_e29_reliable_detached_factor_fusion | qrs_visibility | 0.963761 | 0.083629 |
| E30_e29_reliable_detached_factor_fusion | sqi_basSQI | 0.96964 | 0.0236958 |
| E30_e29_reliable_detached_factor_fusion | template_corr | 0.94743 | 0.0755471 |
