# GM Mechanism Repair Suite

- Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Folds: `3`
- Epochs: `12`
- Seed: `20260921`
- Suite: `all`

## Clean Test Summary

| ('candidate', '') | ('acc', 'mean') | ('acc', 'std') | ('acc', 'max') | ('macro_f1', 'mean') | ('macro_f1', 'std') | ('macro_f1', 'max') | ('good_recall', 'mean') | ('good_recall', 'std') | ('good_recall', 'max') | ('medium_recall', 'mean') | ('medium_recall', 'std') | ('medium_recall', 'max') | ('bad_recall', 'mean') | ('bad_recall', 'std') | ('bad_recall', 'max') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E21_e6_subcls_consistency | 0.922726 | 0.00583333 | 0.92646 | 0.906753 | 0.00882671 | 0.91398 | 0.88182 | 0.0288535 | 0.911243 | 0.864165 | 0.0116389 | 0.873484 | 0.974426 | 0.00563827 | 0.979907 |
| E22_e6_subtype_class_fusion | 0.919661 | 0.0081399 | 0.927205 | 0.90305 | 0.0114586 | 0.911428 | 0.880631 | 0.0110786 | 0.891124 | 0.857372 | 0.0270449 | 0.882149 | 0.972364 | 0.00908827 | 0.980178 |
| E23_e14_subtype_class_fusion | 0.923968 | 0.00820347 | 0.929193 | 0.90796 | 0.0123982 | 0.917283 | 0.875661 | 0.0450604 | 0.905325 | 0.866162 | 0.0100213 | 0.87695 | 0.977958 | 0.00240609 | 0.979912 |
| E24_e6_subtype_fusion_pairrank | 0.924382 | 0.00787329 | 0.93118 | 0.908527 | 0.0118922 | 0.919768 | 0.902949 | 0.0366658 | 0.926627 | 0.856647 | 0.0163557 | 0.872617 | 0.972482 | 0.00281263 | 0.975719 |

## Factor Contract Variance

| feature | encoded_std | encoded_var |
| --- | --- | --- |
| amplitude_entropy | 0.0226455 | 0.000512848 |
| baseline_step | 1 | 1 |
| contact_loss_win_ratio | 0.00384 | 1.47549e-05 |
| detector_agreement | 0.322873 | 0.104248 |
| flatline_ratio | 0.0681741 | 0.00464779 |
| non_qrs_diff_p95 | 1 | 1 |
| non_qrs_rms_ratio | 0.208312 | 0.0433941 |
| qrs_band_ratio | 0.154472 | 0.023862 |
| qrs_visibility | 0.238378 | 0.0568244 |
| sqi_basSQI | 0.120861 | 0.0146074 |
| template_corr | 0.328367 | 0.107825 |

## Pairrank Audit

- candidate pair groups: `870`
- total groups: `11722`

## Local Event Diagnostics

| candidate | fold | seed_index | qrs_event_dice | peak_count_error |
| --- | --- | --- | --- | --- |
| E21_e6_subcls_consistency | 0 | 0 | 0.359458 | 83.4529 |
| E21_e6_subcls_consistency | 1 | 0 | 0.359709 | 80.325 |
| E21_e6_subcls_consistency | 2 | 0 | 0.344987 | 87.9058 |
| E22_e6_subtype_class_fusion | 0 | 0 | 0.35786 | 82.9791 |
| E22_e6_subtype_class_fusion | 1 | 0 | 0.354896 | 81.3448 |
| E22_e6_subtype_class_fusion | 2 | 0 | 0.355569 | 83.3402 |
| E23_e14_subtype_class_fusion | 0 | 0 | 0.359255 | 82.7337 |
| E23_e14_subtype_class_fusion | 1 | 0 | 0.359824 | 83.5081 |
| E23_e14_subtype_class_fusion | 2 | 0 | 0.346721 | 88.0768 |
| E24_e6_subtype_fusion_pairrank | 0 | 0 | 0.361942 | 81.6954 |
| E24_e6_subtype_fusion_pairrank | 1 | 0 | 0.359056 | 81.3973 |
| E24_e6_subtype_fusion_pairrank | 2 | 0 | 0.345309 | 88.8554 |

## Expert Usage

No expert usage rows.

## Decoded Factor Recovery

| candidate | feature | corr_all | mae |
| --- | --- | --- | --- |
| E21_e6_subcls_consistency | amplitude_entropy | 0.215368 | 0.0183637 |
| E21_e6_subcls_consistency | baseline_step | 0.968905 | 0.0129691 |
| E21_e6_subcls_consistency | contact_loss_win_ratio | 0.18598 | 0.00548744 |
| E21_e6_subcls_consistency | detector_agreement | 0.749035 | 0.258541 |
| E21_e6_subcls_consistency | flatline_ratio | 0.844567 | 0.0220186 |
| E21_e6_subcls_consistency | non_qrs_diff_p95 | 0.897065 | 0.186062 |
| E21_e6_subcls_consistency | non_qrs_rms_ratio | 0.96833 | 0.0395039 |
| E21_e6_subcls_consistency | qrs_band_ratio | 0.951139 | 0.505058 |
| E21_e6_subcls_consistency | qrs_visibility | 0.963783 | 0.0816063 |
| E21_e6_subcls_consistency | sqi_basSQI | 0.9656 | 0.02457 |
| E21_e6_subcls_consistency | template_corr | 0.949889 | 0.07435 |
| E22_e6_subtype_class_fusion | amplitude_entropy | 0.268432 | 0.0179914 |
| E22_e6_subtype_class_fusion | baseline_step | 0.962719 | 0.0129473 |
| E22_e6_subtype_class_fusion | contact_loss_win_ratio | 0.198011 | 0.00549327 |
| E22_e6_subtype_class_fusion | detector_agreement | 0.744663 | 0.257656 |
| E22_e6_subtype_class_fusion | flatline_ratio | 0.849724 | 0.0209597 |
| E22_e6_subtype_class_fusion | non_qrs_diff_p95 | 0.894436 | 0.198746 |
| E22_e6_subtype_class_fusion | non_qrs_rms_ratio | 0.968251 | 0.0386845 |
| E22_e6_subtype_class_fusion | qrs_band_ratio | 0.948041 | 0.512362 |
| E22_e6_subtype_class_fusion | qrs_visibility | 0.964349 | 0.0796063 |
| E22_e6_subtype_class_fusion | sqi_basSQI | 0.962864 | 0.0257122 |
| E22_e6_subtype_class_fusion | template_corr | 0.948185 | 0.0738977 |
| E23_e14_subtype_class_fusion | amplitude_entropy | 0.278298 | 0.0170358 |
| E23_e14_subtype_class_fusion | baseline_step | 0.967611 | 0.0130702 |
| E23_e14_subtype_class_fusion | contact_loss_win_ratio | 0.212816 | 0.00538362 |
| E23_e14_subtype_class_fusion | detector_agreement | 0.742259 | 0.257836 |
| E23_e14_subtype_class_fusion | flatline_ratio | 0.850138 | 0.0209313 |
| E23_e14_subtype_class_fusion | non_qrs_diff_p95 | 0.899881 | 0.183018 |
| E23_e14_subtype_class_fusion | non_qrs_rms_ratio | 0.967675 | 0.0396208 |
| E23_e14_subtype_class_fusion | qrs_band_ratio | 0.952008 | 0.513813 |
| E23_e14_subtype_class_fusion | qrs_visibility | 0.965555 | 0.0898251 |
| E23_e14_subtype_class_fusion | sqi_basSQI | 0.966677 | 0.0275414 |
| E23_e14_subtype_class_fusion | template_corr | 0.950375 | 0.0728886 |
| E24_e6_subtype_fusion_pairrank | amplitude_entropy | 0.195906 | 0.0185994 |
| E24_e6_subtype_fusion_pairrank | baseline_step | 0.968265 | 0.0119632 |
| E24_e6_subtype_fusion_pairrank | contact_loss_win_ratio | 0.201387 | 0.00541095 |
| E24_e6_subtype_fusion_pairrank | detector_agreement | 0.751763 | 0.258352 |
| E24_e6_subtype_fusion_pairrank | flatline_ratio | 0.843568 | 0.0219519 |
| E24_e6_subtype_fusion_pairrank | non_qrs_diff_p95 | 0.895326 | 0.197216 |
| E24_e6_subtype_fusion_pairrank | non_qrs_rms_ratio | 0.967659 | 0.0393961 |
| E24_e6_subtype_fusion_pairrank | qrs_band_ratio | 0.951976 | 0.499212 |
| E24_e6_subtype_fusion_pairrank | qrs_visibility | 0.964333 | 0.081036 |
| E24_e6_subtype_fusion_pairrank | sqi_basSQI | 0.96658 | 0.0238641 |
| E24_e6_subtype_fusion_pairrank | template_corr | 0.949598 | 0.0741149 |
