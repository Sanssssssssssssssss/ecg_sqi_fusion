# GM Mechanism Repair Suite

- Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Folds: `3`
- Epochs: `10`
- Seed: `20260901`
- Suite: `all`

## Clean Test Summary

| ('candidate', '') | ('acc', 'mean') | ('acc', 'std') | ('acc', 'max') | ('macro_f1', 'mean') | ('macro_f1', 'std') | ('macro_f1', 'max') | ('good_recall', 'mean') | ('good_recall', 'std') | ('good_recall', 'max') | ('medium_recall', 'mean') | ('medium_recall', 'std') | ('medium_recall', 'max') | ('bad_recall', 'mean') | ('bad_recall', 'std') | ('bad_recall', 'max') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E10_e6_medguard | 0.917177 | 0.0074764 | 0.924969 | 0.899133 | 0.012269 | 0.912847 | 0.866349 | 0.0646682 | 0.94086 | 0.857489 | 0.0029701 | 0.860892 | 0.972449 | 0.0124996 | 0.981872 |
| E11_e6_lowgain | 0.915521 | 0.00267614 | 0.918489 | 0.897354 | 0.00171311 | 0.898376 | 0.858777 | 0.0162527 | 0.877381 | 0.848165 | 0.0127189 | 0.857886 | 0.97911 | 0.00355142 | 0.982852 |
| E12_e6_pairrank_only | 0.919496 | 0.00710467 | 0.927453 | 0.902442 | 0.0105034 | 0.914561 | 0.888113 | 0.0359192 | 0.916129 | 0.852856 | 0.00840646 | 0.862218 | 0.971127 | 0.0124921 | 0.983341 |
| E13_e6_boundary_aux_stronger | 0.919662 | 0.00467191 | 0.922733 | 0.90214 | 0.00655819 | 0.908111 | 0.847494 | 0.0173876 | 0.865476 | 0.878839 | 0.0160512 | 0.89688 | 0.974661 | 0.0018702 | 0.976482 |
| E14_e6_medguard_lowgain | 0.921401 | 0.00660212 | 0.927702 | 0.904896 | 0.00960305 | 0.915255 | 0.890435 | 0.0113754 | 0.898225 | 0.860098 | 0.0184977 | 0.880416 | 0.970151 | 0.00785694 | 0.977462 |

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
| E10_e6_medguard | 0 | 0 | 0.340909 | 94.3923 |
| E10_e6_medguard | 1 | 0 | 0.357559 | 81.9483 |
| E10_e6_medguard | 2 | 0 | 0.346312 | 89.1861 |
| E11_e6_lowgain | 0 | 0 | 0.347265 | 92.796 |
| E11_e6_lowgain | 1 | 0 | 0.337171 | 90.9535 |
| E11_e6_lowgain | 2 | 0 | 0.349089 | 88.7346 |
| E12_e6_pairrank_only | 0 | 0 | 0.353477 | 89.4388 |
| E12_e6_pairrank_only | 1 | 0 | 0.362041 | 81.2661 |
| E12_e6_pairrank_only | 2 | 0 | 0.345386 | 89.7247 |
| E13_e6_boundary_aux_stronger | 0 | 0 | 0.345579 | 93.2112 |
| E13_e6_boundary_aux_stronger | 1 | 0 | 0.343665 | 87.8211 |
| E13_e6_boundary_aux_stronger | 2 | 0 | 0.348241 | 88.3171 |
| E14_e6_medguard_lowgain | 0 | 0 | 0.353468 | 88.5523 |
| E14_e6_medguard_lowgain | 1 | 0 | 0.359285 | 81.642 |
| E14_e6_medguard_lowgain | 2 | 0 | 0.348576 | 88.8072 |

## Expert Usage

No expert usage rows.

## Decoded Factor Recovery

| candidate | feature | corr_all | mae |
| --- | --- | --- | --- |
| E10_e6_medguard | amplitude_entropy | 0.290896 | 0.0190431 |
| E10_e6_medguard | baseline_step | 0.963393 | 0.0146732 |
| E10_e6_medguard | contact_loss_win_ratio | 0.261716 | 0.00694769 |
| E10_e6_medguard | detector_agreement | 0.740118 | 0.259128 |
| E10_e6_medguard | flatline_ratio | 0.84936 | 0.0212684 |
| E10_e6_medguard | non_qrs_diff_p95 | 0.876601 | 0.207877 |
| E10_e6_medguard | non_qrs_rms_ratio | 0.956427 | 0.0471783 |
| E10_e6_medguard | qrs_band_ratio | 0.949509 | 0.515763 |
| E10_e6_medguard | qrs_visibility | 0.96066 | 0.0890038 |
| E10_e6_medguard | sqi_basSQI | 0.958846 | 0.0256983 |
| E10_e6_medguard | template_corr | 0.948332 | 0.0742572 |
| E11_e6_lowgain | amplitude_entropy | 0.245755 | 0.0199216 |
| E11_e6_lowgain | baseline_step | 0.957151 | 0.0147523 |
| E11_e6_lowgain | contact_loss_win_ratio | 0.262665 | 0.008691 |
| E11_e6_lowgain | detector_agreement | 0.728251 | 0.262226 |
| E11_e6_lowgain | flatline_ratio | 0.845356 | 0.0263384 |
| E11_e6_lowgain | non_qrs_diff_p95 | 0.865741 | 0.224397 |
| E11_e6_lowgain | non_qrs_rms_ratio | 0.958207 | 0.0544474 |
| E11_e6_lowgain | qrs_band_ratio | 0.945068 | 0.54384 |
| E11_e6_lowgain | qrs_visibility | 0.96062 | 0.0902291 |
| E11_e6_lowgain | sqi_basSQI | 0.951623 | 0.0307593 |
| E11_e6_lowgain | template_corr | 0.945366 | 0.0756736 |
| E12_e6_pairrank_only | amplitude_entropy | 0.32416 | 0.0174754 |
| E12_e6_pairrank_only | baseline_step | 0.967587 | 0.0128971 |
| E12_e6_pairrank_only | contact_loss_win_ratio | 0.270665 | 0.00621458 |
| E12_e6_pairrank_only | detector_agreement | 0.739696 | 0.257796 |
| E12_e6_pairrank_only | flatline_ratio | 0.855401 | 0.0207674 |
| E12_e6_pairrank_only | non_qrs_diff_p95 | 0.899868 | 0.192771 |
| E12_e6_pairrank_only | non_qrs_rms_ratio | 0.963855 | 0.0404589 |
| E12_e6_pairrank_only | qrs_band_ratio | 0.950204 | 0.51377 |
| E12_e6_pairrank_only | qrs_visibility | 0.961768 | 0.0860135 |
| E12_e6_pairrank_only | sqi_basSQI | 0.965149 | 0.0241833 |
| E12_e6_pairrank_only | template_corr | 0.94913 | 0.0738544 |
| E13_e6_boundary_aux_stronger | amplitude_entropy | 0.274264 | 0.0177694 |
| E13_e6_boundary_aux_stronger | baseline_step | 0.959769 | 0.0153164 |
| E13_e6_boundary_aux_stronger | contact_loss_win_ratio | 0.276475 | 0.00813661 |
| E13_e6_boundary_aux_stronger | detector_agreement | 0.730076 | 0.261391 |
| E13_e6_boundary_aux_stronger | flatline_ratio | 0.844928 | 0.024569 |
| E13_e6_boundary_aux_stronger | non_qrs_diff_p95 | 0.874652 | 0.219922 |
| E13_e6_boundary_aux_stronger | non_qrs_rms_ratio | 0.962916 | 0.047565 |
| E13_e6_boundary_aux_stronger | qrs_band_ratio | 0.94434 | 0.543331 |
| E13_e6_boundary_aux_stronger | qrs_visibility | 0.960047 | 0.0911014 |
| E13_e6_boundary_aux_stronger | sqi_basSQI | 0.95328 | 0.0299117 |
| E13_e6_boundary_aux_stronger | template_corr | 0.946911 | 0.075263 |
| E14_e6_medguard_lowgain | amplitude_entropy | 0.296392 | 0.0175097 |
| E14_e6_medguard_lowgain | baseline_step | 0.968838 | 0.0123282 |
| E14_e6_medguard_lowgain | contact_loss_win_ratio | 0.24544 | 0.00602894 |
| E14_e6_medguard_lowgain | detector_agreement | 0.749528 | 0.256377 |
| E14_e6_medguard_lowgain | flatline_ratio | 0.852858 | 0.0206418 |
| E14_e6_medguard_lowgain | non_qrs_diff_p95 | 0.899394 | 0.194372 |

_Showing 50 of 55 rows._
