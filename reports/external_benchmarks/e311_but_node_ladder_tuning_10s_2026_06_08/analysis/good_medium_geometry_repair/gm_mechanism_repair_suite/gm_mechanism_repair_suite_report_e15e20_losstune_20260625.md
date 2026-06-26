# GM Mechanism Repair Suite

- Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Folds: `3`
- Epochs: `12`
- Seed: `20260901`
- Suite: `all`

## Clean Test Summary

| ('candidate', '') | ('acc', 'mean') | ('acc', 'std') | ('acc', 'max') | ('macro_f1', 'mean') | ('macro_f1', 'std') | ('macro_f1', 'max') | ('good_recall', 'mean') | ('good_recall', 'std') | ('good_recall', 'max') | ('medium_recall', 'mean') | ('medium_recall', 'std') | ('medium_recall', 'max') | ('bad_recall', 'mean') | ('bad_recall', 'std') | ('bad_recall', 'max') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E15_e6_classheavy_lowaux | 0.920573 | 0.00164601 | 0.921968 | 0.903566 | 0.00374791 | 0.907527 | 0.876837 | 0.011207 | 0.889247 | 0.860963 | 0.021372 | 0.884749 | 0.973608 | 0.0138914 | 0.983647 |
| E16_e14_classheavy_lowaux | 0.92165 | 0.0029722 | 0.924969 | 0.904851 | 0.00596505 | 0.911519 | 0.883797 | 0.0295466 | 0.914793 | 0.854335 | 0.0224653 | 0.877816 | 0.976943 | 0.00512343 | 0.982852 |
| E17_e6_focal_gm | 0.919496 | 0.00418318 | 0.924224 | 0.902764 | 0.00815967 | 0.912101 | 0.890513 | 0.0476364 | 0.94086 | 0.851704 | 0.00468383 | 0.856153 | 0.970332 | 0.0145268 | 0.985629 |
| E18_e6_smooth_ce | 0.918751 | 0.00613118 | 0.925217 | 0.901078 | 0.0101156 | 0.911887 | 0.878496 | 0.0340656 | 0.908876 | 0.850248 | 0.0277864 | 0.882149 | 0.975561 | 0.00805238 | 0.984811 |
| E19_e6_nolocal_classfocus | 0.912953 | 0.00685526 | 0.917516 | 0.894165 | 0.0122084 | 0.902585 | 0.855318 | 0.07057 | 0.919527 | 0.854732 | 0.0270686 | 0.882765 | 0.97159 | 0.00545995 | 0.977846 |
| E20_e6_lowlr_balanced | 0.916514 | 0.00611395 | 0.923478 | 0.898298 | 0.0100141 | 0.909663 | 0.877002 | 0.0483944 | 0.929032 | 0.837557 | 0.00630357 | 0.84252 | 0.978382 | 0.00679716 | 0.983341 |

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
| E15_e6_classheavy_lowaux | 0 | 0 | 0.339421 | 90.1317 |
| E15_e6_classheavy_lowaux | 1 | 0 | 0.337294 | 89.3973 |
| E15_e6_classheavy_lowaux | 2 | 0 | 0.339134 | 87.8797 |
| E16_e14_classheavy_lowaux | 0 | 0 | 0.339142 | 86.4191 |
| E16_e14_classheavy_lowaux | 1 | 0 | 0.349783 | 81.9811 |
| E16_e14_classheavy_lowaux | 2 | 0 | 0.338123 | 88.9076 |
| E17_e6_focal_gm | 0 | 0 | 0.35384 | 87.4857 |
| E17_e6_focal_gm | 1 | 0 | 0.359155 | 81.3026 |
| E17_e6_focal_gm | 2 | 0 | 0.356332 | 84.3598 |
| E18_e6_smooth_ce | 0 | 0 | 0.340699 | 91.9911 |
| E18_e6_smooth_ce | 1 | 0 | 0.347539 | 86.204 |
| E18_e6_smooth_ce | 2 | 0 | 0.340012 | 89.6267 |
| E19_e6_nolocal_classfocus | 0 | 0 | 0.265353 | 185.82 |
| E19_e6_nolocal_classfocus | 1 | 0 | 0.351233 | 263.017 |
| E19_e6_nolocal_classfocus | 2 | 0 | 0.360759 | 310.25 |
| E20_e6_lowlr_balanced | 0 | 0 | 0.335491 | 95.1988 |
| E20_e6_lowlr_balanced | 1 | 0 | 0.334191 | 90.3272 |
| E20_e6_lowlr_balanced | 2 | 0 | 0.337183 | 88.7244 |

## Expert Usage

No expert usage rows.

## Decoded Factor Recovery

| candidate | feature | corr_all | mae |
| --- | --- | --- | --- |
| E15_e6_classheavy_lowaux | amplitude_entropy | 0.280841 | 0.0177602 |
| E15_e6_classheavy_lowaux | baseline_step | 0.954553 | 0.0146975 |
| E15_e6_classheavy_lowaux | contact_loss_win_ratio | 0.226052 | 0.00633983 |
| E15_e6_classheavy_lowaux | detector_agreement | 0.772849 | 0.25212 |
| E15_e6_classheavy_lowaux | flatline_ratio | 0.85614 | 0.0205251 |
| E15_e6_classheavy_lowaux | non_qrs_diff_p95 | 0.864145 | 0.227239 |
| E15_e6_classheavy_lowaux | non_qrs_rms_ratio | 0.942639 | 0.0522636 |
| E15_e6_classheavy_lowaux | qrs_band_ratio | 0.942908 | 0.552045 |
| E15_e6_classheavy_lowaux | qrs_visibility | 0.959844 | 0.0869115 |
| E15_e6_classheavy_lowaux | sqi_basSQI | 0.958119 | 0.0266403 |
| E15_e6_classheavy_lowaux | template_corr | 0.949518 | 0.0732973 |
| E16_e14_classheavy_lowaux | amplitude_entropy | 0.263668 | 0.0185176 |
| E16_e14_classheavy_lowaux | baseline_step | 0.95782 | 0.0126702 |
| E16_e14_classheavy_lowaux | contact_loss_win_ratio | 0.256692 | 0.00669388 |
| E16_e14_classheavy_lowaux | detector_agreement | 0.781474 | 0.250051 |
| E16_e14_classheavy_lowaux | flatline_ratio | 0.852763 | 0.0204554 |
| E16_e14_classheavy_lowaux | non_qrs_diff_p95 | 0.87377 | 0.218263 |
| E16_e14_classheavy_lowaux | non_qrs_rms_ratio | 0.944389 | 0.0502287 |
| E16_e14_classheavy_lowaux | qrs_band_ratio | 0.943779 | 0.522943 |
| E16_e14_classheavy_lowaux | qrs_visibility | 0.960079 | 0.0920773 |
| E16_e14_classheavy_lowaux | sqi_basSQI | 0.960102 | 0.024705 |
| E16_e14_classheavy_lowaux | template_corr | 0.949252 | 0.0748174 |
| E17_e6_focal_gm | amplitude_entropy | 0.338994 | 0.0175758 |
| E17_e6_focal_gm | baseline_step | 0.976796 | 0.00925156 |
| E17_e6_focal_gm | contact_loss_win_ratio | 0.250784 | 0.00545389 |
| E17_e6_focal_gm | detector_agreement | 0.758556 | 0.249847 |
| E17_e6_focal_gm | flatline_ratio | 0.85443 | 0.0201545 |
| E17_e6_focal_gm | non_qrs_diff_p95 | 0.918309 | 0.176896 |
| E17_e6_focal_gm | non_qrs_rms_ratio | 0.970725 | 0.0405948 |
| E17_e6_focal_gm | qrs_band_ratio | 0.958783 | 0.479797 |
| E17_e6_focal_gm | qrs_visibility | 0.96518 | 0.0859307 |
| E17_e6_focal_gm | sqi_basSQI | 0.972691 | 0.0207453 |
| E17_e6_focal_gm | template_corr | 0.94864 | 0.0758002 |
| E18_e6_smooth_ce | amplitude_entropy | 0.282762 | 0.0189 |
| E18_e6_smooth_ce | baseline_step | 0.957925 | 0.013787 |
| E18_e6_smooth_ce | contact_loss_win_ratio | 0.285752 | 0.00624321 |
| E18_e6_smooth_ce | detector_agreement | 0.757417 | 0.256887 |
| E18_e6_smooth_ce | flatline_ratio | 0.858673 | 0.020102 |
| E18_e6_smooth_ce | non_qrs_diff_p95 | 0.885026 | 0.208333 |
| E18_e6_smooth_ce | non_qrs_rms_ratio | 0.957112 | 0.0452512 |
| E18_e6_smooth_ce | qrs_band_ratio | 0.945484 | 0.531972 |
| E18_e6_smooth_ce | qrs_visibility | 0.960225 | 0.0856259 |
| E18_e6_smooth_ce | sqi_basSQI | 0.959474 | 0.0248926 |
| E18_e6_smooth_ce | template_corr | 0.94929 | 0.0734113 |
| E19_e6_nolocal_classfocus | amplitude_entropy | 0.269281 | 0.0180331 |
| E19_e6_nolocal_classfocus | baseline_step | 0.954361 | 0.0135274 |
| E19_e6_nolocal_classfocus | contact_loss_win_ratio | 0.204027 | 0.00677761 |
| E19_e6_nolocal_classfocus | detector_agreement | 0.807768 | 0.217627 |
| E19_e6_nolocal_classfocus | flatline_ratio | 0.84073 | 0.0218615 |
| E19_e6_nolocal_classfocus | non_qrs_diff_p95 | 0.853619 | 0.235148 |

_Showing 50 of 66 rows._
