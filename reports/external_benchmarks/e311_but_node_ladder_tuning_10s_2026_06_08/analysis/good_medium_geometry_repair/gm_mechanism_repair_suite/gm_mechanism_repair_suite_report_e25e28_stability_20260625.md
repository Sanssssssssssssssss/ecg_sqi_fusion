# GM Mechanism Repair Suite

- Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Folds: `3`
- Epochs: `12`
- Seed: `20260921`
- Suite: `all`

## Clean Test Summary

| ('candidate', '') | ('acc', 'mean') | ('acc', 'std') | ('acc', 'max') | ('macro_f1', 'mean') | ('macro_f1', 'std') | ('macro_f1', 'max') | ('good_recall', 'mean') | ('good_recall', 'std') | ('good_recall', 'max') | ('medium_recall', 'mean') | ('medium_recall', 'std') | ('medium_recall', 'max') | ('bad_recall', 'mean') | ('bad_recall', 'std') | ('bad_recall', 'max') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E25_e24_lowalpha | 0.922726 | 0.0106438 | 0.930186 | 0.906164 | 0.0156208 | 0.918577 | 0.882048 | 0.0484358 | 0.912426 | 0.85338 | 0.0173509 | 0.87175 | 0.979923 | 0.00558942 | 0.986281 |
| E26_e24_lowlr | 0.921483 | 0.00790381 | 0.928199 | 0.905404 | 0.0124832 | 0.916389 | 0.913846 | 0.0168166 | 0.927957 | 0.843228 | 0.0175315 | 0.856153 | 0.969684 | 0.00172008 | 0.971149 |
| E27_e24_lowalpha_lowlr | 0.919082 | 0.00768981 | 0.92795 | 0.902271 | 0.0127513 | 0.916837 | 0.874593 | 0.0308272 | 0.907527 | 0.867868 | 0.0168862 | 0.887348 | 0.967404 | 0.00615275 | 0.974032 |
| E28_e24_softpair_lowalpha | 0.918833 | 0.00896174 | 0.924969 | 0.901167 | 0.0129163 | 0.912437 | 0.865621 | 0.0140294 | 0.87619 | 0.861965 | 0.0283677 | 0.881282 | 0.974726 | 0.00770383 | 0.981169 |

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
| E25_e24_lowalpha | 0 | 0 | 0.361919 | 81.8288 |
| E25_e24_lowalpha | 1 | 0 | 0.359271 | 80.5046 |
| E25_e24_lowalpha | 2 | 0 | 0.353566 | 85.0266 |
| E26_e24_lowlr | 0 | 0 | 0.349034 | 85.1421 |
| E26_e24_lowlr | 1 | 0 | 0.348911 | 81.5021 |
| E26_e24_lowlr | 2 | 0 | 0.340564 | 88.9737 |
| E27_e24_lowalpha_lowlr | 0 | 0 | 0.347733 | 86.3789 |
| E27_e24_lowalpha_lowlr | 1 | 0 | 0.348244 | 81.9809 |
| E27_e24_lowalpha_lowlr | 2 | 0 | 0.338819 | 89.7952 |
| E28_e24_softpair_lowalpha | 0 | 0 | 0.36028 | 82.9612 |
| E28_e24_softpair_lowalpha | 1 | 0 | 0.358364 | 81.2718 |
| E28_e24_softpair_lowalpha | 2 | 0 | 0.339686 | 91.0885 |

## Expert Usage

No expert usage rows.

## Decoded Factor Recovery

| candidate | feature | corr_all | mae |
| --- | --- | --- | --- |
| E25_e24_lowalpha | amplitude_entropy | 0.243422 | 0.0192724 |
| E25_e24_lowalpha | baseline_step | 0.969836 | 0.0132863 |
| E25_e24_lowalpha | contact_loss_win_ratio | 0.225331 | 0.00540763 |
| E25_e24_lowalpha | detector_agreement | 0.740353 | 0.258347 |
| E25_e24_lowalpha | flatline_ratio | 0.84539 | 0.022321 |
| E25_e24_lowalpha | non_qrs_diff_p95 | 0.899715 | 0.193426 |
| E25_e24_lowalpha | non_qrs_rms_ratio | 0.96876 | 0.0432405 |
| E25_e24_lowalpha | qrs_band_ratio | 0.954585 | 0.526457 |
| E25_e24_lowalpha | qrs_visibility | 0.967749 | 0.0842252 |
| E25_e24_lowalpha | sqi_basSQI | 0.96438 | 0.0274053 |
| E25_e24_lowalpha | template_corr | 0.949451 | 0.0746136 |
| E26_e24_lowlr | amplitude_entropy | 0.255323 | 0.0178433 |
| E26_e24_lowlr | baseline_step | 0.964385 | 0.0142989 |
| E26_e24_lowlr | contact_loss_win_ratio | 0.229406 | 0.00681365 |
| E26_e24_lowlr | detector_agreement | 0.756374 | 0.261711 |
| E26_e24_lowlr | flatline_ratio | 0.839238 | 0.0226052 |
| E26_e24_lowlr | non_qrs_diff_p95 | 0.888787 | 0.198872 |
| E26_e24_lowlr | non_qrs_rms_ratio | 0.964892 | 0.0439758 |
| E26_e24_lowlr | qrs_band_ratio | 0.946575 | 0.512677 |
| E26_e24_lowlr | qrs_visibility | 0.96087 | 0.086689 |
| E26_e24_lowlr | sqi_basSQI | 0.962262 | 0.0275533 |
| E26_e24_lowlr | template_corr | 0.946928 | 0.0748897 |
| E27_e24_lowalpha_lowlr | amplitude_entropy | 0.216981 | 0.0174402 |
| E27_e24_lowalpha_lowlr | baseline_step | 0.964512 | 0.0131664 |
| E27_e24_lowalpha_lowlr | contact_loss_win_ratio | 0.213283 | 0.00649237 |
| E27_e24_lowalpha_lowlr | detector_agreement | 0.75551 | 0.262226 |
| E27_e24_lowalpha_lowlr | flatline_ratio | 0.838724 | 0.0223443 |
| E27_e24_lowalpha_lowlr | non_qrs_diff_p95 | 0.893582 | 0.189581 |
| E27_e24_lowalpha_lowlr | non_qrs_rms_ratio | 0.964757 | 0.0415885 |
| E27_e24_lowalpha_lowlr | qrs_band_ratio | 0.945528 | 0.527858 |
| E27_e24_lowalpha_lowlr | qrs_visibility | 0.959072 | 0.0879765 |
| E27_e24_lowalpha_lowlr | sqi_basSQI | 0.961998 | 0.0249182 |
| E27_e24_lowalpha_lowlr | template_corr | 0.947654 | 0.0743276 |
| E28_e24_softpair_lowalpha | amplitude_entropy | 0.195209 | 0.0194146 |
| E28_e24_softpair_lowalpha | baseline_step | 0.967666 | 0.0132128 |
| E28_e24_softpair_lowalpha | contact_loss_win_ratio | 0.238361 | 0.00556148 |
| E28_e24_softpair_lowalpha | detector_agreement | 0.745696 | 0.259267 |
| E28_e24_softpair_lowalpha | flatline_ratio | 0.850514 | 0.021935 |
| E28_e24_softpair_lowalpha | non_qrs_diff_p95 | 0.884993 | 0.205921 |
| E28_e24_softpair_lowalpha | non_qrs_rms_ratio | 0.964218 | 0.0414297 |
| E28_e24_softpair_lowalpha | qrs_band_ratio | 0.95074 | 0.53262 |
| E28_e24_softpair_lowalpha | qrs_visibility | 0.964203 | 0.0856494 |
| E28_e24_softpair_lowalpha | sqi_basSQI | 0.964508 | 0.0278585 |
| E28_e24_softpair_lowalpha | template_corr | 0.949842 | 0.0759941 |
