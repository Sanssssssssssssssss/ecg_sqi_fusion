# GM Mechanism Repair Suite

- Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Folds: `3`
- Epochs: `10`
- Seed: `20260901`
- Suite: `posthoc`

## Clean Test Summary

| ('candidate', '') | ('acc', 'mean') | ('acc', 'std') | ('acc', 'max') | ('macro_f1', 'mean') | ('macro_f1', 'std') | ('macro_f1', 'max') | ('good_recall', 'mean') | ('good_recall', 'std') | ('good_recall', 'max') | ('medium_recall', 'mean') | ('medium_recall', 'std') | ('medium_recall', 'max') | ('bad_recall', 'mean') | ('bad_recall', 'std') | ('bad_recall', 'max') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4_v112_lowaux_lr15e4 | 0.918917 | 0.0028864 | 0.922236 | 0.901144 | 0.00598136 | 0.90783 | 0.859228 | 0.00641112 | 0.865089 | 0.864086 | 0.0221498 | 0.889081 | 0.976414 | 0.00578135 | 0.982852 |
| E5_factor_contract_only | 0.91991 | 0.00924849 | 0.929689 | 0.902502 | 0.013234 | 0.917211 | 0.863471 | 0.0348022 | 0.888172 | 0.864026 | 0.0273055 | 0.889948 | 0.976273 | 0.00485608 | 0.981872 |
| E6_factor_fused_gm | 0.921981 | 0.00406493 | 0.925466 | 0.905553 | 0.00743209 | 0.913506 | 0.894739 | 0.0444095 | 0.944086 | 0.853715 | 0.00136363 | 0.855286 | 0.972295 | 0.0121524 | 0.979683 |
| E7_family_moe_condsubtype | 0.913285 | 0.00150428 | 0.914264 | 0.894334 | 0.00135207 | 0.89566 | 0.856891 | 0.012192 | 0.870968 | 0.851413 | 0.00214994 | 0.853893 | 0.973255 | 0.00519629 | 0.977701 |
| E8_pairrank_hardsampler | 0.916018 | 0.00800011 | 0.924969 | 0.897255 | 0.0124017 | 0.911502 | 0.878287 | 0.0290241 | 0.903226 | 0.83761 | 0.0306442 | 0.866551 | 0.977224 | 0.00625039 | 0.981872 |
| E9_beat_background_tokens | 0.863757 | 0.00955046 | 0.874286 | 0.831859 | 0.0159808 | 0.850005 | 0.878617 | 0.0975472 | 0.970414 | 0.691933 | 0.0799249 | 0.753281 | 0.956356 | 0.00222459 | 0.958269 |

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
| E4_v112_lowaux_lr15e4 | 0 | 0 | 0.386297 | 79.4971 |
| E4_v112_lowaux_lr15e4 | 1 | 0 | 0.389838 | 108.244 |
| E4_v112_lowaux_lr15e4 | 2 | 0 | 0.386402 | 72.749 |
| E5_factor_contract_only | 0 | 0 | 0.349081 | 92.319 |
| E5_factor_contract_only | 1 | 0 | 0.362608 | 81.2042 |
| E5_factor_contract_only | 2 | 0 | 0.351872 | 88.5872 |
| E6_factor_fused_gm | 0 | 0 | 0.345248 | 94.0527 |
| E6_factor_fused_gm | 1 | 0 | 0.358237 | 81.6875 |
| E6_factor_fused_gm | 2 | 0 | 0.347874 | 88.7597 |
| E7_family_moe_condsubtype | 0 | 0 | 0.346371 | 93.0922 |
| E7_family_moe_condsubtype | 1 | 0 | 0.340295 | 88.1776 |
| E7_family_moe_condsubtype | 2 | 0 | 0.349393 | 86.6344 |
| E8_pairrank_hardsampler | 0 | 0 | 0.336039 | 95.7861 |
| E8_pairrank_hardsampler | 1 | 0 | 0.359352 | 82.0527 |
| E8_pairrank_hardsampler | 2 | 0 | 0.349814 | 86.4911 |
| E9_beat_background_tokens | 0 | 0 | 0.331632 | 89.8708 |
| E9_beat_background_tokens | 1 | 0 | 0.327702 | 93.0239 |
| E9_beat_background_tokens | 2 | 0 | 0.339915 | 85.0626 |

## Expert Usage

| candidate | fold | seed_index | expert | mean_prob | usage_gt_0p5 |
| --- | --- | --- | --- | --- | --- |
| E7_family_moe_condsubtype | 0 | 0 | lowqrs_template | 0.656507 | 0.764969 |
| E7_family_moe_condsubtype | 0 | 0 | baseline_contact_detail | 0.560777 | 0.718012 |
| E7_family_moe_condsubtype | 0 | 0 | generic_overlap | 0.532113 | 0.688199 |
| E7_family_moe_condsubtype | 1 | 0 | lowqrs_template | 0.567094 | 0.697888 |
| E7_family_moe_condsubtype | 1 | 0 | baseline_contact_detail | 0.498719 | 0.632547 |
| E7_family_moe_condsubtype | 1 | 0 | generic_overlap | 0.496626 | 0.630559 |
| E7_family_moe_condsubtype | 2 | 0 | lowqrs_template | 0.553919 | 0.738569 |
| E7_family_moe_condsubtype | 2 | 0 | baseline_contact_detail | 0.578107 | 0.736332 |
| E7_family_moe_condsubtype | 2 | 0 | generic_overlap | 0.532124 | 0.716203 |
| E8_pairrank_hardsampler | 0 | 0 | lowqrs_template | 0.577899 | 0.716025 |
| E8_pairrank_hardsampler | 0 | 0 | baseline_contact_detail | 0.486933 | 0.627578 |
| E8_pairrank_hardsampler | 0 | 0 | generic_overlap | 0.460528 | 0.566211 |
| E8_pairrank_hardsampler | 1 | 0 | lowqrs_template | 0.604732 | 0.721242 |
| E8_pairrank_hardsampler | 1 | 0 | baseline_contact_detail | 0.537133 | 0.69118 |
| E8_pairrank_hardsampler | 1 | 0 | generic_overlap | 0.547454 | 0.67528 |
| E8_pairrank_hardsampler | 2 | 0 | lowqrs_template | 0.554396 | 0.741799 |
| E8_pairrank_hardsampler | 2 | 0 | baseline_contact_detail | 0.568695 | 0.732853 |
| E8_pairrank_hardsampler | 2 | 0 | generic_overlap | 0.529296 | 0.712227 |
| E9_beat_background_tokens | 0 | 0 | lowqrs_template | 0.464707 | 0.556025 |
| E9_beat_background_tokens | 0 | 0 | baseline_contact_detail | 0.399637 | 0.47354 |
| E9_beat_background_tokens | 0 | 0 | generic_overlap | 0.369628 | 0.435528 |
| E9_beat_background_tokens | 1 | 0 | lowqrs_template | 0.468817 | 0.587826 |
| E9_beat_background_tokens | 1 | 0 | baseline_contact_detail | 0.427656 | 0.494161 |
| E9_beat_background_tokens | 1 | 0 | generic_overlap | 0.439064 | 0.504596 |
| E9_beat_background_tokens | 2 | 0 | lowqrs_template | 0.447021 | 0.574553 |
| E9_beat_background_tokens | 2 | 0 | baseline_contact_detail | 0.427058 | 0.561133 |
| E9_beat_background_tokens | 2 | 0 | generic_overlap | 0.457816 | 0.564612 |

## Decoded Factor Recovery

| candidate | feature | corr_all | mae |
| --- | --- | --- | --- |
| E4_v112_lowaux_lr15e4 | amplitude_entropy | 0.26664 | 0.614568 |
| E4_v112_lowaux_lr15e4 | baseline_step | 0.954146 | 0.185234 |
| E4_v112_lowaux_lr15e4 | contact_loss_win_ratio | 0.183002 | 5.65866 |
| E4_v112_lowaux_lr15e4 | detector_agreement | 0.817142 | 0.32919 |
| E4_v112_lowaux_lr15e4 | flatline_ratio | 0.808065 | 3.18413 |
| E4_v112_lowaux_lr15e4 | non_qrs_diff_p95 | 0.881361 | 0.333744 |
| E4_v112_lowaux_lr15e4 | non_qrs_rms_ratio | 0.900219 | 1.56455 |
| E4_v112_lowaux_lr15e4 | qrs_band_ratio | 0 | 4.90499 |
| E4_v112_lowaux_lr15e4 | qrs_visibility | 0.398445 | 3.57707 |
| E4_v112_lowaux_lr15e4 | sqi_basSQI | 0.95416 | 0.52549 |
| E4_v112_lowaux_lr15e4 | template_corr | 0.944622 | 1.14738 |
| E5_factor_contract_only | amplitude_entropy | 0.273223 | 0.0193066 |
| E5_factor_contract_only | baseline_step | 0.962566 | 0.0141003 |
| E5_factor_contract_only | contact_loss_win_ratio | 0.295834 | 0.00690674 |
| E5_factor_contract_only | detector_agreement | 0.737045 | 0.259571 |
| E5_factor_contract_only | flatline_ratio | 0.856757 | 0.0233051 |
| E5_factor_contract_only | non_qrs_diff_p95 | 0.876656 | 0.215849 |
| E5_factor_contract_only | non_qrs_rms_ratio | 0.959362 | 0.050406 |
| E5_factor_contract_only | qrs_band_ratio | 0.949308 | 0.53558 |
| E5_factor_contract_only | qrs_visibility | 0.960838 | 0.0948774 |
| E5_factor_contract_only | sqi_basSQI | 0.960409 | 0.0289505 |
| E5_factor_contract_only | template_corr | 0.947782 | 0.0741199 |
| E6_factor_fused_gm | amplitude_entropy | 0.294021 | 0.0187611 |
| E6_factor_fused_gm | baseline_step | 0.962775 | 0.0147288 |
| E6_factor_fused_gm | contact_loss_win_ratio | 0.269286 | 0.00743819 |
| E6_factor_fused_gm | detector_agreement | 0.737708 | 0.258794 |
| E6_factor_fused_gm | flatline_ratio | 0.852154 | 0.0237884 |
| E6_factor_fused_gm | non_qrs_diff_p95 | 0.878447 | 0.208599 |
| E6_factor_fused_gm | non_qrs_rms_ratio | 0.9615 | 0.0473941 |
| E6_factor_fused_gm | qrs_band_ratio | 0.947411 | 0.532613 |
| E6_factor_fused_gm | qrs_visibility | 0.959783 | 0.0897455 |
| E6_factor_fused_gm | sqi_basSQI | 0.955477 | 0.0282714 |
| E6_factor_fused_gm | template_corr | 0.947422 | 0.0744636 |
| E7_family_moe_condsubtype | amplitude_entropy | 0.288227 | 0.0202266 |
| E7_family_moe_condsubtype | baseline_step | 0.949122 | 0.0146668 |
| E7_family_moe_condsubtype | contact_loss_win_ratio | 0.148553 | 0.00867378 |
| E7_family_moe_condsubtype | detector_agreement | 0.714164 | 0.264027 |
| E7_family_moe_condsubtype | flatline_ratio | 0.846343 | 0.0225587 |
| E7_family_moe_condsubtype | non_qrs_diff_p95 | 0.870673 | 0.209956 |
| E7_family_moe_condsubtype | non_qrs_rms_ratio | 0.949285 | 0.047563 |
| E7_family_moe_condsubtype | qrs_band_ratio | 0.932831 | 0.568718 |
| E7_family_moe_condsubtype | qrs_visibility | 0.957093 | 0.0929034 |
| E7_family_moe_condsubtype | sqi_basSQI | 0.947013 | 0.029592 |
| E7_family_moe_condsubtype | template_corr | 0.943327 | 0.0769305 |
| E8_pairrank_hardsampler | amplitude_entropy | 0.208179 | 0.0209769 |
| E8_pairrank_hardsampler | baseline_step | 0.957642 | 0.0137948 |
| E8_pairrank_hardsampler | contact_loss_win_ratio | 0.208459 | 0.00837799 |
| E8_pairrank_hardsampler | detector_agreement | 0.73606 | 0.261509 |
| E8_pairrank_hardsampler | flatline_ratio | 0.852794 | 0.0227953 |
| E8_pairrank_hardsampler | non_qrs_diff_p95 | 0.875943 | 0.208031 |

_Showing 50 of 66 rows._
