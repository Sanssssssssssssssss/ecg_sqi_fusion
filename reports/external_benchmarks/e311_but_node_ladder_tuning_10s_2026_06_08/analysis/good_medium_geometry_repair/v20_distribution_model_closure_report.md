# V20 Distribution-Matching Closure Report

Generated: 2026-06-21

## Scope

This report closes the current loop:

1. Fit PTB synthetic poor/bad waveform distribution toward BUT natural poor/bad using waveform-computable SQI/physiology features.
2. Train waveform-only Event/Mechanism SQI Query Conformer candidates from scratch on the new V20 protocol.
3. Evaluate cross-dataset transfer without using SQI/PCA/MLP/tree features as inference inputs.

All code and outputs are under external benchmark output/report areas. No `src/sqi_pipeline` changes are required by this loop.

## V20 Distribution Result

Protocol:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\protocol_v20_pc3000_s20260621`

Audit report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\ptb_bad_distribution_fit_audit.md`

Version comparison, robust-z feature gap:

| version | mean_abs_gap | median_abs_gap | max_abs_gap |
| --- | ---: | ---: | ---: |
| v15 | 1.0187 | 0.0815 | 9.6412 |
| v17 | 0.9042 | 0.4270 | 4.1846 |
| v18 | 0.9179 | 0.4181 | 4.7824 |
| v19 | 1.3023 | 0.7809 | 5.6194 |
| v20 | 0.3107 | 0.0841 | 1.0617 |

V20 is the first version that simultaneously keeps QRS morphology visible, avoids the V15 30-45 Hz blow-up, and matches detector-agreement failure by creating too-many-peak / unstable-template morphology rather than pure flatline or pure noise.

Key V20 residual gaps:

| feature | robust_z_gap | interpretation |
| --- | ---: | --- |
| detector_agreement | 0.000 | matched at median; detector failure is now reproduced |
| band_30_45 | -1.062 | still slightly too little high-frequency tail |
| amplitude_entropy | -0.915 | still too low entropy / diversity |
| low_amp_ratio | -0.607 | still less low-amplitude occupancy than BUT |
| sqi_basSQI | -0.419 | baseline spectrum not fully matched |
| qrs_visibility | +0.351 | synthetic QRS still a little too visible |

Peak mechanism audit:

| source | peak_count median/IQR | rr_cv median/IQR | peak_amp_p90 median/IQR | template_consistency median/IQR | detector_agreement median/IQR |
| --- | --- | --- | --- | --- | --- |
| BUT target | 25 / 2 | 0.328 / 0.077 | 2.348 / 0.607 | 0.192 / 0.143 | 0.000 / 0.000 |
| PTB generated | 24 / 3 | 0.356 / 0.130 | 2.162 / 0.278 | 0.184 / 0.078 | 0.000 / 0.167 |

Conclusion: BUT bad detector failure is mainly a "too many plausible peaks plus unstable intervals/templates" problem, not a "too few peaks" problem. V20 now encodes that mechanism.

## Distribution Figures

V20 distribution audit:

- subtype mix: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_01_bad_subtype_mix.png`
- feature boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_02_feature_boxplots.png`
- feature gap bars: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_03_feature_gap_bars.png`
- PCA overlay: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_05_feature_pca_overlay.png`
- peak mechanism: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_06_peak_mechanism.png`

Paper-style SQI evidence chain:

- Fig. 13 SQI domain shift: `E:\GPTProject2\ecg\reports\sqi_paper_aligned\images\fig_13_sqi_domain_shift.png`
- Fig. 12 fSQI mechanism: `E:\GPTProject2\ecg\reports\sqi_paper_aligned\images\fig_12_fsqi_mechanism.png`
- Fig. 14 basSQI domain shift: `E:\GPTProject2\ecg\reports\sqi_paper_aligned\images\fig_14_bassqi_domain_shift.png`

Supplemental quantitative files:

- per-SQI domain AUC: `E:\GPTProject2\ecg\outputs\sqi_supplemental\existing_seed0\final_claims\per_sqi_domain_auc.csv`
- cross-domain AUC matrix: `E:\GPTProject2\ecg\outputs\sqi_supplemental\existing_seed0\final_claims\cross_domain_auc_matrix.csv`
- fSQI linear/RBF comparison: `E:\GPTProject2\ecg\outputs\sqi_supplemental\existing_seed0\final_claims\fsqi_linear_vs_rbf.csv`
- strict five-SQI paper quintuplet rank: `E:\GPTProject2\ecg\outputs\sqi_supplemental\existing_seed0\final_claims\table6_paper_quintuplet_rank.csv`

The paper-selected quintuplet ranks 6/21 among strict five-SQI subsets, with a validation accuracy gap of 0.862 pp from the best five-SQI subset. This supports the "flat plateau, not unique failure" interpretation.

## Model Input and Architecture

Training/evaluation runner:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_event_factorized_cross_dataset_top3.py`

Model base:

`EventFactorizedSQIConformer`, used here as the mechanism-query Conformer family.

Inference input:

- ECG waveform-derived channels only.
- No SQI vector, PCA coordinate, KNN/atlas purity, MLP/tree output, or route/rule artifact enters the model input at inference.

Training-only teacher/diagnostic targets:

- `qrs_visibility`
- `detector_agreement`
- `baseline_step`
- `flatline_ratio`
- `sqi_basSQI`
- `non_qrs_diff_p95`
- `qrs_band_ratio`
- `template_corr`
- `amplitude_entropy`
- `contact_loss_win_ratio`

Mechanism structure:

- high-resolution waveform stem for QRS/contact/reset/detail evidence;
- context tokens for window-level rhythm and morphology;
- SQI/factor query tokens;
- local supervision heads for qrs/baseline/contact/flatline/detail maps;
- artifact auxiliary head separated from diagnostic bad;
- hierarchical quality probability:
  - `P(bad)=b`
  - `P(medium)=(1-b)m`
  - `P(good)=(1-b)(1-m)`

Candidates trained from scratch:

- `E4_query_highres_local_art`: query + high-res + local supervision + artifact aux.
- `P2_ecg_beat_rhythm_mask`: same mechanism model with ECG-aware masked pretraining before classification.
- Additional architecture controls:
  - `E1_query_only`: query tokens without high-res/local/artifact heads.
  - `E2_query_highres`: query tokens plus high-res fusion, without local/artifact aux.

## V20 Cross-Dataset Training Results

Report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_cross_dataset_top3_v20_mech_full_report.md`

Architecture-control report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_cross_dataset_top3_v20_arch_compare_report.md`

Metrics:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_mech_full\cross_dataset_top3_summary.csv`

| direction | candidate | bucket | acc | macro-F1 | good | medium | bad |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PTB-v20 -> BUT | E4 | source PTB test | 0.9822 | 0.9823 | 1.0000 | 0.9980 | 0.9467 |
| PTB-v20 -> BUT | E4 | BUT test | 0.4351 | 0.4906 | 0.9980 | 0.1310 | 1.0000 |
| PTB-v20 -> BUT | E4 | BUT all | 0.7838 | 0.7453 | 0.9818 | 0.2881 | 0.9998 |
| PTB-v20 -> BUT | P2 | source PTB test | 0.9837 | 0.9837 | 1.0000 | 1.0000 | 0.9489 |
| PTB-v20 -> BUT | P2 | BUT test | 0.5324 | 0.5202 | 0.9940 | 0.2826 | 1.0000 |
| PTB-v20 -> BUT | P2 | BUT all | 0.7849 | 0.7702 | 0.8971 | 0.4439 | 0.9998 |
| BUT -> PTB-v20 | E4 | source BUT test | 0.9797 | 0.9842 | 0.9990 | 0.9692 | 1.0000 |
| BUT -> PTB-v20 | E4 | PTB-v20 test | 0.8516 | 0.8555 | 0.7500 | 0.9980 | 0.7889 |
| BUT -> PTB-v20 | E4 | PTB-v20 all | 0.8452 | 0.8492 | 0.7297 | 0.9977 | 0.8083 |
| BUT -> PTB-v20 | P2 | source BUT test | 0.9662 | 0.9741 | 0.9990 | 0.9485 | 1.0000 |
| BUT -> PTB-v20 | P2 | PTB-v20 test | 0.8196 | 0.8192 | 0.8429 | 1.0000 | 0.5933 |
| BUT -> PTB-v20 | P2 | PTB-v20 all | 0.8238 | 0.8259 | 0.8357 | 0.9990 | 0.6367 |

Architecture controls:

| direction | candidate | bucket | acc | macro-F1 | good | medium | bad |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PTB-v20 -> BUT | E1 | BUT test | 0.6908 | 0.6980 | 0.9871 | 0.5301 | 1.0000 |
| PTB-v20 -> BUT | E1 | BUT all | 0.8184 | 0.8292 | 0.8351 | 0.6702 | 0.9998 |
| PTB-v20 -> BUT | E2 | BUT test | 0.5777 | 0.5736 | 0.9960 | 0.3515 | 1.0000 |
| PTB-v20 -> BUT | E2 | BUT all | 0.7997 | 0.8006 | 0.8570 | 0.5666 | 0.9998 |
| BUT -> PTB-v20 | E1 | PTB-v20 test | 0.8736 | 0.8771 | 0.8186 | 1.0000 | 0.7867 |
| BUT -> PTB-v20 | E1 | PTB-v20 all | 0.8772 | 0.8805 | 0.8273 | 0.9993 | 0.8050 |
| BUT -> PTB-v20 | E2 | PTB-v20 test | 0.8849 | 0.8880 | 0.8540 | 0.9941 | 0.7933 |
| BUT -> PTB-v20 | E2 | PTB-v20 all | 0.8866 | 0.8893 | 0.8547 | 0.9937 | 0.8113 |

Old legacy top3 comparison:

- Legacy `BUT -> PTB` bad recall was almost 0 for E0/E1/E2.
- V20 `BUT -> PTB-v20` E4 bad recall is 0.789 on PTB-v20 test and 0.808 on PTB-v20 all.
- V20 `BUT -> PTB-v20` E2 improves further to 0.885 acc, 0.888 macro-F1, and 0.793 bad recall on PTB-v20 test.
- Therefore V20 specifically improves bad/poor morphology transfer.

Remaining failure:

- `PTB-v20 -> BUT` now preserves good and bad but under-recognizes BUT medium.
- The best V20 forward transfer is E1 on BUT test: 0.691 acc, 0.987 good, 0.530 medium, 1.000 bad. This is better than legacy E2 test acc 0.656 and medium 0.495, but still far below the target.
- This means bad distribution alignment is no longer the only bottleneck.
- The next synthetic loop must align natural BUT good/medium boundary distribution under the same waveform-computable feature extractor, not only bad.

## Feature Recovery Diagnosis

Important cross-domain recovery signals:

- On PTB-v20 source test, E4 recovers `baseline_step` (corr 0.982), `sqi_basSQI` (0.980), `non_qrs_diff_p95` (0.918), `flatline_ratio` (0.796), and `qrs_visibility` (0.500).
- On BUT cross test after PTB-v20 training, E4 still has weak or wrong recovery for `baseline_step` (-0.352), `sqi_basSQI` (-0.089), and `detector_agreement` (-0.319), while `flatline_ratio` (0.710) and `non_qrs_diff_p95` (0.634) transfer better.
- P2 improves BUT cross-test `qrs_visibility` (0.536) and keeps `flatline_ratio` (0.781), but `sqi_basSQI` remains negative (-0.262) and medium recall remains low.

Interpretation:

The model can learn the V20 synthetic bad mechanism. The transfer failure is now mostly a domain/feature mismatch in BUT medium/good boundary and baseline/SQI geometry, not pure model capacity. The architecture comparison also says local/artifact aux should not be forced into the main path yet: E1/E2 transfer better than E4/P2, while E4/P2 remain useful diagnostics for feature recovery.

## Next Experimental Move

V20 should be treated as the current bad/poor baseline. The next data generator should be V21:

1. Keep V20 bad subtype/peak mechanism.
2. Recompute BUT and PTB good/medium features with the same waveform extractor.
3. Add good/medium matching targets for:
   - `sqi_basSQI`
   - `baseline_step`
   - `qrs_band_ratio`
   - `qrs_visibility`
   - `non_qrs_diff_p95`
   - `amplitude_entropy`
   - `band_15_30` / `band_30_45`
4. Avoid forcing all medium to bad-like morphology; current PTB-v20 -> BUT low medium recall is the warning sign.
5. Re-run the same E4/P2 from scratch.

This keeps the method explainable: distribution fitting is driven by waveform-computable SQI/physiology targets; the final model remains waveform-only.
