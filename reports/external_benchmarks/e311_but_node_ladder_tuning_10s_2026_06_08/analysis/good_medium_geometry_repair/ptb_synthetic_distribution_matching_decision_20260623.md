# PTB Synthetic Distribution Matching Decision - 2026-06-23

## Decision

The current optimization target is distribution and visual likeness, not cross-dataset accuracy.  The fixed downstream model remains `EventFactorizedSQIConformer` with waveform-only inference; no MLP/tree/route/rule artifact is part of the formal model path.

The best current synthetic protocol is:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78rawcarrier_ot_fast\protocol_v78rawcarrier_ot_fast_pc1500_s20260681`

Use `v78rawcarrier_ot_fast` as the next distribution baseline.  Reject `v79rawcarrier_bad_envelope_ot_fast`: the formal bad-envelope calibration made bad gaps worse.

## What Changed

- Built a raw PTB-XL carrier protocol from `records100` lead II, excluding PTB-XL rows with baseline/static/burst/electrode noise notes.
- Added `--base-protocol` to the subtype-balanced generator so synthetic rows can start from raw PTB-XL carrier instead of replay/boundary banks.
- Added a v78 OT-like subtype matcher that combines:
  - robust waveform/SQI feature z-scores;
  - BUT-fitted PCA coordinates;
  - low-dimensional waveform prototype distance;
  - nearest-neighbor shortlist plus linear assignment.
- Kept BUT reference fixed at `margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`.

## Version Comparison

| version | class | median gap | quantile loss | sliced Wasserstein | MMD | PCA gap | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v75 | good | 5.159 | 5.127 | 14.601 | 1.338 | 0.980 | 1.000 |
| v75 | medium | 1.131 | 1.103 | 1.528 | 0.990 | 0.860 | 1.000 |
| v75 | bad | 9.517 | 9.513 | 16.636 | 1.279 | 0.975 | 1.000 |
| v78 | good | 1.110 | 1.102 | 1.771 | 1.088 | 0.891 | 1.000 |
| v78 | medium | 0.725 | 0.739 | 1.389 | 0.934 | 0.777 | 1.000 |
| v78 | bad | 8.325 | 8.251 | 14.269 | 1.287 | 0.975 | 1.000 |
| v79 | good | 1.118 | 1.104 | 1.764 | 1.088 | 0.880 | 1.000 |
| v79 | medium | 0.721 | 0.739 | 1.366 | 0.933 | 0.768 | 1.000 |
| v79 | bad | 10.914 | 10.870 | 16.840 | 1.324 | 0.970 | 1.000 |

Interpretation: v78 gives the first real good/medium distribution improvement from raw PTB-XL carrier.  AUC remains 1.0, so the domains are still statistically separable even when PCA looks closer.  This is acceptable for this stage but must be reported honestly.

## Figures

- v78 shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78rawcarrier_ot_fast\v78rawcarrier_ot_fast_shared_pca_but_vs_ptb.png`
- v78 feature CDF audit: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v78rawcarrier_ot_fast_distribution_first\v78rawcarrier_ot_fast_distribution_first_key_feature_cdf_overlap.png`
- v78 waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78rawcarrier_ot_fast`
- raw PTB carrier audit: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\raw_ptbxl_carrier_max9000`

## Remaining Gaps

- Good/medium are much more visually plausible than the replay-bank versions, but CDF gaps remain for `qrs_visibility`, `sqi_basSQI`, `non_qrs_diff_p95`, and `amplitude_entropy`.
- Bad still needs a mechanism-specific noise-envelope generator.  Simple energy calibration is insufficient and can worsen distribution metrics.
- `discriminative_auc=1.0` across versions means synthetic and BUT remain separable; future work should reduce AUC with distribution transforms before training.

## Next Data Step

Do not train on v79.  Next generator iteration should keep v78 good/medium, then replace only bad generation with BUT-style noise-envelope matching:

- match intermittent amplitude envelope, not just RMS;
- match band ratios per subtype, especially 15-30 Hz and 30-45 Hz;
- match detector failure mechanism without making all bad rows homogeneous dense noise;
- keep contact/reset/flatline subtype, which is already visually plausible.
