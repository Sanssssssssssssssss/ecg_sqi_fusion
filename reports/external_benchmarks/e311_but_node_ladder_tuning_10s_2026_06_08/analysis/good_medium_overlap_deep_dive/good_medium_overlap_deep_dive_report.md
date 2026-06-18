# Good/Medium Overlap Deep Dive

Report-only analysis. Original and filtered-original metrics are not used for node selection or promotion.

## Counts

- Original all: `32956` windows = good `17043`, medium `10628`, bad `5285`.
- Original test: `8477` windows = good `3640`, medium `4426`, bad `411`.
- Original test after dropping only bad/outlier_low_confidence: `8185` windows = good `3640`, medium `4426`, bad core `119`.

## Overlap-Only Read

- N7200 current best overlap-only: acc `0.9267`, good->medium `494`, medium->good `561`.
- Original filtered best macro overlap-only: acc `0.7277`, good->medium `909`, medium->good `1286`.
- Interpretation: the clean/node frontier is already mostly a good/medium boundary problem; original filtered is much worse because many original good/medium rows live in outlier_low_confidence and high-PC2/outer-shell overlap.

## Good Eaten As Medium

- `amplitude_entropy` KS=0.727, median gap=+1.36 ref-IQR
- `low_amp_ratio` KS=0.709, median gap=-1.21 ref-IQR
- `sqi_sSQI` KS=0.605, median gap=-1.17 ref-IQR
- `sqi_kSQI` KS=0.568, median gap=-0.71 ref-IQR
- `baseline_step` KS=0.525, median gap=+1.21 ref-IQR
- `non_qrs_rms_ratio` KS=0.440, median gap=+0.47 ref-IQR
- `qrs_prom_p90` KS=0.439, median gap=-0.46 ref-IQR
- `mean_abs` KS=0.435, median gap=+0.73 ref-IQR

Practical generator implication: create `lightly_contaminated_good` rows with modest baseline/non-QRS contamination and entropy lift, while keeping enough QRS visibility/SQI to remain good.

## Medium Eaten As Good

- `sqi_kSQI` KS=0.485, median gap=+0.70 ref-IQR
- `sqi_sSQI` KS=0.481, median gap=+0.57 ref-IQR
- `amplitude_entropy` KS=0.475, median gap=-0.70 ref-IQR
- `qrs_prom_p90` KS=0.453, median gap=+0.43 ref-IQR
- `low_amp_ratio` KS=0.452, median gap=+1.08 ref-IQR
- `rms` KS=0.434, median gap=+0.76 ref-IQR
- `ptp_p99_p01` KS=0.424, median gap=+0.67 ref-IQR
- `std` KS=0.413, median gap=+0.69 ref-IQR

Practical generator implication: create `visible_qrs_medium` rows with stronger QRS/SQI/high-amplitude structure while preserving enough instability/detail to remain medium.

## Artifacts

- `overlap_counts.csv`
- `overlap_metrics_summary.json`
- `feature_gap_good_to_medium.csv`
- `feature_gap_medium_to_good.csv`
- `overlap_feature_rank_table.csv`
- `threshold_candidates.csv`
- `n7200_overlap_pca_confusion.png`
- `n7200_overlap_umap_confusion.png`
- `n7200_overlap_density_heatmap.png`
- `original_filtered_overlap_pca_confusion.png`
- `original_filtered_overlap_umap_confusion.png`
- `original_filtered_overlap_error_waveforms.png`
- `original_filtered_overlap_correct_waveforms.png`

## Next Modeling Step

Stop one-sided N7200 weight sweeps. Run a paired overlap generator on N6800/N6900/N7000 first, then return to N7200 only after clean/node acc gets back near 0.95.
