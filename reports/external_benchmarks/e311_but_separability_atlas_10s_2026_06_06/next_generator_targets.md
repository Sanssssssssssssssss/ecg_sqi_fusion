# Next Generator Targets

- First match BUT feature distributions on the top separating dimensions, class-wise.
- Then run small validation grids only for candidate distributions whose medium and bad distances both move in the correct direction.
- Treat feature target score as a gate, not as a final metric; BUT original test remains final.

## Target table

| feature | family | separability_score | max_ks | max_abs_cliffs |
| --- | --- | --- | --- | --- |
| sample_entropy_proxy | Detail reliability | 0.9616 | 0.9432 | 0.9645 |
| higuchi_fd_proxy | Detail reliability | 0.8406 | 0.9364 | 0.9241 |
| sqi_bSQI | SQI | 0.8028 | 0.9568 | 0.9937 |
| non_qrs_diff_p95 | Detail reliability | 0.778 | 0.9337 | 0.9414 |
| template_corr | QRS detectability | 0.7685 | 0.967 | 0.989 |
| diff_abs_median | Detail reliability | 0.7625 | 0.934 | 0.9072 |
| zero_crossing_rate | Motion/frequency | 0.7444 | 0.9337 | 0.912 |
| band_30_45 | Motion/frequency | 0.7255 | 0.9336 | 0.9103 |
| hjorth_mobility | Motion/frequency | 0.7143 | 0.9336 | 0.8924 |
| sqi_pSQI | SQI | 0.7115 | 0.9325 | 0.8891 |
| wavelet_e4 | Motion/frequency | 0.7047 | 0.9336 | 0.8986 |
| diff_zero_crossing_rate | Motion/frequency | 0.687 | 0.9316 | 0.9028 |
| flatline_ratio | Contact/flat/fatal | 0.6835 | 0.9376 | 0.9189 |
| band_5_15 | Motion/frequency | 0.6828 | 0.9127 | 0.9804 |
| non_qrs_rms_ratio | Detail reliability | 0.6823 | 0.9231 | 0.9711 |
| spectral_entropy | Motion/frequency | 0.6823 | 0.8918 | 0.9214 |
| band_15_30 | Motion/frequency | 0.681 | 0.9086 | 0.8773 |
| sqi_sSQI | SQI | 0.6709 | 0.9467 | 0.9583 |
| rr_count_detector_b | QRS detectability | 0.6644 | 0.9172 | 0.9359 |
| rms | Amplitude/global | 0.6638 | 0.9153 | 0.977 |