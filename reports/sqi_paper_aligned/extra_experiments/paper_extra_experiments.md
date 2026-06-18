# Paper Extra Experiments

## Window-Length Experiment

Skipped for this rerun. Clifford et al. describe the 5-10 s window-length test as being performed on single-lead data. This repository does not have the paper's single-lead manual labels, so generating a window curve would be a weak proxy rather than a faithful replication.

## MIT-BIH Transfer Proxy

MIT-BIH has arrhythmia annotations but no signal-quality labels. We therefore treat MIT-BIH windows as acceptable-proxy examples and report acceptance / proxy false-rejection rate, not paper-exact accuracy.

- MIT-BIH records processed: 48
- MIT-BIH feature rows: 17280

Challenge weak-label proxy performance:

| model | n_train_leads | n_val_leads | n_test_leads | C | gamma | threshold | test_Ac | test_Se | test_Sp | test_AUC | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| singlelead_weak_svm | 12996 | 2784 | 2772 | 25.000 | 0.031 | 0.385 | 0.902 | 0.931 | 0.872 | 0.932 | 1203 | 177 | 96 | 1296 |
| singlelead_weak_lm_mlp_J5 | 12996 | 2784 | 2772 |  |  | 0.442 | 0.904 | 0.956 | 0.851 | 0.940 | 1174 | 206 | 61 | 1331 |

MIT-BIH transfer summary:

| model | threshold | n_windows_leads | n_records | n_leads | acceptance_rate | false_rejection_rate_proxy | mean_p_accept | median_p_accept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| singlelead_weak_svm | 0.385 | 17280 | 48 | 96 | 0.870 | 0.130 | 0.737 | 0.843 |
| singlelead_weak_lm_mlp_J5 | 0.442 | 17280 | 48 | 96 | 0.866 | 0.134 | 0.782 | 0.900 |

Directional comparison: the paper reports approximately 93% accuracy for its single-lead MIT-BIH transfer experiment. This rerun is weaker by design: it accepts 87.0% (SVM) and 86.6% (MLP) of MIT-BIH lead-windows under a Set-a-trained proxy, so it should be read as a false-rejection/generalization signal rather than paper-exact accuracy.

## Execution-Time Benchmark

Paper Table 8 reference:

| component | paper_ms |
| --- | --- |
| kSQI | 0.330 |
| sSQI | 0.290 |
| pSQI/basSQI shared PSD | 1.920 |
| fSQI | 0.070 |
| P&T/eplimited | 2.460 |
| wqrs | 33.180 |
| SVM predict | 0.100 |
| MLP predict | 0.001 |

This run, per-lead components:

| scope | component | n | mean_ms | median_ms | p95_ms | std_ms | min_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| per_lead | MLP_predict | 100 | 0.259 | 0.233 | 0.361 | 0.051 | 0.214 | 0.441 |
| per_lead | SVM_predict | 100 | 0.664 | 0.564 | 1.106 | 0.206 | 0.503 | 1.265 |
| per_lead | eplimited_PandT | 100 | 38.658 | 37.050 | 45.209 | 7.707 | 28.235 | 92.983 |
| per_lead | fSQI | 100 | 0.035 | 0.023 | 0.056 | 0.037 | 0.018 | 0.372 |
| per_lead | kSQI | 100 | 0.845 | 0.702 | 1.318 | 0.309 | 0.381 | 2.077 |
| per_lead | sSQI | 100 | 0.523 | 0.380 | 0.898 | 0.223 | 0.288 | 1.136 |
| per_lead | shared_pSQI_basSQI_PSD | 100 | 0.434 | 0.373 | 0.681 | 0.171 | 0.225 | 1.408 |
| per_lead | wqrs | 100 | 42.617 | 39.685 | 57.590 | 13.304 | 29.843 | 129.746 |

This run, full 12-lead end-to-end:

| scope | component | n | mean_ms | median_ms | p95_ms | std_ms | min_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_12lead | feature84_ms | 100 | 16.007 | 14.755 | 23.320 | 4.270 | 10.403 | 32.455 |
| full_12lead | mlp_predict_ms | 100 | 0.287 | 0.252 | 0.408 | 0.063 | 0.230 | 0.464 |
| full_12lead | qrs_ms | 100 | 570.524 | 566.782 | 674.609 | 62.547 | 402.119 | 794.101 |
| full_12lead | svm_predict_ms | 100 | 0.485 | 0.398 | 0.795 | 0.152 | 0.358 | 0.867 |
| full_12lead | total_ms | 100 | 587.306 | 585.075 | 694.277 | 64.741 | 413.461 | 813.096 |

Timing trend: wqrs is still the slowest per-lead detector component (42.6 ms mean), and the full 12-lead path is dominated by QRS (570.5 ms of 587.3 ms mean total). The qualitative ordering matches Table 8, while absolute times include Python, temporary WFDB record writing, and external executable launch overhead.

## Limitations

- MIT-BIH transfer is a weak-label rhythm/generalization proxy, not a replication of the paper's single-lead classifier with single-lead quality labels.
- Timing includes Python overhead, WFDB temp-record writing, and C executable launch overhead; it is therefore expected to be slower than paper Table 8 Matlab component timings in some places and not hardware-comparable.
- The paper's single-lead 5-10 s window experiment remains intentionally unimplemented in this repo until single-lead labels are available.
