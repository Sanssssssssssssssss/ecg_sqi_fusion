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
| per_lead | MLP_predict | 100 | 0.293 | 0.277 | 0.430 | 0.063 | 0.222 | 0.510 |
| per_lead | SVM_predict | 100 | 0.813 | 0.724 | 1.164 | 0.220 | 0.551 | 1.204 |
| per_lead | eplimited_PandT | 100 | 41.083 | 40.934 | 49.105 | 4.352 | 30.008 | 52.859 |
| per_lead | fSQI | 100 | 0.033 | 0.033 | 0.054 | 0.012 | 0.018 | 0.084 |
| per_lead | kSQI | 100 | 0.749 | 0.765 | 1.105 | 0.219 | 0.400 | 1.418 |
| per_lead | sSQI | 100 | 0.563 | 0.627 | 0.782 | 0.176 | 0.305 | 0.984 |
| per_lead | shared_pSQI_basSQI_PSD | 100 | 0.428 | 0.421 | 0.634 | 0.177 | 0.213 | 1.496 |
| per_lead | wqrs | 100 | 42.790 | 41.535 | 52.402 | 6.380 | 32.518 | 76.805 |

This run, full 12-lead end-to-end:

| scope | component | n | mean_ms | median_ms | p95_ms | std_ms | min_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_12lead | feature84_ms | 100 | 18.678 | 18.045 | 26.506 | 4.437 | 10.419 | 27.910 |
| full_12lead | mlp_predict_ms | 100 | 0.298 | 0.272 | 0.401 | 0.065 | 0.241 | 0.699 |
| full_12lead | qrs_ms | 100 | 565.939 | 572.149 | 631.008 | 50.334 | 405.988 | 764.384 |
| full_12lead | svm_predict_ms | 100 | 0.525 | 0.479 | 0.739 | 0.126 | 0.371 | 0.853 |
| full_12lead | total_ms | 100 | 585.443 | 591.197 | 649.627 | 52.498 | 418.275 | 786.646 |

Timing trend: wqrs is still the slowest per-lead detector component (42.8 ms mean), and the full 12-lead path is dominated by QRS (565.9 ms of 585.4 ms mean total). The qualitative ordering matches Table 8, while absolute times include Python, temporary WFDB record writing, and external executable launch overhead.

## Limitations

- MIT-BIH transfer is a weak-label rhythm/generalization proxy, not a replication of the paper's single-lead classifier with single-lead quality labels.
- Timing includes Python overhead, WFDB temp-record writing, and C executable launch overhead; it is therefore expected to be slower than paper Table 8 Matlab component timings in some places and not hardware-comparable.
- The paper's single-lead 5-10 s window experiment remains intentionally unimplemented in this repo until single-lead labels are available.
