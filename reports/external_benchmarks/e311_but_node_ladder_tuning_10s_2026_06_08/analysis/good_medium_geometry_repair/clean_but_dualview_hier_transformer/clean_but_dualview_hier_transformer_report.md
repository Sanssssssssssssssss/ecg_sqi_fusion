# Clean BUT Dual-View Hierarchical Transformer

No variable-length modeling. Input is fixed 10s waveform-derived channels only. Interpretable SQI/QRS/baseline/detail features supervise auxiliary heads but are not inference inputs.

Formal auxiliary targets exclude PCA/atlas/KNN geometry proxies (`pc*`, `pca_margin`, `boundary_confidence`, `region_confidence`, `knn_label_purity`).

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Aux MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | clean_val | 0.603774 | 0.407246 | 0.722222 | 0.531915 | 0.000000 | 15 | 22 | 5 | 0.7455 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | clean_test | 0.571429 | 0.388277 | 0.604167 | 0.589744 | 0.000000 | 19 | 16 | 4 | 0.7195 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | clean_val | 0.731183 | 0.770120 | 0.854167 | 0.567901 | 0.888889 | 14 | 34 | 1 | 0.4758 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | clean_test | 0.727273 | 0.701301 | 0.848837 | 0.605634 | 0.500000 | 13 | 28 | 4 | 0.4699 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | clean_val | 0.875661 | 0.862231 | 0.956522 | 0.777070 | 0.785714 | 9 | 34 | 3 | 0.4462 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | clean_test | 0.850829 | 0.805387 | 0.960000 | 0.724832 | 0.615385 | 8 | 40 | 5 | 0.4445 |

## Key Feature Recovery

| Candidate | Feature | Corr | MAE |
| --- | --- | ---: | ---: |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | amplitude_entropy | 0.7383 | 0.8082 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | baseline_step | 0.6998 | 0.7331 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | contact_loss_win_ratio | -0.1760 | 0.3439 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | detector_agreement | -0.0148 | 0.5584 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | flatline_ratio | 0.4510 | 0.7592 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | non_qrs_diff_p95 | 0.3964 | 0.6973 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | qrs_band_ratio | 0.7418 | 0.7651 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | qrs_visibility | 0.2216 | 0.7960 |
| segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | sqi_basSQI | 0.6208 | 0.7528 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | amplitude_entropy | 0.8178 | 0.4519 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | baseline_step | 0.7296 | 0.4816 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | contact_loss_win_ratio | 0.1915 | 0.1541 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | detector_agreement | 0.4108 | 0.4750 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | flatline_ratio | 0.5763 | 0.5487 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | non_qrs_diff_p95 | 0.5087 | 0.5780 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | qrs_band_ratio | 0.8883 | 0.3575 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | qrs_visibility | 0.7388 | 0.5329 |
| segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | sqi_basSQI | 0.8919 | 0.3006 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | amplitude_entropy | 0.8082 | 0.4955 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | baseline_step | 0.8094 | 0.4211 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | contact_loss_win_ratio | 0.4895 | 0.2154 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | detector_agreement | 0.4124 | 0.5648 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | flatline_ratio | 0.7915 | 0.4337 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | non_qrs_diff_p95 | 0.8460 | 0.3930 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | qrs_band_ratio | 0.9495 | 0.3356 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | qrs_visibility | 0.7718 | 0.5034 |
| segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep | sqi_basSQI | 0.9408 | 0.2934 |

## Checkpoints

- `segment_10_20s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep`: best_epoch=2, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\segment_10_20s_keep_outlier_source80_seed20260619\dualview_convtx_hier__duration_only_12ep\ckpt_best.pt`
- `segment_10_30s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep`: best_epoch=10, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\segment_10_30s_keep_outlier_source80_seed20260619\dualview_convtx_hier__duration_only_12ep\ckpt_best.pt`
- `segment_10_60s_keep_outlier_source80_seed20260619_dualview_convtx_hier__duration_only_12ep`: best_epoch=12, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\segment_10_60s_keep_outlier_source80_seed20260619\dualview_convtx_hier__duration_only_12ep\ckpt_best.pt`

## Interpretation

- `clean_test` answers whether fixed-10s cleaned labels are learnable.
- Legacy full/original buckets are emitted only with `--include-full-diagnostic` and are not selection targets.
- CV policies created from `margin_ge_5s_drop_outlier` are clean fixed-10s learnability checks, not record-heldout external tests.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_dualview_hier_transformer\clean_but_dualview_hier_transformer_metrics.csv`
Feature recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_dualview_hier_transformer\clean_but_dualview_hier_transformer_feature_recovery.csv`