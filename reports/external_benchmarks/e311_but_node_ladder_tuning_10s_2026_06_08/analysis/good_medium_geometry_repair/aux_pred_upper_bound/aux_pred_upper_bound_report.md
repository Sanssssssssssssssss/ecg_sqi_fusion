# Aux-Prediction Upper-Bound Audit

This is an external-only diagnostic.  Classifiers are fit on synthetic-train aux values only; BUT rows are report-only.

## Best Aux-Pred Classifiers On Original Test

| candidate | feature set | classifier | acc | macro-F1 | good R | medium R | bad R |
|---|---|---|---:|---:|---:|---:|---:|
| featurefirst_top20_hardrec_a050 | top20_interpretable | hgb | 0.846172 | 0.723253 | 0.860165 | 0.884772 | 0.306569 |
| featurefirst_top20_hardrec_a050 | p20_failure_axis | hgb | 0.846054 | 0.722857 | 0.872802 | 0.874831 | 0.299270 |
| featurefirst_top20_hardrec_a050 | top20_interpretable | logreg | 0.845464 | 0.718454 | 0.870604 | 0.875508 | 0.299270 |
| featurefirst_top20_hardrec_a050 | top20_interpretable | extratrees | 0.845228 | 0.715457 | 0.874725 | 0.873249 | 0.282238 |
| featurefirst_top20_hardrec_a050 | top23_hard | extratrees | 0.844992 | 0.714826 | 0.874451 | 0.873249 | 0.279805 |
| featurefirst_top20_hardrec_wideaux_a050 | full_aux | logreg | 0.820219 | 0.698257 | 0.747253 | 0.929056 | 0.294404 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | full_aux | extratrees | 0.818214 | 0.696377 | 0.887088 | 0.812020 | 0.274939 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | top20_interpretable | extratrees | 0.818214 | 0.693997 | 0.886264 | 0.813150 | 0.270073 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | top23_hard | extratrees | 0.817742 | 0.694150 | 0.886538 | 0.812020 | 0.270073 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | p20_failure_axis | hgb | 0.816327 | 0.698877 | 0.875824 | 0.816313 | 0.289538 |
| featurefirst_top20_hardrec_wideaux_a050 | full_aux | extratrees | 0.816327 | 0.697831 | 0.733516 | 0.933122 | 0.291971 |
| featurefirst_top20_hardrec_wideaux_a050 | top23_hard | extratrees | 0.815265 | 0.697217 | 0.735714 | 0.929282 | 0.291971 |
| featurefirst_top20_hardrec_wideaux_a050 | top23_hard | hgb | 0.815147 | 0.696939 | 0.735714 | 0.928152 | 0.301703 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | p20_failure_axis | extratrees | 0.815147 | 0.692921 | 0.890385 | 0.803886 | 0.270073 |
| featurefirst_top20_hardrec_wideaux_a050 | top20_interpretable | extratrees | 0.814911 | 0.696914 | 0.733791 | 0.930185 | 0.291971 |
| featurefirst_top20_hardrec_longaux_a050 | full_aux | logreg | 0.797098 | 0.688651 | 0.913462 | 0.748531 | 0.289538 |
| featurefirst_top20_hardrec_longaux_a050 | top20_interpretable | logreg | 0.791790 | 0.675056 | 0.908242 | 0.739268 | 0.326034 |
| featurefirst_top20_hardrec_longaux_a050 | p20_failure_axis | logreg | 0.790964 | 0.681387 | 0.911264 | 0.735879 | 0.318735 |
| featurefirst_top20_hardrec_longaux_a050 | top23_hard | hgb | 0.789666 | 0.676515 | 0.901648 | 0.740624 | 0.326034 |
| featurefirst_top20_hardrec_longaux_a050 | hard_missing_axes | logreg | 0.788722 | 0.679619 | 0.912637 | 0.730230 | 0.321168 |

## Feature-Space Reference On Original Test

| candidate | feature set | classifier | acc | macro-F1 | good R | medium R | bad R |
|---|---|---|---:|---:|---:|---:|---:|
| featurefirst_top20_hardrec_a050 | p20_failure_axis | extratrees | 0.935708 | 0.640715 | 0.964286 | 0.999096 | 0.000000 |
| featurefirst_top20_hardrec_longaux_a050 | p20_failure_axis | extratrees | 0.935708 | 0.640715 | 0.964286 | 0.999096 | 0.000000 |
| featurefirst_top20_hardrec_wideaux_a050 | p20_failure_axis | extratrees | 0.935708 | 0.640715 | 0.964286 | 0.999096 | 0.000000 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | p20_failure_axis | extratrees | 0.935708 | 0.640715 | 0.964286 | 0.999096 | 0.000000 |
| featurefirst_top20_hardrec_a050 | full_aux | extratrees | 0.932405 | 0.638554 | 0.955769 | 0.999774 | 0.000000 |
| featurefirst_top20_hardrec_longaux_a050 | full_aux | extratrees | 0.932405 | 0.638554 | 0.955769 | 0.999774 | 0.000000 |
| featurefirst_top20_hardrec_wideaux_a050 | full_aux | extratrees | 0.932405 | 0.638554 | 0.955769 | 0.999774 | 0.000000 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | full_aux | extratrees | 0.932405 | 0.638554 | 0.955769 | 0.999774 | 0.000000 |
| featurefirst_top20_hardrec_a050 | hard_missing_axes | extratrees | 0.930164 | 0.637170 | 0.950549 | 0.999774 | 0.000000 |
| featurefirst_top20_hardrec_longaux_a050 | hard_missing_axes | extratrees | 0.930164 | 0.637170 | 0.950549 | 0.999774 | 0.000000 |
| featurefirst_top20_hardrec_wideaux_a050 | hard_missing_axes | extratrees | 0.930164 | 0.637170 | 0.950549 | 0.999774 | 0.000000 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | hard_missing_axes | extratrees | 0.930164 | 0.637170 | 0.950549 | 0.999774 | 0.000000 |

## Hard Feature Recovery

| candidate | split | feature | corr | mae_z |
|---|---|---|---:|---:|
| featurefirst_top20_hardrec_a050 | original_all | qrs_visibility | 0.487195 | 0.854189 |
| featurefirst_top20_hardrec_a050 | original_all | detector_agreement | 0.304026 | 0.700587 |
| featurefirst_top20_hardrec_a050 | original_all | baseline_step | 0.492856 | 0.580477 |
| featurefirst_top20_hardrec_a050 | original_all | sqi_basSQI | 0.307018 | 0.551658 |
| featurefirst_top20_hardrec_a050 | original_all | flatline_ratio | 0.802130 | 0.545322 |
| featurefirst_top20_hardrec_a050 | original_all | pc2 | 0.114956 | 0.685941 |
| featurefirst_top20_hardrec_a050 | original_all | pc3 | 0.471645 | 0.766379 |
| featurefirst_top20_hardrec_a050 | original_all | pca_margin | 0.887050 | 0.318268 |
| featurefirst_top20_hardrec_a050 | original_all | boundary_confidence | 0.251876 | 0.755418 |
| featurefirst_top20_hardrec_a050 | original_all | knn_label_purity | 0.401074 | 0.540952 |
| featurefirst_top20_hardrec_a050 | synthetic_test | qrs_visibility | 0.168494 | 0.883907 |
| featurefirst_top20_hardrec_a050 | synthetic_test | detector_agreement | 0.259998 | 0.675765 |
| featurefirst_top20_hardrec_a050 | synthetic_test | baseline_step | 0.324098 | 0.811107 |
| featurefirst_top20_hardrec_a050 | synthetic_test | sqi_basSQI | 0.202088 | 0.894751 |
| featurefirst_top20_hardrec_a050 | synthetic_test | flatline_ratio | 0.555303 | 0.718472 |
| featurefirst_top20_hardrec_a050 | synthetic_test | pc2 | 0.053762 | 0.998218 |
| featurefirst_top20_hardrec_a050 | synthetic_test | pc3 | 0.409088 | 0.757242 |
| featurefirst_top20_hardrec_a050 | synthetic_test | pca_margin | 0.747684 | 0.299777 |
| featurefirst_top20_hardrec_a050 | synthetic_test | boundary_confidence | 0.332299 | 0.627704 |
| featurefirst_top20_hardrec_a050 | synthetic_test | knn_label_purity | 0.434709 | 0.676243 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | qrs_visibility | 0.342080 | 0.921725 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | detector_agreement | 0.301329 | 0.698690 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | baseline_step | 0.343871 | 0.616384 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | sqi_basSQI | 0.079123 | 0.578424 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | flatline_ratio | 0.817721 | 0.502324 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | pc2 | -0.225007 | 0.712889 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | pc3 | 0.474881 | 0.760369 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | pca_margin | 0.876073 | 0.321537 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | boundary_confidence | 0.157571 | 0.783434 |
| featurefirst_top20_hardrec_longaux_a050 | original_all | knn_label_purity | 0.334383 | 0.531745 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | qrs_visibility | 0.166570 | 0.880491 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | detector_agreement | 0.264213 | 0.672906 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | baseline_step | 0.322878 | 0.822980 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | sqi_basSQI | 0.207188 | 0.894832 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | flatline_ratio | 0.559213 | 0.714060 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | pc2 | 0.048793 | 1.001456 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | pc3 | 0.407191 | 0.765127 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | pca_margin | 0.746292 | 0.300686 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | boundary_confidence | 0.334896 | 0.635444 |
| featurefirst_top20_hardrec_longaux_a050 | synthetic_test | knn_label_purity | 0.437652 | 0.675314 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | qrs_visibility | 0.411356 | 0.876754 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | detector_agreement | 0.313295 | 0.700678 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | baseline_step | 0.410187 | 0.606812 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | sqi_basSQI | 0.238040 | 0.556977 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | flatline_ratio | 0.832920 | 0.504100 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | pc2 | -0.000056 | 0.690419 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | pc3 | 0.486542 | 0.756783 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | pca_margin | 0.888544 | 0.319176 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | boundary_confidence | 0.198625 | 0.771550 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | original_all | knn_label_purity | 0.344397 | 0.549114 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | qrs_visibility | 0.171913 | 0.870275 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | detector_agreement | 0.246953 | 0.692453 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | baseline_step | 0.324869 | 0.818119 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | sqi_basSQI | 0.213117 | 0.891272 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | flatline_ratio | 0.555055 | 0.715098 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | pc2 | 0.068088 | 0.993251 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | pc3 | 0.403942 | 0.754227 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | pca_margin | 0.747497 | 0.300947 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | boundary_confidence | 0.333941 | 0.633045 |
| featurefirst_top20_hardrec_longaux_qrsbase_a050 | synthetic_test | knn_label_purity | 0.432018 | 0.681886 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | qrs_visibility | 0.480795 | 0.857233 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | detector_agreement | 0.283523 | 0.704421 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | baseline_step | 0.496073 | 0.581793 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | sqi_basSQI | 0.271287 | 0.552886 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | flatline_ratio | 0.791754 | 0.533654 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | pc2 | 0.130706 | 0.682243 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | pc3 | 0.482327 | 0.760363 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | pca_margin | 0.882689 | 0.323756 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | boundary_confidence | 0.248764 | 0.752194 |
| featurefirst_top20_hardrec_wideaux_a050 | original_all | knn_label_purity | 0.424374 | 0.519820 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | qrs_visibility | 0.169656 | 0.879867 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | detector_agreement | 0.255024 | 0.670802 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | baseline_step | 0.323379 | 0.816940 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | sqi_basSQI | 0.206217 | 0.895056 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | flatline_ratio | 0.549759 | 0.720236 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | pc2 | 0.056375 | 0.995693 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | pc3 | 0.408254 | 0.760417 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | pca_margin | 0.745436 | 0.304865 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | boundary_confidence | 0.330399 | 0.632034 |
| featurefirst_top20_hardrec_wideaux_a050 | synthetic_test | knn_label_purity | 0.433946 | 0.671818 |

## Interpretation

- If aux_pred classifiers approach the true-feature reference, the bottleneck is the final class head/fusion.
- If aux_pred classifiers stay near the waveform model's own accuracy, the bottleneck is feature recovery in the encoder/tokenizer.
- The true-feature rows are an upper reference for the 47-column geometry/SQI space, not a deployable waveform-only result.

## Files

- metrics: `E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/aux_pred_upper_bound/aux_pred_upper_bound_metrics.csv`
- recovery: `E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/aux_pred_upper_bound/aux_pred_feature_recovery.csv`