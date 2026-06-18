# Bad Outlier Internal Gate Gap

Internal-gate checkpoint stress bucket: 29/292 bad outlier stress hit.

## Prediction split

pred_class
good      202
medium     61
bad        29

## Feature summary (selected medians)

                  group   n   pc1_med  baseline_step_med  qrs_visibility_med  non_qrs_diff_p95_med  band_30_45_med  flatline_ratio_med  pca_margin_med
      train_bad_outlier 827 10.413526           0.027372            0.244737              0.374253        0.103975            0.007206       10.344219
        test_stress_hit  29 -1.507149           1.121648            0.051036              0.086822        0.018545            0.218575       -5.944014
  test_stress_miss_good 202 -3.941972           1.382584            0.040383              0.042081        0.007433            0.392714       -6.580175
test_stress_miss_medium  61 -6.512429           1.483594            0.019741              0.019856        0.004832            0.632506       -5.114377

Files: `bad_outlier_stress_internal_gate_rows.csv`, `bad_outlier_stress_hit_miss_feature_summary.csv`, `bad_outlier_train_test_comparison.csv`, `bad_outlier_simple_rule_grid.csv`.
