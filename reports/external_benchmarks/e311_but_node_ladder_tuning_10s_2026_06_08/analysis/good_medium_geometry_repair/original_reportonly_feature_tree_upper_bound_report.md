# Original Report-Only Feature Tree Upper Bound

This intentionally trains on original train+val labels and is therefore report-only. It answers whether the original bad-stress bucket is separable by simple ECG geometry/SQI features. It is not used for model selection.
- orig_reportonly_dtree_depth8_leaf10: original test acc=0.9385, macro-F1=0.8290, G/M/B=0.994/0.941/0.416
- orig_reportonly_dtree_depth6_leaf25: original test acc=0.9260, macro-F1=0.8206, G/M/B=0.973/0.935/0.416
- orig_reportonly_dtree_depth4_leaf50: original test acc=0.9211, macro-F1=0.8172, G/M/B=0.963/0.934/0.416
- orig_reportonly_dtree_depth5_leaf50: original test acc=0.9184, macro-F1=0.8152, G/M/B=0.980/0.914/0.416
- orig_reportonly_dtree_depth3_leaf50: original test acc=0.9099, macro-F1=0.7655, G/M/B=0.969/0.919/0.290
- orig_reportonly_dtree_depth2_leaf50: original test acc=0.7445, macro-F1=0.6506, G/M/B=0.943/0.624/0.290

## Best Rule Text

```text
|--- pc1 <= 3.97
|   |--- pc1 <= -1.93
|   |   |--- boundary_confidence <= 0.35
|   |   |   |--- qrs_visibility <= 0.34
|   |   |   |   |--- boundary_confidence <= 0.29
|   |   |   |   |   |--- pc1 <= -3.26
|   |   |   |   |   |   |--- pca_margin <= 0.36
|   |   |   |   |   |   |   |--- amplitude_entropy <= 0.67
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- amplitude_entropy >  0.67
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- pca_margin >  0.36
|   |   |   |   |   |   |   |--- pc1 <= -4.30
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |   |--- pc1 >  -4.30
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- pc1 >  -3.26
|   |   |   |   |   |   |--- pc2 <= 5.12
|   |   |   |   |   |   |   |--- diff_abs_p95 <= 0.17
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- diff_abs_p95 >  0.17
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- pc2 >  5.12
|   |   |   |   |   |   |   |--- diff_abs_p95 <= 0.08
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |   |--- diff_abs_p95 >  0.08
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |--- boundary_confidence >  0.29
|   |   |   |   |   |--- fatal_or_score <= 0.92
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- fatal_or_score >  0.92
|   |   |   |   |   |   |--- pca_margin <= 0.28
|   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |--- pca_margin >  0.28
|   |   |   |   |   |   |   |--- class: 1
|   |   |   |--- qrs_visibility >  0.34
|   |   |   |   |--- boundary_confidence <= 0.34
|   |   |   |   |   |--- qrs_visibility <= 0.43
|   |   |   |   |   |   |--- boundary_confidence <= 0.26
|   |   |   |   |   |   |   |--- pc2 <= 3.06
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- pc2 >  3.06
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- boundary_confidence >  0.26
|   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- qrs_visibility >  0.43
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- boundary_confidence >  0.34
|   |   |   |   |   |--- class: 1
|   |   |--- boundary_confidence >  0.35
|   |   |   |--- sqi_sSQI <= -0.44
|   |   |   |   |--- flatline_ratio <= 0.39
|   |   |   |   |   |--- boundary_confidence <= 0.43
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- boundary_confidence >  0.43
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- flatline_ratio >  0.39
|   |   |   |   |   |--- template_corr <= 0.49
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- template_corr >  0.49
|   |   |   |   |   |   |--- class: 0
|   |   |   |--- sqi_sSQI >  -0.44
|   |   |   |   |--- knn_label_purity <= 0.48
|   |   |   |   |   |--- ptp_p99_p01 <= 1.96
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- ptp_p99_p01 >  1.96
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- knn_label_purity >  0.48
|   |   |   |   |   |--- pc4 <= -2.34
|   |   |   |   |   |   |--- pc3 <= -0.79
|   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |--- pc3 >  -0.79
|   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- pc4 >  -2.34
|   |   |   |   |   |   |--- template_corr <= 0.48
|   |   |   |   |   |   |   |--- pc4 <= -0.36
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- pc4 >  -0.36
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |--- template_corr >  0.48
|   |   |   |   |   |   |   |--- boundary_confidence <= 0.42
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |   |--- boundary_confidence >  0.42
|   |   |   |   |   |   |   |   |--- class: 0
|   |--- pc1 >  -1.93
|   |   |--- pca_margin <= 0.00
|   |   |   |--- pca_margin <= -4.13
|   |   |   |   |--- class: 2
|   |   |   |--- pca_margin >  -4.13
|   |   |   |   |--- pca_margin <= -0.26
|   |   |   |   |   |--- pc2 <= -3.10
|   |   |   |   |   |   |--- pc1 <= -1.44
|   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- pc1 >  -1.44
|   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- pc2 >  -3.10
|   |   |   |   |   |   |--- pc1 <= -1.90
|   |   |   |   |   |   |   |--- pc2 <= -1.35
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- pc2 >  -1.35
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |--- pc1 >  -1.90
|   |   |   
```
