# N17043 Feature Tree Original Bucketed Report

Transparent feature-tree diagnostic. Trained only on N17043 train+val node rows; original BUT remains report-only.

## dtree_depth3_leaf50_trainval_node
- node_test: n=8185, acc=0.9428, macro-F1=0.9611, recalls G/M/B=0.969/0.920/1.000
- node_all: n=31755, acc=0.9713, macro-F1=0.9767, recalls G/M/B=0.978/0.949/1.000
- original_test_all_10s+: n=8477, acc=0.9103, macro-F1=0.7668, recalls G/M/B=0.969/0.920/0.290
- original_all_10s+: n=32956, acc=0.9613, macro-F1=0.9600, recalls G/M/B=0.978/0.949/0.931
- original_bad_outlier_stress: n=1201, acc=0.6978, macro-F1=0.2740, recalls G/M/B=0.000/0.000/0.698
- original_bad_core_nearboundary: n=4084, acc=1.0000, macro-F1=0.3333, recalls G/M/B=0.000/0.000/1.000

```text
|--- pc1 <= 7.82
|   |--- pc1 <= -1.93
|   |   |--- boundary_confidence <= 0.35
|   |   |   |--- class: 1
|   |   |--- boundary_confidence >  0.35
|   |   |   |--- class: 0
|   |--- pc1 >  -1.93
|   |   |--- pca_margin <= 0.00
|   |   |   |--- class: 0
|   |   |--- pca_margin >  0.00
|   |   |   |--- class: 1
|--- pc1 >  7.82
|   |--- class: 2

```

## dtree_depth4_leaf50_trainval_node
- node_test: n=8185, acc=0.9481, macro-F1=0.9647, recalls G/M/B=0.963/0.935/1.000
- node_all: n=31755, acc=0.9730, macro-F1=0.9781, recalls G/M/B=0.976/0.959/1.000
- original_test_all_10s+: n=8477, acc=0.9154, macro-F1=0.7703, recalls G/M/B=0.963/0.935/0.290
- original_all_10s+: n=32956, acc=0.9630, macro-F1=0.9615, recalls G/M/B=0.976/0.959/0.931
- original_bad_outlier_stress: n=1201, acc=0.6978, macro-F1=0.2740, recalls G/M/B=0.000/0.000/0.698
- original_bad_core_nearboundary: n=4084, acc=1.0000, macro-F1=0.3333, recalls G/M/B=0.000/0.000/1.000

```text
|--- pc1 <= 7.82
|   |--- pc1 <= -1.93
|   |   |--- boundary_confidence <= 0.35
|   |   |   |--- pc2 <= 0.61
|   |   |   |   |--- class: 1
|   |   |   |--- pc2 >  0.61
|   |   |   |   |--- class: 1
|   |   |--- boundary_confidence >  0.35
|   |   |   |--- sqi_sSQI <= -0.44
|   |   |   |   |--- class: 1
|   |   |   |--- sqi_sSQI >  -0.44
|   |   |   |   |--- class: 0
|   |--- pc1 >  -1.93
|   |   |--- pca_margin <= 0.00
|   |   |   |--- pca_margin <= -0.26
|   |   |   |   |--- class: 0
|   |   |   |--- pca_margin >  -0.26
|   |   |   |   |--- class: 0
|   |   |--- pca_margin >  0.00
|   |   |   |--- band_15_30 <= 0.37
|   |   |   |   |--- class: 1
|   |   |   |--- band_15_30 >  0.37
|   |   |   |   |--- class: 0
|--- pc1 >  7.82
|   |--- class: 2

```

## dtree_depth5_leaf50_trainval_node
- node_test: n=8185, acc=0.9451, macro-F1=0.9628, recalls G/M/B=0.980/0.915/1.000
- node_all: n=31755, acc=0.9767, macro-F1=0.9810, recalls G/M/B=0.987/0.951/1.000
- original_test_all_10s+: n=8477, acc=0.9126, macro-F1=0.7683, recalls G/M/B=0.980/0.915/0.290
- original_all_10s+: n=32956, acc=0.9665, macro-F1=0.9643, recalls G/M/B=0.987/0.951/0.931
- original_bad_outlier_stress: n=1201, acc=0.6978, macro-F1=0.2740, recalls G/M/B=0.000/0.000/0.698
- original_bad_core_nearboundary: n=4084, acc=1.0000, macro-F1=0.3333, recalls G/M/B=0.000/0.000/1.000

```text
|--- pc1 <= 7.82
|   |--- pc1 <= -1.93
|   |   |--- boundary_confidence <= 0.35
|   |   |   |--- pc2 <= 0.61
|   |   |   |   |--- boundary_confidence <= 0.26
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- boundary_confidence >  0.26
|   |   |   |   |   |--- class: 1
|   |   |   |--- pc2 >  0.61
|   |   |   |   |--- boundary_confidence <= 0.29
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- boundary_confidence >  0.29
|   |   |   |   |   |--- class: 0
|   |   |--- boundary_confidence >  0.35
|   |   |   |--- sqi_sSQI <= -0.44
|   |   |   |   |--- flatline_ratio <= 0.39
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- flatline_ratio >  0.39
|   |   |   |   |   |--- class: 0
|   |   |   |--- sqi_sSQI >  -0.44
|   |   |   |   |--- knn_label_purity <= 0.58
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- knn_label_purity >  0.58
|   |   |   |   |   |--- class: 0
|   |--- pc1 >  -1.93
|   |   |--- pca_margin <= 0.00
|   |   |   |--- pca_margin <= -0.26
|   |   |   |   |--- pc2 <= -3.07
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- pc2 >  -3.07
|   |   |   |   |   |--- class: 0
|   |   |   |--- pca_margin >  -0.26
|   |   |   |   |--- boundary_confidence <= 0.39
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- boundary_confidence >  0.39
|   |   |   |   |   |--- class: 0
|   |   |--- pca_margin >  0.00
|   |   |   |--- band_15_30 <= 0.37
|   |   |   |   |--- qrs_visibility <= 0.58
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- qrs_visibility >  0.58
|   |   |   |   |   |--- class: 0
|   |   |   |--- band_15_30 >  0.37
|   |   |   |   |--- class: 0
|--- pc1 >  7.82
|   |--- class: 2

```

## dtree_depth8_leaf10_trainval_node
- node_test: n=8185, acc=0.9765, macro-F1=0.9840, recalls G/M/B=0.991/0.964/1.000
- node_all: n=31755, acc=0.9897, macro-F1=0.9917, recalls G/M/B=0.993/0.980/1.000
- original_test_all_10s+: n=8477, acc=0.9429, macro-F1=0.7891, recalls G/M/B=0.991/0.964/0.290
- original_all_10s+: n=32956, acc=0.9791, macro-F1=0.9751, recalls G/M/B=0.993/0.980/0.931
- original_bad_outlier_stress: n=1201, acc=0.6978, macro-F1=0.2740, recalls G/M/B=0.000/0.000/0.698
- original_bad_core_nearboundary: n=4084, acc=1.0000, macro-F1=0.3333, recalls G/M/B=0.000/0.000/1.000

```text
|--- pc1 <= 7.82
|   |--- pc1 <= -1.93
|   |   |--- boundary_confidence <= 0.35
|   |   |   |--- pc2 <= 0.61
|   |   |   |   |--- boundary_confidence <= 0.34
|   |   |   |   |   |--- ptp_p99_p01 <= 2.16
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- ptp_p99_p01 >  2.16
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- boundary_confidence >  0.34
|   |   |   |   |   |--- class: 1
|   |   |   |--- pc2 >  0.61
|   |   |   |   |--- boundary_confidence <= 0.29
|   |   |   |   |   |--- pc3 <= -0.37
|   |   |   |   |   |   |--- pca_margin <= 0.36
|   |   |   |   |   |   |   |--- pc1 <= -3.51
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- pc1 >  -3.51
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- pca_margin >  0.36
|   |   |   |   |   |   |   |--- pc1 <= -4.17
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |   |--- pc1 >  -4.17
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- pc3 >  -0.37
|   |   |   |   |   |   |--- pca_margin <= -0.65
|   |   |   |   |   |   |   |--- amplitude_entropy <= 0.77
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |   |--- amplitude_entropy >  0.77
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |--- pca_margin >  -0.65
|   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- boundary_confidence >  0.29
|   |   |   |   |   |--- fatal_or_score <= 0.92
|   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |--- fatal_or_score >  0.92
|   |   |   |   |   |   |--- pca_margin <= -0.26
|   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |--- pca_margin >  -0.26
|   |   |   |   |   |   |   |--- pc1 <= -4.21
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |   |--- pc1 >  -4.21
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |--- boundary_confidence >  0.35
|   |   |   |--- sqi_sSQI <= -0.44
|   |   |   |   |--- flatline_ratio <= 0.39
|   |   |   |   |   |--- flatline_ratio <= 0.28
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- flatline_ratio >  0.28
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- flatline_ratio >  0.39
|   |   |   |   |   |--- class: 0
|   |   |   |--- sqi_sSQI >  -0.44
|   |   |   |   |--- knn_label_purity <= 0.48
|   |   |   |   |   |--- amplitude_entropy <= 0.64
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- amplitude_entropy >  0.64
|   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- knn_label_purity >  0.48
|   |   |   |   |   |--- pc4 <= -2.34
|   |   |   |   |   |   |--- pc3 <= -0.79
|   |   |   |   |   |   |   |--- ptp_p99_p01 <= 1.02
|   |   |   |   |   |   |   |   |--- class: 0
|   |   |   |   |   |   |   |--- ptp_p99_p01 >  1.02
|   |   |   |   |   |   |   |   |--- class: 0
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
|   |   |--- pca_marg
```
