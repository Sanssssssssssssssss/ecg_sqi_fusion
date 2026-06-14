# Current 7SQI Baseline

Fresh current-experiment baseline: train on the N17043 synthetic train split, select on synthetic val only, evaluate the same original BUT held-out split. Original BUT is not used for training or selection.

## Setup

- Synthetic variant: `nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6`
- Features: `sqi_bSQI, sqi_iSQI, sqi_kSQI, sqi_sSQI, sqi_pSQI, sqi_fSQI, sqi_basSQI`
- Classifier candidates: RBF SVM, balanced RBF SVM, MLP
- Selected model: `7SQI_MLP`
- Selected params: `{"alpha": 0.0001, "hidden_layer_sizes": [64]}`

## Main Results

| Model | Bucket | n | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7SQI baseline | synthetic_test | 1951 | 0.777037 | 0.725383 | 0.412134 | 0.928571 | 0.726141 |
| 7SQI baseline | original_test_all_10s+ | 8477 | 0.635720 | 0.388833 | 0.275275 | 0.991188 | 0.000000 |
| 7SQI baseline | original_all_10s+ | 32956 | 0.721598 | 0.769978 | 0.508831 | 0.970173 | 0.907852 |
| UFormer+47 SQI/geometry branch, threshold p_bad>=0.13 | original_test_all_10s+ | 8477 | 0.963548 | 0.930683 | 0.956319 | 0.972887 | 0.927007 |
| UFormer+47 SQI/geometry branch, threshold p_bad>=0.13 | original_all_10s+ | 32956 | 0.985374 | 0.985233 | 0.988852 | 0.975348 | 0.994324 |

## Original Buckets

| Bucket | n | Acc | Macro-F1 | Good R | Medium R | Bad R | Confusion [g,m,b]x[g,m,b] |
|---|---:|---:|---:|---:|---:|---:|---|
| original_test_all_10s+ | 8477 | 0.635720 | 0.388833 | 0.275275 | 0.991188 | 0.000000 | `[[1002, 2638, 0], [39, 4387, 0], [46, 365, 0]]` |
| original_all_10s+ | 32956 | 0.721598 | 0.769978 | 0.508831 | 0.970173 | 0.907852 | `[[8672, 8371, 0], [317, 10311, 0], [47, 440, 4798]]` |
| original_test_without_bad_outlier_stress | 8185 | 0.658400 | 0.395485 | 0.275275 | 0.991188 | 0.000000 | `[[1002, 2638, 0], [39, 4387, 0], [0, 119, 0]]` |
| bad_core_nearboundary | 119 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `[[0, 0, 0], [0, 0, 0], [0, 119, 0]]` |
| bad_outlier_stress | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `[[0, 0, 0], [0, 0, 0], [46, 246, 0]]` |

## Interpretation

- The 7SQI-only baseline is a deliberately narrow feature baseline: it sees SQI summary statistics only, no waveform tokens and no 47-feature geometry branch.
- Selection is clean: the original BUT rows are held out and report-only.
- The gap versus the UFormer+geometry branch quantifies how much the current full model gains from waveform representation plus target-aware geometry/SQI fusion.
