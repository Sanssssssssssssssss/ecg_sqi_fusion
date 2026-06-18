# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7180_gm_trim_bad_boundaryblocks_origbridge_badstressp_f9488c4af248`
- Prediction mode: `simple_pc1_gm_gate_t226`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8320, macro-F1=0.8531, recall good/medium/bad=0.8153/0.8199/0.9097
- `original_test_all_10s+`: n=8477, acc=0.7510, macro-F1=0.5205, recall good/medium/bad=0.9321/0.6706/0.0122
- `original_test_good_medium_only`: n=8066, acc=0.7886, macro-F1=0.5254, recall good/medium/bad=0.9321/0.6706/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.0420, macro-F1=0.0269, recall good/medium/bad=0.0000/0.0000/0.0420
- `original_test_bad_outlier_stress`: n=292, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7778, macro-F1=0.5484, recall good/medium/bad=0.9321/0.6706/0.0420
- `original_test_good_medium_overlap`: n=7492, acc=0.7724, macro-F1=0.5126, recall good/medium/bad=0.9314/0.6252/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9718, macro-F1=0.3286, recall good/medium/bad=0.0000/0.0000/0.9718
- `original_all_bad_outlier_stress`: n=1201, acc=0.6986, macro-F1=0.2742, recall good/medium/bad=0.0000/0.0000/0.6986

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7180_gm_trim_bad_boundaryblocks_origbridge_badstressp_f9488c4af248__simple_pc1_gm_gate_t226_original_bucketed_confusions.png)
