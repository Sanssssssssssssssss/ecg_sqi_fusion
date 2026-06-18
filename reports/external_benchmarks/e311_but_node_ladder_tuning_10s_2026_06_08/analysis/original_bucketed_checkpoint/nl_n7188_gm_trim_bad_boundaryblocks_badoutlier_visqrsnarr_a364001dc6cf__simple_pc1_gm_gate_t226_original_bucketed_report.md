# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf`
- Prediction mode: `simple_pc1_gm_gate_t226`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8324, macro-F1=0.8511, recall good/medium/bad=0.8139/0.8111/0.9353
- `original_test_all_10s+`: n=8477, acc=0.7582, macro-F1=0.6401, recall good/medium/bad=0.9305/0.6570/0.3212
- `original_test_good_medium_only`: n=8066, acc=0.7804, macro-F1=0.5250, recall good/medium/bad=0.9305/0.6570/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0445, macro-F1=0.0284, recall good/medium/bad=0.0000/0.0000/0.0445
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7836, macro-F1=0.7209, recall good/medium/bad=0.9305/0.6570/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.7655, macro-F1=0.5126, recall good/medium/bad=0.9298/0.6134/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9998, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/0.9998
- `original_all_bad_outlier_stress`: n=1201, acc=0.7161, macro-F1=0.2782, recall good/medium/bad=0.0000/0.0000/0.7161

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf__simple_pc1_gm_gate_t226_original_bucketed_confusions.png)
