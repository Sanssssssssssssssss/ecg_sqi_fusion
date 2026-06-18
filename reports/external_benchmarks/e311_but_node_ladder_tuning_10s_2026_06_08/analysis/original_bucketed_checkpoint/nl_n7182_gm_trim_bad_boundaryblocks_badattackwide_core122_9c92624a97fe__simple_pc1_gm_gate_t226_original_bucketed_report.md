# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7182_gm_trim_bad_boundaryblocks_badattackwide_core122_9c92624a97fe`
- Prediction mode: `simple_pc1_gm_gate_t226`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8275, macro-F1=0.8367, recall good/medium/bad=0.8102/0.8081/0.9220
- `original_test_all_10s+`: n=8477, acc=0.7457, macro-F1=0.5630, recall good/medium/bad=0.9242/0.6543/0.1484
- `original_test_good_medium_only`: n=8066, acc=0.7761, macro-F1=0.5347, recall good/medium/bad=0.9242/0.6543/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.2857, macro-F1=0.1481, recall good/medium/bad=0.0000/0.0000/0.2857
- `original_test_bad_outlier_stress`: n=292, acc=0.0925, macro-F1=0.0564, recall good/medium/bad=0.0000/0.0000/0.0925
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7690, macro-F1=0.5644, recall good/medium/bad=0.9242/0.6543/0.2857
- `original_test_good_medium_overlap`: n=7492, acc=0.7591, macro-F1=0.5213, recall good/medium/bad=0.9234/0.6069/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9792, macro-F1=0.3298, recall good/medium/bad=0.0000/0.0000/0.9792
- `original_all_bad_outlier_stress`: n=1201, acc=0.7277, macro-F1=0.2808, recall good/medium/bad=0.0000/0.0000/0.7277

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7182_gm_trim_bad_boundaryblocks_badattackwide_core122_9c92624a97fe__simple_pc1_gm_gate_t226_original_bucketed_confusions.png)
