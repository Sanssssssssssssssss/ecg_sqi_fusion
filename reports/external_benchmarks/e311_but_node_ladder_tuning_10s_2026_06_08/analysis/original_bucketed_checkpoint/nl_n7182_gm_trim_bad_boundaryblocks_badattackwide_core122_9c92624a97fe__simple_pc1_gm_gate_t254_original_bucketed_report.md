# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7182_gm_trim_bad_boundaryblocks_badattackwide_core122_9c92624a97fe`
- Prediction mode: `simple_pc1_gm_gate_t254`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8134, macro-F1=0.8269, recall good/medium/bad=0.7697/0.8295/0.9220
- `original_test_all_10s+`: n=8477, acc=0.7547, macro-F1=0.5700, recall good/medium/bad=0.9110/0.6826/0.1484
- `original_test_good_medium_only`: n=8066, acc=0.7856, macro-F1=0.5419, recall good/medium/bad=0.9110/0.6826/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.2857, macro-F1=0.1481, recall good/medium/bad=0.0000/0.0000/0.2857
- `original_test_bad_outlier_stress`: n=292, acc=0.0925, macro-F1=0.0564, recall good/medium/bad=0.0000/0.0000/0.0925
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7784, macro-F1=0.5716, recall good/medium/bad=0.9110/0.6826/0.2857
- `original_test_good_medium_overlap`: n=7492, acc=0.7694, macro-F1=0.5297, recall good/medium/bad=0.9100/0.6391/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9792, macro-F1=0.3298, recall good/medium/bad=0.0000/0.0000/0.9792
- `original_all_bad_outlier_stress`: n=1201, acc=0.7277, macro-F1=0.2808, recall good/medium/bad=0.0000/0.0000/0.7277

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7182_gm_trim_bad_boundaryblocks_badattackwide_core122_9c92624a97fe__simple_pc1_gm_gate_t254_original_bucketed_confusions.png)
