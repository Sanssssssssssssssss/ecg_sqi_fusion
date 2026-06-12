# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7155_gm_trim_bad_boundaryblocks_breakthrough_softbala_5c50f0ee6d7a`
- Prediction mode: `medium_guarded_pmed001`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8350, macro-F1=0.8573, recall good/medium/bad=0.7754/0.8940/0.9088
- `original_test_all_10s+`: n=8477, acc=0.7819, macro-F1=0.5317, recall good/medium/bad=0.7014/0.9207/0.0000
- `original_test_good_medium_only`: n=8066, acc=0.8217, macro-F1=0.5434, recall good/medium/bad=0.7014/0.9207/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8098, macro-F1=0.5400, recall good/medium/bad=0.7014/0.9207/0.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8081, macro-F1=0.5363, recall good/medium/bad=0.6982/0.9098/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9706, macro-F1=0.3284, recall good/medium/bad=0.0000/0.0000/0.9706
- `original_all_bad_outlier_stress`: n=1201, acc=0.6986, macro-F1=0.2742, recall good/medium/bad=0.0000/0.0000/0.6986

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7155_gm_trim_bad_boundaryblocks_breakthrough_softbala_5c50f0ee6d7a__medium_guarded_pmed001_original_bucketed_confusions.png)
