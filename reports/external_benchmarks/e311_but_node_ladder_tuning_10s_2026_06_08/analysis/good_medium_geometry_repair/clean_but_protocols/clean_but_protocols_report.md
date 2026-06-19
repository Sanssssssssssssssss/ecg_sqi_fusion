# Clean BUT Protocol Materialization

These are fixed-length 10s protocol variants. No variable-length model is used here.

| policy | n | good | medium | bad | record_count | path |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| margin_ge_2s_keep_outlier | 29959 | 15381 | 9411 | 5167 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_2s_keep_outlier` |
| margin_ge_5s_keep_outlier | 29410 | 15042 | 9212 | 5156 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier` |
| margin_ge_2s_drop_outlier | 21914 | 11458 | 6374 | 4082 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_2s_drop_outlier` |
| margin_ge_5s_drop_outlier | 21575 | 11228 | 6265 | 4082 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier` |
| margin_ge_8s_drop_outlier | 21291 | 11057 | 6152 | 4082 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_8s_drop_outlier` |
| margin_ge_10s_drop_outlier | 21087 | 10926 | 6080 | 4081 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_10s_drop_outlier` |
| clean_core_plus_overlap_margin2 | 17823 | 11458 | 6365 | 0 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\clean_core_plus_overlap_margin2` |
| clean_core_only_margin2 | 4944 | 3196 | 1748 | 0 | 16 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\clean_core_only_margin2` |
| segment_10_20s_keep_outlier | 904 | 469 | 388 | 47 | 16 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\segment_10_20s_keep_outlier` |
| segment_10_30s_keep_outlier | 1710 | 893 | 728 | 89 | 16 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\segment_10_30s_keep_outlier` |
| segment_10_60s_keep_outlier | 3677 | 2017 | 1524 | 136 | 18 | `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\segment_10_60s_keep_outlier` |

## How To Use

Each policy directory contains `signals.npz`, `metadata.csv`, `original_region_atlas.csv`, `window_segment_margins.csv`, and `audit.json`. The atlas `idx` is reset to the new signal row index, while `source_idx` preserves the row in the original p1 protocol.

Recommended first training diagnostics:

1. `margin_ge_5s_keep_outlier`: strict fixed-10s margin cleanup while preserving stress/outlier labels.
2. `margin_ge_5s_drop_outlier`: strict fixed-10s clean body, bad core only.
3. `clean_core_plus_overlap_margin2`: good/medium learnable-body sanity, no bad.

Full BUT and bad outlier stress must remain separate report-only evaluations.
