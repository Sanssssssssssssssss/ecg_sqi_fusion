# PTB v14 Feature-Matched Model -> BUT Keep-Outlier Report

- Training/normalization source: PTB v14 train only.
- BUT is report-only; no BUT row used for training/selection.
- This is a transfer sanity check, not model selection.

## Metrics

| candidate | but_split | n_rows | acc | macro_f1 | good_recall | medium_recall | bad_recall | bad_to_nonbad | nonbad_to_bad | subtype_acc | bad_subtype_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_baseline_keep | train | 21117 | 0.794668 | 0.79034 | 0.819278 | 0.562679 | 0.992445 | 36 | 199 | 0.223659 | 0 |
| E1_baseline_keep | val | 991 | 0.957619 | 0.858689 | 0.976498 | 0.683333 | 0.9625 | 3 | 9 | 0.102926 | 0.85 |
| E1_baseline_keep | test | 7302 | 0.581074 | 0.515639 | 0.917857 | 0.331884 | 0.379421 | 193 | 143 | 0.324843 | 0.00321543 |
| E1_subtype_aux_keep | train | 21117 | 0.798219 | 0.805402 | 0.766088 | 0.689563 | 0.992655 | 35 | 198 | 0.581333 | 0 |
| E1_subtype_aux_keep | val | 991 | 0.871847 | 0.758859 | 0.869565 | 0.8 | 0.95 | 4 | 10 | 0.887992 | 0.85 |
| E1_subtype_aux_keep | test | 7302 | 0.648452 | 0.564432 | 0.735714 | 0.599847 | 0.395498 | 188 | 234 | 0.589838 | 0.0257235 |

## Output Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_v14_bad_subtype_featurematched_sanity\ptb_v14_to_but_keepoutlier_metrics.csv`
- Subtype CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_v14_bad_subtype_featurematched_sanity\ptb_v14_to_but_keepoutlier_subtype_metrics.csv`