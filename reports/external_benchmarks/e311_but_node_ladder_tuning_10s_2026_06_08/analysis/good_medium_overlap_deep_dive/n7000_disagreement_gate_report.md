# N7000 Good/Medium Disagreement Gate Analysis

Report-only analysis. Original BUT is not used for selection. This compares the medium-strong compact N7000 model against the good-strong dense N7000 model.

## Compared Models

- Medium-strong: `nl_n7000_gm_trim_bad_scan_014_sc_overlap_compact_pca_core_ddc377ccab88` / `calibrated` acc `0.9346`, good `0.8646`, medium `0.9670`, bad `0.9993`.
- Good-strong: `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg045_vqm045__d0f7b043135d` / `medium_guarded_pmed0005` acc `0.9326`, good `0.9873`, medium `0.8556`, bad `0.9709`.
- Balanced dense reference: `nl_n7000_gm_trim_bad_dense_joint_n6800full_lcg025_vqm035__bfc4b452a166` / `medium_guarded_pmed0005` acc `0.9297`, good `0.9471`, medium `0.8874`, bad `0.9723`.

## Core Counts

- Good rescued by dense: `860` rows.
- Medium lost by dense: `813` rows.
- GM-focus good rescued: `860` rows.
- GM-focus medium lost: `813` rows.

## Top Feature Separators

| feature | KS | threshold | direction | balanced acc | rescued median | lost median |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `pc1` | 0.998 | -1.561 | low_is_positive | 0.989 | -3.658 | -0.4761 |
| `flatline_ratio` | 0.965 | 0.1537 | high_is_positive | 0.982 | 0.2498 | 0.09047 |
| `pc3` | 0.930 | 1.356 | low_is_positive | 0.963 | -1.282 | 2.846 |
| `sample_entropy_proxy` | 0.871 | 0.3926 | low_is_positive | 0.936 | 0.3335 | 0.457 |
| `qrs_visibility` | 0.867 | 0.5054 | high_is_positive | 0.929 | 0.6151 | 0.3264 |
| `non_qrs_diff_p95` | 0.827 | 0.07919 | low_is_positive | 0.912 | 0.05229 | 0.1176 |
| `diff_zero_crossing_rate` | 0.790 | 0.4223 | low_is_positive | 0.893 | 0.3462 | 0.4928 |
| `template_corr` | 0.765 | 0.6136 | high_is_positive | 0.880 | 0.6867 | 0.5572 |
| `band_30_45` | 0.596 | 0.02222 | low_is_positive | 0.796 | 0.0165 | 0.02966 |
| `knn_label_purity` | 0.528 | 0.9667 | high_is_positive | 0.764 | 1 | 0.8667 |

## Report-Only Gate Simulation

Default prediction is the medium-strong model. The gate switches to the good-strong model only when the two models disagree as medium-vs-good and the listed feature band is satisfied. This is diagnostic only, not a promoted classifier.

| gate features | switched | acc | macro-F1 | good | medium | bad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pc1+flatline_ratio` | 856 | 0.9815 | 0.9836 | 0.9863 | 0.9664 | 0.9993 |
| `pc1` | 873 | 0.9811 | 0.9833 | 0.9870 | 0.9647 | 0.9993 |
| `pc1+pc3` | 840 | 0.9805 | 0.9828 | 0.9839 | 0.9663 | 0.9993 |
| `flatline_ratio+pc3` | 831 | 0.9805 | 0.9827 | 0.9831 | 0.9669 | 0.9993 |
| `pc1+flatline_ratio+pc3` | 831 | 0.9805 | 0.9827 | 0.9831 | 0.9669 | 0.9993 |
| `flatline_ratio` | 875 | 0.9805 | 0.9827 | 0.9863 | 0.9637 | 0.9993 |

## Reading

If the top separators are stable, the next implementation step should be a narrow gate or generator that rescues good only in the rescued feature band while preserving visible-QRS medium rows. Do not continue broad one-sided class-weight sweeps.

## Artifacts

- `n7000_disagreement_feature_gate_candidates.csv`
- `n7000_report_only_feature_gate_eval.csv`
- `n7000_gm_focus_disagreement_rows.csv`
- `n7000_disagreement_pca.png`
