# Interpretable Clean PTB/BUT Cross-Dataset Checks

Fixed 10s only. No variable-length inputs. Classifier input is waveform-derived channels; interpretable features supervise training only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| but_clean_margin_ge_5s_drop_outlier_dualview_convtx_hier | train_domain_val | 0.863158 | 0.469250 | 0.855072 | 1.000000 | 0.000000 | 90 | 0 | 1 |
| but_clean_margin_ge_5s_drop_outlier_dualview_convtx_hier | ptb_synthetic_test | 0.788314 | 0.558814 | 0.901674 | 0.898539 | 0.000000 | 47 | 125 | 241 |
| but_clean_margin_ge_5s_drop_outlier_dualview_convtx_hier | but_clean_margin_ge_5s_drop_outlier_test | 0.929978 | 0.911671 | 0.999004 | 0.904670 | 0.788136 | 1 | 198 | 25 |

## Checkpoints

- `but_clean_margin_ge_5s_drop_outlier_dualview_convtx_hier` (but_clean): best_epoch=3, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\interpretable_clean_cross_dataset\but_clean_margin_ge_5s_drop_outlier_dualview_convtx_hier\ckpt_best.pt`

## Interpretation Contract

- `ptb_synthetic_test` measures target-aware PTB synthetic learnability.
- `but_clean_*_test` measures cleaned fixed-10s BUT learnability.
- `legacy_full_*` buckets are emitted only when explicitly requested and are not the modeling target.
- Large asymmetry between within-domain and cross-domain performance is evidence of domain/split/label-policy shift, not proof that variable-length is required.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\interpretable_clean_cross_dataset\interpretable_clean_cross_dataset_margin_ge_5s_drop_outlier_dualview_convtx_hier_metrics.csv`