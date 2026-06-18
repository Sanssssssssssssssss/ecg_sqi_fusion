# Waveform Primitive Learnability Audit

Created: 2026-06-18T11:55:49

This audit uses synthetic/PTB waveform-derived primitive stats only. Original BUT rows are report-only.

## Key Hard Targets

| target | category | max_abs_pearson | best_primitive | ridge_r2 | trees_r2 | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| pc3 | atlas_geometry_proxy | 0.3119 | ch1_detector_agreement_bank_03 | 0.0477 | 0.1305 | weak waveform fact; treat as geometry proxy, not primary aux target |
| knn_label_purity | atlas_geometry_proxy | 0.3138 | ch1_qrs_visibility_bank_16 | 0.0380 | 0.1241 | weak waveform fact; treat as geometry proxy, not primary aux target |
| detector_agreement | waveform_qrs_detector | 0.2635 | ch1_atlas_27 | -0.0401 | 0.0223 | weakly learnable from current primitives; redesign target or generator labels |
| baseline_step | waveform_baseline_flatline | 0.3906 | ch1_sparse_event_bank_13 | -0.0717 | -0.0140 | weakly learnable from current primitives; redesign target or generator labels |
| boundary_confidence | atlas_geometry_proxy | 0.3387 | ch1_atlas_24 | -0.1015 | -0.0360 | weak waveform fact; treat as geometry proxy, not primary aux target |
| pc2 | atlas_geometry_proxy | 0.1681 | ch0_stress_bank_01 | -0.1789 | -0.1242 | weak waveform fact; treat as geometry proxy, not primary aux target |
| sqi_basSQI | waveform_baseline_flatline | 0.2428 | ch1_sparse_event_bank_13 | -0.1467 | -0.1350 | weakly learnable from current primitives; redesign target or generator labels |
| qrs_visibility | waveform_qrs_detector | 0.3688 | ch0_baseline_frequency_bank_10 | -0.3998 | -0.2781 | weakly learnable from current primitives; redesign target or generator labels |

## Primitive Classifier Diagnostic

These are diagnostic waveform-primitive baselines, not final models.

```json
{
  "primitive_logreg_synthetic_test": {
    "acc": 0.9969246540235777,
    "good_recall": 0.99581589958159,
    "medium_recall": 0.997564935064935,
    "bad_recall": 0.995850622406639
  },
  "primitive_logreg_original_all_10s+_report_only": {
    "acc": 0.8244932637456002,
    "good_recall": 0.7683506424925189,
    "medium_recall": 0.8610274745954084,
    "bad_recall": 0.9320719016083254
  },
  "primitive_logreg_original_test_all_10s+_report_only": {
    "acc": 0.799575321458063,
    "good_recall": 0.8717032967032967,
    "medium_recall": 0.7871667419792138,
    "bad_recall": 0.2944038929440389
  },
  "primitive_trees_synthetic_test": {
    "acc": 0.9943618657098924,
    "good_recall": 0.9895397489539749,
    "medium_recall": 0.9967532467532467,
    "bad_recall": 0.991701244813278
  },
  "primitive_trees_original_all_10s+_report_only": {
    "acc": 0.8255856293239471,
    "good_recall": 0.7823153200727572,
    "medium_recall": 0.8421151674821227,
    "bad_recall": 0.9318826868495743
  },
  "primitive_trees_original_test_all_10s+_report_only": {
    "acc": 0.7827061460422319,
    "good_recall": 0.9082417582417582,
    "medium_recall": 0.7252598282873927,
    "bad_recall": 0.2895377128953771
  }
}
```

## Outputs

- CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_primitive_learnability_qrs_stress_v5_robust3.csv`
- JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_primitive_learnability_qrs_stress_v5_robust3.json`
