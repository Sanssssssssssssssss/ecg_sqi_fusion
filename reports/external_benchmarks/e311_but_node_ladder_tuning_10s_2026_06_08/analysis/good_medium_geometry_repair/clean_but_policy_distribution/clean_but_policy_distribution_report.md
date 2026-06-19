# Clean BUT Policy Distribution

## Main Takeaways

- Full fixed-10s BUT has **32,956** windows: good 17,043, medium 10,628, bad 5,285.
- The current clean body (`margin>=5s` + drop `outlier_low_confidence`) keeps **21,575** windows: good 11,228, medium 6,265, bad 4,082.
- The duration-only `10-60s` sanity slice keeps only **3,677** windows and almost no stable bad body, so it is too small as a main protocol.
- Good/medium overlap is intentionally retained; low-confidence outliers are held out as stress diagnostics rather than hidden.

## Why This Is Defensible

The clean protocol separates two concepts: label-window reliability and waveform-geometry confidence. A sample is not removed because the model gets it wrong. It is held out when the 10s window sits too close to a label transition or when the atlas marks it as low-confidence/outlier geometry. The correct claim is therefore `clean high-confidence BUT`, not full BUT.

## Figures

![fig1_policy_class_retention](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_policy_distribution\fig1_policy_class_retention.png)

![fig2_region_composition](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_policy_distribution\fig2_region_composition.png)

![fig3_margin_distribution_by_region](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_policy_distribution\fig3_margin_distribution_by_region.png)

![fig4_pca_clean_vs_stress](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_policy_distribution\fig4_pca_clean_vs_stress.png)

![fig5_waveform_feature_boxes](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_policy_distribution\fig5_waveform_feature_boxes.png)

![fig6_cleaning_contract](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_policy_distribution\fig6_cleaning_contract.png)

## Policy Count Table

| policy | split_scope | n | record_count | good | medium | bad |
| --- | --- | --- | --- | --- | --- | --- |
| Full 10s protocol | all | 32956 | 18 | 17043 | 10628 | 5285 |
| Full 10s protocol | test | 8477 | 3 | 3640 | 4426 | 411 |
| Margin >=5s, keep outlier | all | 29410 | 18 | 15042 | 9212 | 5156 |
| Margin >=5s, keep outlier | test | 7302 | 3 | 3080 | 3911 | 311 |
| Margin >=5s, drop outlier | all | 21575 | 18 | 11228 | 6265 | 4082 |
| Margin >=5s, drop outlier | test | 3199 | 3 | 1004 | 2077 | 118 |
| Margin >=10s, drop outlier | all | 21087 | 18 | 10926 | 6080 | 4081 |
| Margin >=10s, drop outlier | test | 3108 | 3 | 956 | 2035 | 117 |
| Segment 10-60s, keep outlier | all | 3677 | 18 | 2017 | 1524 | 136 |
| Segment 10-60s, keep outlier | test | 1313 | 3 | 617 | 578 | 118 |
