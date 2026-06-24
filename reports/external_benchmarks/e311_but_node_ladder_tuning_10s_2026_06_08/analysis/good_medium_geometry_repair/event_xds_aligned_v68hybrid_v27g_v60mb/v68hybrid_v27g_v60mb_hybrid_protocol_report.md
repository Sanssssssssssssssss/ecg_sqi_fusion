# v68hybrid_v27g_v60mb Hybrid PTB Protocol

- Created: `2026-06-23 17:28:15`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v68hybrid_v27g_v60mb\protocol_v68hybrid_v27g_v60mb_pc3000_s20260654`
- Good/bad source: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\protocol_v60raw_pc3000_s20260650`
- Good source: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v27_pca_subtype_gm_featurematched\protocol_v27_pca_subtype_pc3000_s20260622`
- Bad source: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\protocol_v60raw_pc3000_s20260650`
- Medium source: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\protocol_v60raw_pc3000_s20260650`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Rationale

v63raw improved medium boundary distribution but hurt cross-domain good and bad transfer. This hybrid protocol tests the explicit hypothesis that v60raw good/bad rows should be preserved while v63raw medium rows repair the medium distribution.

## Counts

| split | class_name | display_subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 65 |
| test | bad | bad_contact_reset_flatline | 65 |
| test | bad | bad_dense_right_island | 65 |
| test | bad | bad_detector_template_disagree | 65 |
| test | bad | bad_highfreq_detail_noise | 64 |
| test | bad | bad_low_qrs_visibility | 64 |
| test | bad | bad_other_boundary | 64 |
| test | medium | medium_clean_core | 75 |
| test | medium | medium_hard_baseline_lowqrs | 75 |
| test | medium | medium_isolated_lowqrs | 75 |
| test | medium | medium_outlier_or_bad_boundary | 75 |
| test | medium | medium_overlap_boundary | 75 |
| test | medium | medium_visible_qrs_detail | 75 |
| train | bad | bad_baseline_wander_lowfreq | 300 |
| train | bad | bad_contact_reset_flatline | 300 |
| train | bad | bad_dense_right_island | 300 |
| train | bad | bad_detector_template_disagree | 300 |
| train | bad | bad_highfreq_detail_noise | 300 |
| train | bad | bad_low_qrs_visibility | 300 |
| train | bad | bad_other_boundary | 300 |
| train | medium | medium_clean_core | 350 |
| train | medium | medium_hard_baseline_lowqrs | 350 |
| train | medium | medium_isolated_lowqrs | 350 |
| train | medium | medium_outlier_or_bad_boundary | 350 |
| train | medium | medium_overlap_boundary | 350 |
| train | medium | medium_visible_qrs_detail | 350 |
| val | bad | bad_baseline_wander_lowfreq | 64 |
| val | bad | bad_contact_reset_flatline | 64 |
| val | bad | bad_dense_right_island | 64 |
| val | bad | bad_detector_template_disagree | 64 |
| val | bad | bad_highfreq_detail_noise | 64 |
| val | bad | bad_low_qrs_visibility | 64 |
| val | bad | bad_other_boundary | 64 |
| val | medium | medium_clean_core | 75 |
| val | medium | medium_hard_baseline_lowqrs | 75 |
| val | medium | medium_isolated_lowqrs | 75 |
| val | medium | medium_outlier_or_bad_boundary | 75 |
| val | medium | medium_overlap_boundary | 75 |
| val | medium | medium_visible_qrs_detail | 75 |

## Files

- Metadata: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v68hybrid_v27g_v60mb\protocol_v68hybrid_v27g_v60mb_pc3000_s20260654\metadata.csv`
- Atlas: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v68hybrid_v27g_v60mb\protocol_v68hybrid_v27g_v60mb_pc3000_s20260654\original_region_atlas.csv`
- Signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v68hybrid_v27g_v60mb\protocol_v68hybrid_v27g_v60mb_pc3000_s20260654\signals.npz`
- Counts: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v68hybrid_v27g_v60mb\protocol_v68hybrid_v27g_v60mb_pc3000_s20260654\ptb_hybrid_subtype_counts.csv`