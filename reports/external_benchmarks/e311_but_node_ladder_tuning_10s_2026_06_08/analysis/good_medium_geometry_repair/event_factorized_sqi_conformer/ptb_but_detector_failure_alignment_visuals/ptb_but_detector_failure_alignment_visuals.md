# PTB/BUT Detector-Failure Alignment Visuals

These plots compare clean BUT train/test against old aligned PTB, aggressive low-QRS PTB, and the new mixed detector-failure PTB variant.

## Class Counts

| source | class_name | n |
| --- | --- | --- |
| BUT test | bad | 118 |
| BUT test | good | 1004 |
| BUT test | medium | 2077 |
| BUT train | bad | 3963 |
| BUT train | good | 9603 |
| BUT train | medium | 4145 |
| PTB v1 | bad | 3000 |
| PTB v1 | good | 3000 |
| PTB v1 | medium | 3000 |
| PTB v10 extremeMed | bad | 3000 |
| PTB v10 extremeMed | good | 3000 |
| PTB v10 extremeMed | medium | 3000 |
| PTB v11 styleReplay | bad | 3000 |
| PTB v11 styleReplay | good | 3000 |
| PTB v11 styleReplay | medium | 3000 |
| PTB v6 aggressive | bad | 3000 |
| PTB v6 aggressive | good | 3000 |
| PTB v6 aggressive | medium | 3000 |
| PTB v7 mixed | bad | 3000 |
| PTB v7 mixed | good | 3000 |
| PTB v7 mixed | medium | 3000 |
| PTB v8 visibleQRS | bad | 3000 |
| PTB v8 visibleQRS | good | 3000 |
| PTB v8 visibleQRS | medium | 3000 |
| PTB v9 strongMed | bad | 3000 |
| PTB v9 strongMed | good | 3000 |
| PTB v9 strongMed | medium | 3000 |

## Feature Distribution

![feature medians](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\ptb_but_detector_failure_alignment_visuals\feature_median_p10_p90_by_class.png)

## Medium-Class KDE

![medium kde](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\ptb_but_detector_failure_alignment_visuals\medium_distribution_kde.png)

## Representative Waveforms

![waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\ptb_but_detector_failure_alignment_visuals\but_vs_ptb_v7_waveform_contact_sheet.png)

## Files

- Feature rows: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\ptb_but_detector_failure_alignment_visuals\ptb_but_alignment_features.csv`
- Feature summary: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\ptb_but_detector_failure_alignment_visuals\ptb_but_alignment_feature_summary.csv`
- Waveform selection: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\ptb_but_detector_failure_alignment_visuals\waveform_contact_sheet_selection.csv`