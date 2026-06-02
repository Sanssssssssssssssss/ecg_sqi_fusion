# Focused 2h Full-Joint SQI Report

State: `running`, completed: `10`.

Data integrity: full split training each epoch, expected train/val/test = `10935/2184/2202`; this is not cached-head-only.

Baseline no-SQI target: acc `0.962307`, recalls `0.94959/0.95095/0.98638`.

## Top Candidates

| rank | variant | scale | class weight | acc | good | medium | bad | denoise | corrected/harmed | path |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | gated | 0.0350 | 1,1.40,1.70 | 0.964578 | 0.95504 | 0.94959 | 0.98910 | 2.8838 | 6/1 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_gated_sc0p035_cw1c1p40c1p70_mg0p016_gq0p006_lq0p12_e8_5c641a42 |
| 2 | gated | 0.0400 | 1,1.38,1.70 | 0.964124 | 0.95640 | 0.94687 | 0.98910 | 2.9106 | 9/4 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_gated_sc0p04_cw1c1p38c1p70_mg0p01_gq0p006_lq0p1_e8_c39faa17 |
| 3 | delta | 0.0175 | 1,1.44,1.72 | 0.961853 | 0.94142 | 0.95504 | 0.98910 | 2.9831 | 8/1 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_delta_sc0p0175_cw1c1p44c1p72_mg0p018_gq0p005_lq0p1_e8_b1617c49 |
| 4 | gated | 0.0300 | 1,1.36,1.70 | 0.961853 | 0.95777 | 0.94142 | 0.98638 | 2.9039 | 6/4 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_gated_sc0p03_cw1c1p36c1p70_mg0p012_gq0p006_lq0p1_e8_1520dc58 |
| 5 | gated | 0.0325 | 1,1.38,1.72 | 0.961399 | 0.94687 | 0.94959 | 0.98774 | 2.9070 | 3/0 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_gated_sc0p0325_cw1c1p38c1p72_mg0p014_gq0p006_lq0p12_e8_7290e56b |
| 6 | none | 0.0000 | 1,1.45,1.65 | 0.961399 | 0.94959 | 0.94823 | 0.98638 | 2.9834 | 0/0 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_none_sc0p0_cw1c1p45c1p65_mg0p0_gq0p0_lq0p0_e8_c635a5ff |
| 7 | delta | 0.0150 | 1,1.42,1.72 | 0.960945 | 0.94687 | 0.94959 | 0.98638 | 2.8524 | 0/0 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_delta_sc0p015_cw1c1p42c1p72_mg0p019_gq0p006_lq0p1_e8_34eb2fc8 |
| 8 | gated | 0.0375 | 1,1.36,1.66 | 0.959128 | 0.94414 | 0.94959 | 0.98365 | 3.0617 | 18/2 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_gated_sc0p0375_cw1c1p36c1p66_mg0p01_gq0p004_lq0p08_e8_13d439cc |
| 9 | delta | 0.0200 | 1,1.44,1.74 | 0.957766 | 0.93324 | 0.95640 | 0.98365 | 3.0189 | 8/1 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_delta_sc0p02_cw1c1p44c1p74_mg0p019_gq0p003_lq0p11_e8_12a469ac |
| 10 | gated | 0.0350 | 1,1.38,1.68 | 0.957312 | 0.94005 | 0.94823 | 0.98365 | 3.0746 | 12/3 | E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\artifact\full_sqi_2h\focused2h_gated_sc0p035_cw1c1p38c1p68_mg0p012_gq0p005_lq0p1_e8_33421b0f |

## Visuals

- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused_2h_pareto.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused_2h_train_curves.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p035_cw1c1p40c1p70_mg0p016_gq0p006_lq0p12_e8_5c641a42_balanced_gallery.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p035_cw1c1p40c1p70_mg0p016_gq0p006_lq0p12_e8_5c641a42_corrected_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p035_cw1c1p40c1p70_mg0p016_gq0p006_lq0p12_e8_5c641a42_harmed_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p035_cw1c1p40c1p70_mg0p016_gq0p006_lq0p12_e8_5c641a42_hard_bad.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p04_cw1c1p38c1p70_mg0p01_gq0p006_lq0p1_e8_c39faa17_balanced_gallery.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p04_cw1c1p38c1p70_mg0p01_gq0p006_lq0p1_e8_c39faa17_corrected_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p04_cw1c1p38c1p70_mg0p01_gq0p006_lq0p1_e8_c39faa17_harmed_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p04_cw1c1p38c1p70_mg0p01_gq0p006_lq0p1_e8_c39faa17_hard_bad.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_delta_sc0p0175_cw1c1p44c1p72_mg0p018_gq0p005_lq0p1_e8_b1617c49_balanced_gallery.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_delta_sc0p0175_cw1c1p44c1p72_mg0p018_gq0p005_lq0p1_e8_b1617c49_corrected_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_delta_sc0p0175_cw1c1p44c1p72_mg0p018_gq0p005_lq0p1_e8_b1617c49_harmed_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_delta_sc0p0175_cw1c1p44c1p72_mg0p018_gq0p005_lq0p1_e8_b1617c49_hard_bad.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p03_cw1c1p36c1p70_mg0p012_gq0p006_lq0p1_e8_1520dc58_balanced_gallery.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p03_cw1c1p36c1p70_mg0p012_gq0p006_lq0p1_e8_1520dc58_corrected_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p03_cw1c1p36c1p70_mg0p012_gq0p006_lq0p1_e8_1520dc58_harmed_by_sqi.png`
- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid\reports\focused_2h\focused2h_gated_sc0p03_cw1c1p36c1p70_mg0p012_gq0p006_lq0p1_e8_1520dc58_hard_bad.png`
