# BUT Split-Seed Capacity and Error Geometry Update

External-only diagnostic. BUT rows are used here only to diagnose capacity/split behavior. This must not be mixed with PTB->BUT external claims.

## Capacity Summary

| run | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall |
| --- | --- | --- | --- | --- | --- | --- |
| window_random_init4 | but_test | 0.917476 | 0.922753 | 0.941021 | 0.845322 | 0.986755 |
| hard_test_init6 | but_test | 0.846727 | 0.601288 | 0.909459 | 0.847207 | 0.0445205 |
| cand6529_init_seed20261022_4ep | but_test | 0.845884 | 0.728523 | 0.899381 | 0.997076 | 0.285714 |
| balanced_best_init6 | but_test | 0.840353 | 0.824099 | 0.896283 | 0.711502 | 0.761905 |
| cand184_init_seed20261021_4ep | but_test | 0.82668 | 0.826274 | 0.819154 | 0.846493 | 0.761905 |
| current_init_seed20261023_lr18e4_4ep | but_test | 0.803232 | 0.69814 | 0.931319 | 0.742431 | 0.323601 |
| but_waveform_capacity_currentbest_init | but_test | 0.801817 | 0.698493 | 0.928297 | 0.741527 | 0.3309 |
| but_waveform_capacity_currentbest_init_rows | but_test | 0.801817 | 0.698493 | 0.928297 | 0.741527 | 0.3309 |
| but_waveform_capacity | but_test | 0.719712 | 0.632719 | 0.957967 | 0.563714 | 0.289538 |

## Main Readout

- Current split direct BUT training still lands around `0.803` test acc: good is high, medium is middling, bad outlier is poor.
- Strict subject-level split seeds land around `0.827-0.846` test acc in this quick diagnostic. Which class fails changes with the held-out records: large record-100001 style tests hurt good/medium, while bad-heavy small tests hurt bad recall.
- Window-random diagnostic reaches `0.917`, so the waveform Transformer has intra-record capacity. The missing piece is record/region generalization, especially outlier_low_confidence morphology.
- The current test is dominated by record `111001`; medium outlier rows are often predicted good, while bad outlier rows split into good/medium. Record `125001` contributes a separate good->medium failure shell.

## Key Error Features

| tag | class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_init_seed20261023_lr18e4_4ep_visual | good | pca_margin | 0.930018 | 1.71499 | -1.2976 | -3.0126 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | good | class_margin_percentile | 0.930018 | 0.468051 | 0.0540398 | -0.414012 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | good | pca_margin_rank | 0.930018 | 0.468051 | 0.0540398 | -0.414012 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | good | sample_entropy_proxy | 0.904991 | 0.227202 | 0.501241 | 0.27404 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | good | pc1 | 0.899752 | -5.39166 | -1.42353 | 3.96812 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | good | row_pos | 0.899599 | 10621 | 16773 | 6152 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | good | boundary_confidence | 0.899599 | 0.593198 | 0.124244 | -0.468954 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | good | region_confidence | 0.891268 | 0.157181 | 0.031061 | -0.12612 | 3390 | 250 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | pc1 | 0.710838 | -0.46033 | -4.14398 | -3.68365 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | row_pos | 0.700919 | 22405 | 26893 | 4488 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | boundary_confidence | 0.700919 | 0.619844 | 0.103928 | -0.515916 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | region_confidence | 0.697374 | 0.53808 | 0.0259821 | -0.512098 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | pca_margin | 0.695226 | 1.97362 | -0.0135975 | -1.98722 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | class_margin_percentile | 0.695226 | 0.509738 | 0.10844 | -0.401298 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | pca_margin_rank | 0.695226 | 0.509738 | 0.10844 | -0.401298 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | medium | knn_label_purity | 0.680994 | 0.866667 | 0.1 | -0.766667 | 3286 | 1140 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | pca_margin | 0.929383 | 5.27078 | -6.27673 | -11.5475 | 133 | 278 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | class_margin_percentile | 0.929383 | 0.0789026 | 0.0349101 | -0.0439924 | 133 | 278 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | pca_margin_rank | 0.929383 | 0.0789026 | 0.0349101 | -0.0439924 | 133 | 278 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | row_pos | 0.922838 | 32541 | 32816.5 | 275.5 | 133 | 278 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | boundary_confidence | 0.922838 | 0.389658 | 0.0228382 | -0.36682 | 133 | 278 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | region_confidence | 0.922838 | 0.303933 | 0.00570956 | -0.298224 | 133 | 278 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | knn_label_purity | 0.912046 | 0.966667 | 0 | -0.966667 | 133 | 278 |
| current_init_seed20261023_lr18e4_4ep_visual | bad | pc7 | 0.887191 | 3.25041 | -0.442154 | -3.69257 | 133 | 278 |
| cand184_init_seed20261021_4ep_visual | good | row_pos | 0.71396 | 7716.5 | 15379 | 7662.5 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | good | boundary_confidence | 0.71396 | 0.679602 | 0.370752 | -0.30885 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | good | pca_margin | 0.711552 | 1.79037 | -0.857648 | -2.64801 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | good | class_margin_percentile | 0.711552 | 0.48316 | 0.0902423 | -0.392918 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | good | pca_margin_rank | 0.711552 | 0.48316 | 0.0902423 | -0.392918 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | good | region_confidence | 0.698899 | 0.59805 | 0.106204 | -0.491846 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | good | flatline_ratio | 0.680155 | 0.241793 | 0.115292 | -0.126501 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | good | sample_entropy_proxy | 0.663535 | 0.334888 | 0.408788 | 0.0738999 | 5132 | 1133 |
| cand184_init_seed20261021_4ep_visual | medium | flatline_ratio | 0.773696 | 0.0936749 | 0.182546 | 0.0888711 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | medium | sample_entropy_proxy | 0.73186 | 0.49171 | 0.366299 | -0.125411 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | medium | pc1 | 0.708136 | -0.508326 | -2.31013 | -1.80181 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | medium | row_pos | 0.693843 | 21572 | 26063 | 4491 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | medium | boundary_confidence | 0.693843 | 0.671613 | 0.240456 | -0.431157 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | medium | pca_margin | 0.692381 | 2.13636 | -0.313587 | -2.44994 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | medium | class_margin_percentile | 0.692381 | 0.560877 | 0.0559842 | -0.504893 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | medium | pca_margin_rank | 0.692381 | 0.560877 | 0.0559842 | -0.504893 | 2305 | 418 |
| cand184_init_seed20261021_4ep_visual | bad | hjorth_mobility | 0.9875 | 1.88861 | 0.447535 | -1.44107 | 80 | 25 |
| cand184_init_seed20261021_4ep_visual | bad | hjorth_complexity | 0.9875 | 1.00606 | 2.31634 | 1.31028 | 80 | 25 |
| cand184_init_seed20261021_4ep_visual | bad | wavelet_e4 | 0.9875 | 0.862985 | 0.00312839 | -0.859857 | 80 | 25 |
| cand184_init_seed20261021_4ep_visual | bad | zero_crossing_rate | 0.975 | 0.79984 | 0.10008 | -0.69976 | 80 | 25 |
| cand184_init_seed20261021_4ep_visual | bad | pca_margin | 0.975 | 9.56066 | -7.17643 | -16.7371 | 80 | 25 |
| cand184_init_seed20261021_4ep_visual | bad | class_margin_percentile | 0.975 | 0.101798 | 0.0162725 | -0.0855251 | 80 | 25 |
| cand184_init_seed20261021_4ep_visual | bad | pca_margin_rank | 0.975 | 0.101798 | 0.0162725 | -0.0855251 | 80 | 25 |
| cand184_init_seed20261021_4ep_visual | bad | row_pos | 0.9625 | 32437.5 | 32619 | 181.5 | 80 | 25 |
| cand6529_init_seed20261022_4ep_visual | good | knn_label_purity | 0.82121 | 1 | 0.366667 | -0.633333 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | good | row_pos | 0.786681 | 8719 | 16534 | 7815 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | good | boundary_confidence | 0.786681 | 0.649475 | 0.187718 | -0.461757 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | good | pca_margin | 0.778181 | 1.82584 | -1.42464 | -3.25048 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | good | class_margin_percentile | 0.778181 | 0.490348 | 0.0458839 | -0.444464 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | good | pca_margin_rank | 0.778181 | 0.490348 | 0.0458839 | -0.444464 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | good | pc1 | 0.761287 | -3.74628 | -0.963623 | 2.78266 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | good | flatline_ratio | 0.730941 | 0.30024 | 0.135308 | -0.164932 | 581 | 65 |
| cand6529_init_seed20261022_4ep_visual | bad | sqi_basSQI | 0.481818 | 0.985269 | 0.97499 | -0.0102785 | 44 | 110 |
| cand6529_init_seed20261022_4ep_visual | bad | lf_ratio | 0.477273 | 0.013508 | 0.0221209 | 0.00861292 | 44 | 110 |
| cand6529_init_seed20261022_4ep_visual | bad | band_0p3_1 | 0.477273 | 0.013508 | 0.0221209 | 0.00861292 | 44 | 110 |
| cand6529_init_seed20261022_4ep_visual | bad | pc1 | 0.468182 | 9.07989 | 8.72243 | -0.357451 | 44 | 110 |
| cand6529_init_seed20261022_4ep_visual | bad | wavelet_e4 | 0.463636 | 0.18165 | 0.165123 | -0.0165265 | 44 | 110 |
| cand6529_init_seed20261022_4ep_visual | bad | row_pos | 0.454545 | 32520.5 | 32568.5 | 48 | 44 | 110 |
| cand6529_init_seed20261022_4ep_visual | bad | pca_margin | 0.454545 | 5.46666 | 5.05766 | -0.408993 | 44 | 110 |
| cand6529_init_seed20261022_4ep_visual | bad | class_margin_percentile | 0.454545 | 0.0831599 | 0.0737938 | -0.00936613 | 44 | 110 |

## Waveform Panels

- Current split error waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\current_init_seed20261023_lr18e4_4ep_visual_test_error_waveform_panels.png`
- Candidate 184 waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\cand184_init_seed20261021_4ep_visual_test_error_waveform_panels.png`
- Candidate 6529 waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\cand6529_init_seed20261022_4ep_visual_test_error_waveform_panels.png`

## Working Hypothesis for Next Model Step

- Good/medium mutual eating is not one global boundary. It is two record-specific outlier shells: QRS-visible low-detail good/medium rows and low-QRS/low-baseline-confidence medium rows.
- Bad failure is mostly not the already-learned right-island core. The hard cases are broad `outlier_low_confidence` rows with contact/baseline drift, low high-frequency/detail, low RR/QRS count reliability, and sometimes deceptively visible QRS. The model needs a bad-outlier stress token/head that preserves non-bad specificity.
- For the Transformer, the next useful architecture target is not more route logic: it is local event/contact tokens plus auxiliary losses for basSQI, flatline/contact, RR detector count/agreement, high-frequency detail, and bad specificity.

Comparison CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\but_capacity_split_seed_comparison_20260618.csv`
Feature gap CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\but_capacity_key_error_feature_gaps_20260618.csv`
