# Lead-I Raw PTB Carrier Distribution Check

## Decision

The previous raw-carrier distribution baseline used PTB-XL lead II.  The method
now supports an explicit `--lead` argument, and the current preferred carrier
for the distribution-first PTB synthetic line is PTB-XL lead I:

`raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680`

This keeps the carrier definition closer to the intended clean PTB lead-I
morphology while preserving the same subtype-conditional OT/herding selection
logic used by v78.

## Versions Compared

| version | lead | pool multiplier | status |
| --- | --- | ---: | --- |
| `v78rawcarrier_ot_fast` | II | 15 | historical baseline |
| `v78lead1_rawcarrier_ot_fast` | I | 15 | lead-I check |
| `v78lead1_pool30_rawcarrier_ot` | I | 30 | current distribution baseline |

## Aggregate Distribution Metrics

Lower is better except discriminative AUC, where closer to 0.5 is better.

| version | class | median gap | quantile loss | sliced-W | MMD | PCA gap | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v78rawcarrier_ot_fast` | all | 3.9677 | 3.9388 | 6.8049 | 1.1245 | 0.8919 | 1.0000 |
| `v78lead1_rawcarrier_ot_fast` | all | 3.9109 | 3.8896 | 6.5908 | 1.1272 | 0.8833 | 1.0000 |
| `v78lead1_pool30_rawcarrier_ot` | all | 3.9103 | 3.9472 | 6.4990 | 1.1238 | 0.8788 | 1.0000 |
| `v78rawcarrier_ot_fast` | good | 1.1095 | 1.1016 | 1.7710 | 1.0877 | 0.8907 | 1.0000 |
| `v78lead1_pool30_rawcarrier_ot` | good | 1.0006 | 0.9985 | 1.7309 | 1.1058 | 0.8607 | 1.0000 |
| `v78rawcarrier_ot_fast` | medium | 0.7254 | 0.7391 | 1.3887 | 0.9339 | 0.7772 | 1.0000 |
| `v78lead1_pool30_rawcarrier_ot` | medium | 0.6694 | 0.7084 | 1.3813 | 0.9431 | 0.7641 | 1.0000 |
| `v78rawcarrier_ot_fast` | bad | 8.3251 | 8.2509 | 14.2692 | 1.2870 | 0.9748 | 1.0000 |
| `v78lead1_pool30_rawcarrier_ot` | bad | 8.3035 | 8.3667 | 13.5602 | 1.2658 | 0.9738 | 1.0000 |

## Visual Interpretation

- Lead I improves the good/medium shell.  The PTB synthetic points no longer
  form isolated center clusters; they cover the BUT diagonal shell more
  naturally.
- Increasing the candidate pool from 15x to 30x gives small additional gains
  for good/medium PCA density and robust gaps.
- Bad remains the limiting class.  The issue is not carrier lead or candidate
  pool size; several bad subtypes still collapse into a similar thick-noise
  morphology.  The next generator change should be bad-mechanism-specific:
  baseline wander, contact/reset/flatline, low-QRS, high-frequency/detail, and
  detector/template disagreement should have visibly distinct waveform
  envelopes.

## Key Outputs

- Lead-I carrier protocol:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\raw_ptbxl_carrier_protocols\raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680`
- Current distribution baseline protocol:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_pool30_rawcarrier_ot\protocol_v78lead1_pool30_rawcarrier_ot_pc1500_s20260684`
- Shared PCA:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_pool30_rawcarrier_ot\v78lead1_pool30_rawcarrier_ot_shared_pca_but_vs_ptb.png`
- Distribution-first audit:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v78lead1_pool30_rawcarrier_ot_distribution_first`

## Remaining Feature Gaps

The largest remaining gaps are stable across lead-I pool sizes:

- bad `sqi_basSQI`: PTB remains too low relative to BUT.
- bad/good/medium `non_qrs_diff_p95`: PTB synthetic still has too much
  non-QRS local derivative/detail energy.
- good/medium `qrs_visibility`: PTB QRS remains too visible/saturated compared
  with BUT target windows.
- bad `detector_agreement` and `band_30_45`: still mismatched.

These gaps are generator-mechanism problems, not model-training problems.
