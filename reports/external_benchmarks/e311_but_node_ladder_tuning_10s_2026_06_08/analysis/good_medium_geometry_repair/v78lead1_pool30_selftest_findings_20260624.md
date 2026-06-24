# v78 Lead-I Pool30 Self-Test Findings

## Scope

This report checks whether the current distribution-first synthetic protocol is
learnable by the frozen waveform-only Event-Factorized SQI Conformer family.

No cross-dataset score is used here.  Each dataset is trained and tested only
within itself from random initialization:

- PTB synthetic v78 lead-I pool30 train/val -> PTB synthetic test.
- BUT keep-outlier/drop-mediumlike-bad train/val -> BUT test.

## Protocols

- PTB synthetic:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_pool30_rawcarrier_ot\protocol_v78lead1_pool30_rawcarrier_ot_pc1500_s20260684`
- BUT reference:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Main Results

| model | domain | bucket | acc | macro-F1 | good recall | medium recall | bad recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 query-only | BUT only | test | 0.9219 | 0.9292 | 0.9229 | 0.9072 | 0.9714 |
| E4 query+hires+local+artifact | BUT only | test | 0.9167 | 0.9248 | 0.9066 | 0.9131 | 0.9918 |
| E1 query-only | PTB synthetic only | test | 0.8306 | 0.8305 | 0.8267 | 0.7072 | 0.9558 |
| E4 query+hires+local+artifact | PTB synthetic only | test | 0.8217 | 0.8207 | 0.6756 | 0.8108 | 0.9779 |

## Interpretation

BUT self-test is substantially stronger than PTB synthetic self-test under the
same waveform-only model family.  This means the current BUT protocol is
learnable and internally coherent enough for a simple Event-Factorized
Conformer to reach about 0.92 test accuracy.

The PTB synthetic protocol is not yet equally coherent.  Bad is easy to learn,
but the good/medium boundary is unstable:

- E1 keeps good recall higher but medium recall falls to 0.707.
- E4 improves medium recall to 0.811 and bad recall to 0.978, but good recall
  collapses to 0.676.

This is not primarily a model-capacity problem.  It is evidence that the current
PTB synthetic distribution and subtype labels still create a good/medium
boundary that is harder and less natural than the BUT boundary.

## Feature Recovery Notes

The model can recover several waveform-computable factors:

- `baseline_step`, `sqi_basSQI`, `non_qrs_diff_p95`, and `flatline_ratio` are
  generally learnable.
- `qrs_visibility` is weak inside PTB bad (`corr_bad` about 0.02), despite being
  learnable in BUT bad (`corr_bad` about 0.86).
- This points back to synthetic bad morphology: the current bad generator
  creates strong bad labels, but not enough mechanism-readable variation for
  low-QRS / detector disagreement / detail-noise subtypes.

## Output Files

- E1 report:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e1_report.md`
- E4 report:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4_report.md`
- Combined summary:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_compare_e1_e4.csv`

## Next Data Step

Do not change the formal model yet.  Improve the synthetic protocol first:

1. Keep lead-I raw PTB carrier and subtype-conditional OT/herding.
2. Preserve the current good/medium PCA shell, because it is much better than
   previous replay-bank protocols.
3. Rework synthetic bad and hard good/medium boundaries so low-QRS, detector
   disagreement, baseline/contact, and high-frequency/detail have distinct
   waveform envelopes rather than one thick-noise morphology.
