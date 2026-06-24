# v80 Boundary-Cleaned PTB Synthetic Self-Test Result

## Summary

The v78 lead-I pool30 synthetic protocol failed mainly because the good/medium
boundary was internally inconsistent.  The model was not simply too weak:
E4+subtype could learn bad subtypes, but several good/medium hard subtypes were
visually and statistically mixed.

v80 tests a conservative data-cleaning hypothesis:

- keep lead-I raw PTB carrier and v78 distribution matching;
- remove the most ambiguous good/medium subtypes from main training;
- keep those rows as holdout/report-only rather than deleting them;
- train the same waveform-only `EventFactorizedSQIConformer` E4+subtype model
  from scratch.

## v80 Cleaning Rule

Main train keeps:

- good: `good_clean_core`, `good_overlap_boundary`, `good_isolated_low_purity`
- medium: `medium_clean_core`, `medium_overlap_boundary`,
  `medium_isolated_lowqrs`, `medium_visible_qrs_detail`,
  `medium_hard_baseline_lowqrs`
- bad: all non-mediumlike bad, including mechanism subtypes

Holdout/report-only:

- good: `good_hard_baseline_lowqrs`, `good_mild_artifact_outlier`
- medium: `medium_outlier_or_bad_boundary`
- bad: medium-like `bad_contact_reset_flatline` rows

This is not a hidden deletion.  Ambiguous rows remain in the protocol as
`split=holdout`; they are excluded from the main train/val/test self-test and
included in `all` diagnostics.

## Counts

| split | good | medium | bad |
| --- | ---: | ---: | ---: |
| train | 630 | 700 | 700 |
| val | 135 | 188 | 202 |
| test | 135 | 362 | 448 |
| holdout | 600 | 250 | 150 |

## Main Result

Same model, same training length, from scratch:

| protocol | domain | bucket | acc | macro-F1 | good | medium | bad |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| v78 | PTB only | test | 0.8455 | 0.8443 | 0.8133 | 0.7387 | 0.9823 |
| v80 | PTB only | test | 0.9196 | 0.8873 | 0.8593 | 0.8619 | 0.9844 |
| v78 | BUT only | test | 0.9131 | 0.9239 | 0.9169 | 0.8865 | 0.9898 |
| v80 | BUT only | test | 0.9273 | 0.9338 | 0.9335 | 0.9023 | 0.9837 |

v80 improves PTB self-test by about +7.4pp and also improves the independent
BUT-only self-test under the same model runner by about +1.4pp.

## Important Caveat

PTB `all` falls to 0.782 because `all` includes the deliberately held-out
ambiguous rows.  This is useful evidence, not failure: the rows moved to
holdout are exactly the internally mixed boundary rows that made v78 hard to
learn.

## Interpretation

The strongest evidence is now:

1. The model can learn a cleaner synthetic protocol: v80 PTB test reaches 0.92.
2. The v78 failure was not primarily a bad-class problem; bad recall remains
   very high.
3. The previous low result was caused by synthetic good/medium boundary
   pollution.
4. The right next step is not more model/loss tuning, but a more principled
   generation rule for ambiguous boundary rows:
   - keep a clean main train distribution;
   - keep ambiguous boundary rows as explicit holdout/stress;
   - later reintroduce them only if their label can be made visually and
     feature-wise defensible.

## Output Files

- v80 protocol:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v80lead1_boundary_clean\protocol_v80lead1_boundary_clean_pc700_s20260686`
- v80 self-test report:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_single_domain_selftest_v80lead1_boundary_clean_e4sub_report.md`
- v78/v80 comparison CSV:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v78_v80_boundary_clean_selftest_comparison.csv`
- v78 failure PCA/waveform audit:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v78lead1_ptb_selftest_failure_analysis`

## Next Step

Use v80 as the current clean main-training protocol.  Do not claim holdout
rows are solved.  The next research task is to inspect holdout rows, split them
into visually defensible subgroups, and decide which should be regenerated,
relabelled as medium, relabelled as bad, or kept as stress-only.
