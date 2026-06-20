# PTB/BUT Alignment Diagnosis - 2026-06-20

## Summary

The cross-dataset failure is now explained by a data-generation mismatch, not by lack of training time.

The current PTB synthetic pools do not contain enough BUT-like low-QRS / detector-failure morphology. Selection, reweighting, and raw-amplitude calibration cannot fix this because the diagnostic features are computed after robust per-window normalization.

## Key Evidence

### V1 nearest-neighbor alignment

Protocol:

`outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/event_xds_aligned_v1/protocol_ptb_buttrain_aligned_pc3000_s20260620`

The protocol was balanced at 9000 rows: 3000 good, 3000 medium, 3000 bad. Matching used only clean-BUT train split rows and waveform-computable features.

Best PTB-aligned -> BUT result:

- `E2_query_highres`, `cross_test`: acc 0.3926, macro-F1 0.3915, good/medium/bad recall 0.9970 / 0.0660 / 1.0000.
- `E2_query_highres`, `cross_all`: acc 0.7231, macro-F1 0.6138, good/medium/bad recall 0.9939 / 0.0575 / 1.0000.

Failure mode: almost all BUT medium windows are eaten by good/bad. Bad recall is already high, so the blocker is the medium/bad and good/medium geometry, not bad recall alone.

Best BUT -> PTB-aligned result:

- `E1_query_only`, `cross_test`: acc 0.6712, macro-F1 0.5954, good/medium/bad recall 1.0000 / 1.0000 / 0.0584.

Failure mode: BUT-trained model almost never recognizes current PTB synthetic bad as bad. This proves PTB bad morphology is not semantically aligned with BUT bad.

### PTB pool support is missing the BUT low-QRS regime

Audit:

`outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/event_xds_aligned_v1/ptb_pool_feature_quantiles.csv`

For medium:

- BUT test median `qrs_visibility` is 0.1505 and `qrs_band_ratio` is 0.4529.
- PTB pool medium minimum `qrs_visibility` is 0.5722 and minimum `qrs_band_ratio` is 1.9593.

For bad:

- BUT train median `qrs_visibility` is 0.2479 and `qrs_band_ratio` is 0.8112.
- PTB pool bad 1st percentile `qrs_visibility` is 0.6778 and 1st percentile `qrs_band_ratio` is 2.2181.

This means the PTB pool cannot cover the low-QRS detector-failure shell needed by clean BUT, even before model training.

### V2 raw morphology calibration is ineffective

Audit:

`outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/event_xds_aligned_v2_morphcal/aligned_feature_distribution_audit.csv`

Raw amplitude calibration did not meaningfully move `qrs_visibility`, `qrs_band_ratio`, or robust `rms`, because these features are computed from `robust_z(x)`. The fix must alter relative morphology, not only raw amplitude.

### V3 detector-failure postprocessing is too discrete

Report:

`reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/event_factorized_sqi_conformer/event_factorized_cross_dataset_aligned_ptb_v3_detectorfail_report.md`

V3 introduced monotonic/contact-like detector-failure samples, but it overshot and created a brittle two-endpoint dataset:

- PTB -> BUT: medium recall 0.0000.
- BUT -> PTB: bad recall improved to 0.3969 on cross_test, but good recall collapsed to 0.0796.

This validates the direction, but post-hoc protocol-level waveform edits are not enough. The generator must create a continuous detector-failure family at source.

## Root Cause

Current PTB generation contains:

- good/medium with too-visible QRS peaks,
- bad that is often high-amplitude/high-QRS or broad noise,
- too little low-QRS, low `qrs_band_ratio`, baseline/contact detector-failure morphology.

Clean BUT contains many medium and bad rows where quality is poor because the detector cannot reliably identify stable QRS/RR structure, not simply because SNR is lower.

## Required Next Fix

Add a generator-native artifact family, not a post-hoc selector:

`but_detector_failure_artifact`

Target feature bands should be generated with rejection sampling against `compute_primitives`:

- medium detector-failure target: `qrs_visibility` 0.10-0.40, `qrs_band_ratio` 0.25-0.90, `baseline_step` 0.30-0.85, `non_qrs_diff_p95` 0.06-0.18.
- bad detector-failure target: `qrs_visibility` 0.05-0.30, `qrs_band_ratio` 0.20-0.90, `band_15_30` / `band_30_45` controlled but not pure white noise.
- good boundary target: keep QRS/RR readable while adding mild baseline/flat/contact ambiguity.

The generator must store local artifact provenance/masks so Event-Factorized SQI Conformer can train local/event heads consistently.

## Current Decision

Do not claim cross-dataset success from v1/v2/v3/v4.

The useful finding is narrower and strong: the missing regime is measurable, waveform-computable, and generator-addressable. The next experiment should implement source-level detector-failure PTB synthesis and then rerun the same top model cross-dataset harness.
