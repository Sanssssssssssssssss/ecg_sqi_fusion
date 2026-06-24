# V95 Mainline Protocol Decision

Created: 2026-06-24

## Decision

Use `protocol_v95_gm_boundary_clean_drop4_from_v94_s20260710` as the current **primary high-confidence PTB synthetic training protocol**.

This protocol should be described as a clean, interpretable, learnable boundary protocol. It should not be described as a complete natural-prior reproduction of every BUT ambiguous/outlier region.

## Protocol Rule

V95 is derived from v94 by removing four good/medium subtype buckets that were repeatedly shown to be visually and feature-wise ambiguous:

- `good_isolated_low_purity`
- `good_mild_artifact_outlier`
- `medium_hard_baseline_lowqrs`
- `medium_isolated_lowqrs`

The retained training blocks are:

- Good:
  - `good_clean_core`
  - `good_overlap_boundary`
  - `good_hard_baseline_lowqrs`
- Medium:
  - `medium_clean_core`
  - `medium_overlap_boundary`
  - `medium_visible_qrs_detail`
  - `medium_outlier_or_bad_boundary`
- Bad:
  - `bad_dense_right_island`
  - `bad_detector_template_disagree`
  - `bad_baseline_wander_lowfreq`
  - `bad_contact_reset_flatline`
  - `bad_low_qrs_visibility`
  - `bad_highfreq_detail_noise`
  - `bad_other_boundary`

Rows kept: `3400`.

Rows dropped: `1100`.

Class counts:

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| good | 630 | 135 | 135 | 900 |
| medium | 700 | 152 | 148 | 1000 |
| bad | 1050 | 224 | 226 | 1500 |

## Why This Is Explainable

The rule is not an arbitrary accuracy filter. It removes four buckets where the good/medium boundary is not visually stable enough for a clean three-class training target. The remaining buckets keep:

- visible QRS good and good/medium overlap;
- medium rows with visible QRS but degraded morphology/detail;
- all major bad mechanisms, including baseline wander, contact/reset/flatline, low-QRS, high-frequency/detail, detector/template disagreement, and dense noise-like bad.

This makes v95 a cleaner supervised-learning target: the model is asked to learn waveform quality mechanisms rather than unresolved label ambiguity.

## Current Metrics

Best PTB v95 from-scratch normal waveform model:

- Candidate: `E1_query_only`
- PTB test acc: `0.962672`
- PTB test macro-F1: `0.955186`
- Recalls: good `0.940741`, medium `0.925676`, bad `1.000000`

Current BUT fixed self-test milestone:

- Candidate: `E0_noquery_nohi_nolocal_noart`
- BUT test acc: `0.932772`
- BUT test macro-F1: `0.938225`
- Recalls: good `0.931848`, medium `0.919110`, bad `0.989796`

## Visual Evidence

- Shared PCA:
  `reports/.../v95_gm_boundary_clean_distribution_transport/v95_shared_pca.png`
- Good examples:
  `reports/.../v95_gm_boundary_clean_distribution_transport/v95_good_individual_waveform_examples.png`
- Medium examples:
  `reports/.../v95_gm_boundary_clean_distribution_transport/v95_medium_individual_waveform_examples.png`
- Bad examples:
  `reports/.../v95_gm_boundary_clean_distribution_transport/v95_bad_individual_waveform_examples.png`

## Caveat

V95 is the best current training mainline, but not the final natural-prior distribution matching protocol.

Distribution audit still shows domain-separable gaps, especially in bad:

- bad RBF-MMD `0.8529`;
- bad sliced-Wasserstein `8.0511`;
- bad domain AUC `0.9978`;
- largest gaps include `detector_agreement`, `sqi_iSQI`, `raw_diff_abs_p95`, `non_qrs_diff_p95`, `band_15_30`, and `baseline_step`.

Use v98/natural-prior protocols for distribution visualization and stress evidence. Use v95 for the current formal clean training line.

