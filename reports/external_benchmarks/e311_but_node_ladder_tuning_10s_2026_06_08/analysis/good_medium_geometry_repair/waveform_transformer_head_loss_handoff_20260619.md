# Waveform Transformer Head/Loss Handoff - 2026-06-19

This note is the entry point for another reviewer/GPT to inspect the current ECG quality experiment.  The immediate question is no longer whether tabular SQI/geometry features can separate the classes; they can.  The open model-design question is how to make a waveform-only Transformer recover the useful waveform facts and use them through better heads/losses without relying on route/rule artifacts or feature-only classifiers.

## Locked Scope

- Final inference input: ECG waveform-derived channels only.
- Formal model family: Transformer/Conformer-style waveform model.  No route artifact, no rule-engine frontier, no MLP/tree/47-feature classifier as the official result.
- Training signals allowed: PTB synthetic labels, clean-BUT labels when explicitly doing clean-BUT CV/fine-tune, and waveform-computable auxiliary targets.
- 47-column SQI/geometry table: allowed as diagnostics/teacher targets only.  It is not a formal inference input.
- Original/full BUT: legacy stress diagnostic only.  The active clean protocol excludes low-confidence/outlier windows by rule.

## Current Data Policy

The active BUT protocol is:

- fixed 10s windows;
- `margin_ge_5s_drop_outlier`;
- low-confidence/outlier windows removed from the main target;
- no variable-length input;
- window-level CV folds stratified by record, class, and `original_region`.

The generated CV folds are:

- `margin_ge_5s_drop_outlier_cv_seed20260619_fold0`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold1`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold2`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold3`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold4`

Important caveat: these are clean-window CV folds, not record-heldout external tests.  They are intended to answer whether the cleaned task is learnable after removing windows judged low-confidence/outlier by the current interpretable policy.

Read:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_cv_goal_status_20260619.md`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_but_protocols/`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_but_policy_distribution/`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_but_protocol_visuals/`

## Best Current Results

### Clean-BUT waveform Transformer CV

Model: `dualview_conformer_hier`

5-fold clean test:

- accuracy mean: `0.976893`
- accuracy min: `0.964747`
- macro-F1 mean: `0.979419`
- good recall mean/min: `0.974335 / 0.934898`
- medium recall mean/min: `0.966520 / 0.933065`
- bad recall mean/min: `0.999756 / 0.998778`

Interpretation: the cleaned BUT protocol is learnable by a waveform Transformer.

### PTB synthetic-only transfer

Model: `ptb_mediumshell_b0p5_best`

5-fold clean test:

- accuracy mean: `0.915601`
- accuracy min: `0.910599`
- macro-F1 mean: `0.926475`
- good recall mean/min: `0.882082 / 0.878801`
- medium recall mean/min: `0.920850 / 0.909810`
- bad recall mean/min: `0.999756 / 0.998778`

Interpretation: PTB synthetic alone learns useful bad/medium structure, but still under-recognizes dirty-good clean-BUT windows.

### PTB-init plus clean-BUT fine-tune

Model: `dualview_convtx_hier_ptbinit_b0p5`

5-fold clean test:

- accuracy mean: `0.983773`
- accuracy min: `0.980306`
- macro-F1 mean: `0.985392`
- good recall mean/min: `0.985484 / 0.980769`
- medium recall mean/min: `0.970613 / 0.949004`
- bad recall mean/min: `0.999266 / 0.997552`

Interpretation: PTB synthetic pretraining plus clean-rule BUT fine-tuning is currently the strongest clean, waveform-only Transformer chain.

## Bad-Outlier Work

The old PTB bad generation was visually too close to pure noise/contact.  We regenerated a controlled bad shell so `bad` includes visible-QRS baseline/contact/reset morphology instead of only extreme noise.

Main controlled bad bank:

- `bad_core_guard`: 770 rows
- `bad_controlled_qrs_visible`: 660 rows
- `bad_controlled_contact`: 550 rows
- `bad_mild_noise_guard`: 220 rows

Representative artifact:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/ptb_bad_waveform_feature_match/ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_waveforms.png`

Read:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/controlled_bad_outlier_regeneration_report_20260619.md`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/ptb_bad_alignment_decision_report_20260619.md`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/build_ptb_bad_waveform_feature_match.py`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_ptb_bad_alignment_cross_dataset.py`

Key finding: adding controlled bad improves bad recall, but if treated as ordinary `bad` too aggressively, it damages the good/medium boundary.  The model needs to learn an artifact/stress factor while preserving non-bad hard negatives.

## Current Generator Logic

### Good/medium banks

The good/medium boundary is built from explicit blocks instead of SNR-only rules.  The current relevant block ideas are:

- clean overlap body;
- good rescue with QRS-visible/flat/baseline edge cases;
- medium low-QRS hard negatives;
- visible-QRS medium detail degradation;
- high-SQI medium-like boundary rows;
- dirty-good baseline/outlier protection rows.

Important builder:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/build_ptb_gm_boundary_replay_bank.py`

Recent changes in that builder:

- source pools are more diverse;
- `--neighbor-k` controls source diversity;
- `medium_high_sqi_qrsvisible` and `good_baseline_outlier_protect` roles were added.

### Bank combination and class balancing

Important combiner:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/combine_ptb_boundary_banks.py`

Recent changes:

- `--balance-classes`
- `--target-per-class`
- `--balance-blocks`
- manifest audit columns: `replay_source_row`, `replay_repeat_id`, `replay_balanced`

Example banks:

- `ptb_combo_balanced2500_block_corebad_ctl10_gm_keepout_diverse_v2_signals.npz`
- `ptb_combo_balanced2500_block_corebad_ctl10_gm_keepout_blocks_v3_signals.npz`

Large `.npz` artifacts are intentionally kept in the external output tree, not committed as normal repo assets.

## Core Model Implementations To Read

Start with these files:

1. `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_clean_but_dualview_hier_transformer.py`
2. `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_ptb_bad_alignment_cross_dataset.py`
3. `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_waveform_geometry_student.py`
4. `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_waveform_sqi_module_probe.py`
5. `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_feature_token_transformer_probe.py`
6. `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/analyze_waveform_feature_learnability.py`
7. `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/analyze_waveform_hard_feature_learnability.py`

The `outputs/` tree is ignored by default, so the relevant `.py` files are force-added in the current publication commit.  The generated data/checkpoints remain local output artifacts.

## What The Current Transformer Learns

Based on current diagnostics:

Stronger recovery:

- `flatline_ratio`
- `non_qrs_diff_p95`
- `qrs_band_ratio`
- `amplitude_entropy`

Moderate recovery:

- `qrs_visibility`
- `baseline_step`
- `sqi_basSQI`

Still weak:

- `detector_agreement`

High-risk diagnostic-only targets:

- `pc1`, `pc2`, `pca_margin`, atlas/region confidence, KNN purity.

Those geometry targets can explain why the data separates, but they should not become the formal model claim unless they are translated into waveform-computable physiology/SQI targets.

## Why Head/Loss Needs Redesign

The current evidence suggests the remaining issue is not simply model size.  The likely bottleneck is task decomposition:

- A single global class head can learn shortcuts.
- Bad stress morphology needs an auxiliary artifact factor, but artifact must not imply `bad` automatically.
- Good/medium needs its own boundary head because the error is mostly a subtle quality boundary, not a generic 3-class problem.
- Detector agreement/QRS reliability is not just high-frequency energy; it requires beat-level consistency and peak agreement.
- Baseline/contact/flatline features are often local or worst-case, so pure mean pooling can dilute them.

The next model should make the heads mirror the interpretable waveform tasks.

## Recommended Next Head/Loss Design

Keep inference waveform-only, but organize the Transformer around task heads:

- `[QRS]` query/token head:
  - predicts QRS visibility, QRS band ratio, detector agreement, RR reliability;
  - should be supervised by waveform-computable targets only.
- `[BASELINE]` head:
  - predicts baseline step/ramp/span and `sqi_basSQI`.
- `[FLATLINE_CONTACT]` head:
  - predicts flat/contact/low-amplitude run length and density.
- `[BAND_DETAIL]` head:
  - predicts band powers, non-QRS detail, derivative/detail statistics.
- `[GM_BOUNDARY]` head:
  - explicit good-vs-medium binary loss over good/medium rows.
- `[BAD_STRESS]` head:
  - predicts controlled artifact/stress factor;
  - includes explicit non-bad hard-negative penalty so artifact does not collapse into `bad`.
- 3-class ordinal/class head:
  - uses the shared Transformer representation and task-query embeddings;
  - retains an ordinal good < medium < bad loss.

Candidate total loss:

```text
L = L_3class_ce_or_focal
  + lambda_gm * L_good_medium_boundary
  + lambda_bad * L_bad_specificity
  + lambda_ord * L_ordinal_quality
  + lambda_qrs * L_qrs_reliability_targets
  + lambda_base * L_baseline_targets
  + lambda_flat * L_flat_contact_targets
  + lambda_band * L_band_detail_targets
  + lambda_artifact * L_artifact_factor
  + lambda_rank * L_boundary_ranking_or_contrastive
```

Important guardrail: the artifact factor can be an auxiliary target, but the final class decision must not be a route/rule override.

## Visual/Report Locations

Use these report folders to inspect failures visually:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/representative_waveform_examples/`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/ptb_bad_waveform_feature_match/`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/ptb_bad_alignment_cross_dataset/`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/`
- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_but_protocol_visuals/`

## Practical Review Checklist

For a new head/loss proposal, answer these before training:

1. Which waveform-computable SQI/physiology target does each head learn?
2. Does any target leak atlas/KNN/test geometry as a formal inference signal?
3. How does the loss prevent artifact stress from becoming a direct bad shortcut?
4. How are good/medium boundary errors measured separately from global accuracy?
5. Are the failure panels improving visually, especially dirty-good vs visible-QRS medium and controlled bad outliers?
6. Does the model still work under clean-BUT CV, PTB synthetic-only transfer, and PTB-init clean-BUT fine-tune?
