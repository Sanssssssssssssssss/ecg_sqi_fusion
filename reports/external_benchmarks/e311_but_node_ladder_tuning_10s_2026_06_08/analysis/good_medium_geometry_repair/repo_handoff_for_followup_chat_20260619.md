# Follow-up Chat Handoff: BUT / PTB Synthetic SQI-Geometry Research

This handoff is for another Codex/chat session after pulling the pushed branch. It summarizes where to inspect code, reports, visualizations, and data artifacts for the current ECG SQI/BUT research state.

## Branch And Scope

- Branch: `experiment/factorial-local-splits`
- Repository: `https://github.com/Sanssssssssssssssss/ecg_sqi_fusion.git`
- Scope pushed here includes:
  - SQI/QRS code changes under `src/sqi_pipeline`.
  - Transformer/UFormer model and training changes under `src/transformer_pipeline`.
  - External benchmark runner scripts under `src/transformer_pipeline/external_benchmarks`.
  - Experiment-only analysis runners under `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair`.
  - Reports, CSV summaries, diagnostics, and visualization PNGs under `reports/external_benchmarks/...`.

## Current Big Picture

The current problem is not simply "the model cannot learn SQI." Direct BUT diagnostic training shows the waveform Transformer can recover many waveform-computable SQI/geometry targets, but it still fails to convert them into a stable three-way good/medium/bad decision under a hard record/difficulty shift.

The current BUT test split is dominated by record `111001`, especially `outlier_low_confidence` and good/medium overlap windows. This split is much harder than train/val by record and region composition. Bad core is learned; bad outlier stress from record `111001` is the difficult part.

## Most Important Reports To Read First

1. Direct BUT training + feature recovery update:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_direct_transformer_recovery/but_direct_training_feature_gap_update_20260618.md`

   Key result:
   - `aug_convtx_balanced_focal_trainval` raw: acc `0.814793`, good/medium/bad recall `0.963462 / 0.734975 / 0.357664`.
   - Bad-calibrated threshold `0.05`: acc `0.808659`, good/medium/bad recall `0.963187 / 0.688884 / 0.729927`.
   - Interpretation: bad signal exists, but improving bad steals too many medium samples.

2. BUT split/capacity/error-gap report:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/but_split_seed_capacity_error_update_20260618.md`

   Key result:
   - Current split direct BUT capacity is around `0.80`.
   - Strict seed-roll variants can move to roughly `0.84-0.85`.
   - Window-random diagnostic upper bound is around `0.917`, confirming intra-record capacity is much higher and strict record generalization is the bottleneck.

3. Waveform Transformer direct BUT reports:

   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_transformer_original_adaptation_report.md`
   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_transformer_augmented_original_report.md`

4. Hard feature / architecture exploration notes:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/hardfeature_framework_exploration_update_20260618.md`

5. Boundary-block synthetic generation / node ladder reports:

   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_report.md`
   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_iteration_summary_20260611.md`
   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_boundary_research_decision_20260613.md`

## Key Visualizations

1. Direct BUT raw error waveforms:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/aug_convtx_balanced_focal_trainval_raw_visual_test_error_waveform_panels.png`

2. Direct BUT bad-calibrated error waveforms:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/aug_convtx_balanced_focal_trainval_badcal_t005_visual_test_error_waveform_panels.png`

3. Current split visual panels from prior capacity analysis:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/current_init_seed20261023_lr18e4_4ep_visual_test_error_waveform_panels.png`

4. Presentation figures:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/presentation_figures/`

5. Original bucketed confusion/evaluation figures:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/original_bucketed_checkpoint/`

## Key CSV/Data Artifacts

1. Direct BUT predictions and feature recovery:

   `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_direct_transformer_recovery/`

   Important files:
   - `aug_convtx_balanced_focal_trainval_raw_predictions.csv`
   - `aug_convtx_balanced_focal_trainval_raw_feature_recovery.csv`
   - `aug_convtx_balanced_focal_trainval_badcal_t005_predictions.csv`
   - `aug_convtx_balanced_focal_trainval_badcal_t005_feature_recovery.csv`

2. Capacity/error-gap CSVs:

   `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/`

3. Original bucketed report-only predictions:

   `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/original_bucketed_checkpoint/`

4. Node ladder registry/diagnostics:

   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_ladder_diagnostic_metrics.csv`
   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_promotion_decisions.csv`
   - `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.csv`
   - `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_ladder_diagnostic_metrics.csv`
   - `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_promotion_decisions.csv`
   - `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.csv`

## Code Locations

SQI / QRS-related code:

- `src/sqi_pipeline/qrs/paper_detectors.py`
- `src/sqi_pipeline/diagnostics/paper_extra_experiments.py`
- `src/sqi_pipeline/diagnostics/plot_paper_figures.py`

Transformer/UFormer model and training code:

- `src/transformer_pipeline/models/uformer1d.py`
- `src/transformer_pipeline/train_uformer_mainline.py`
- `src/transformer_pipeline/e311_uformer_data.py`

External benchmark code:

- `src/transformer_pipeline/external_benchmarks/run.py`
- `src/transformer_pipeline/external_benchmarks/but_node_ladder_tuning_10s.py`
- `src/transformer_pipeline/external_benchmarks/but_original_aware_semiclean_boundary_10s.py`
- `src/transformer_pipeline/external_benchmarks/but_sqi_fusion_ptb_train.py`
- `src/transformer_pipeline/external_benchmarks/but_bad_boundary_tuning.py`
- `src/transformer_pipeline/external_benchmarks/analyze_ptb_but_sqi_gap.py`

Experiment-only waveform/geometry runners:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_waveform_geometry_student.py`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_waveform_transformer_original_adaptation.py`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_waveform_transformer_augmented_original.py`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_but_split_interpretable_plan.py`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/analyze_but_direct_transformer_recovery.py`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/analyze_but_capacity_error_gap.py`

## Current Technical Diagnosis

Features that the waveform Transformer recovers reasonably well on BUT test:

- `sqi_basSQI`
- `qrs_band_ratio`
- `baseline_step`
- `flatline_ratio`
- `diff_abs_p95`
- `non_qrs_diff_p95`
- high-frequency/detail/wavelet energies

Features/targets still weak or unstable:

- `detector_agreement`
- `contact_loss_win_ratio`
- contact/baseline artifact shape under record `111001`
- converting bad-stress evidence into bad recall without over-predicting bad on medium.

The most important next research question is not "can a backend classifier separate the rows?" Feature/MLP/tree upper bounds already show separability. The key question is how to make the waveform Transformer learn and use reliable QRS/RR/baseline/contact/detail evidence in a record-robust way.

## Suggested Next Step For The Follow-up Chat

Start from the two direct BUT reports and the badcal waveform panel. Then inspect the feature recovery CSVs to compare correctly vs incorrectly classified `111001/outlier_low_confidence` medium and bad rows.

The next architecture experiment should keep waveform-only inference, but split the Transformer supervision into:

1. QRS/RR reliability query head,
2. baseline/contact artifact query head,
3. detail/frequency query head,
4. bad-stress head with explicit non-bad specificity loss,
5. good/medium boundary head with protection against medium->bad and medium->good collapse.

Avoid treating rule/router artifacts as the final model. They are useful diagnostics, but the main model claim should remain waveform-only Transformer inference, with SQI/geometry used as training-time targets and analysis tools.
