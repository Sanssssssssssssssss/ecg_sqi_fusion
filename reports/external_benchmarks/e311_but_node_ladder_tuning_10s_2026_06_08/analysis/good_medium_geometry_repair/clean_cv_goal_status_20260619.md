# Clean BUT CV Goal Status 2026-06-19

## Current Scope

The active target is the cleaned fixed-10s BUT protocol, not legacy full/original-all BUT.

Main policy:

- `margin_ge_5s_drop_outlier`
- fixed 10s windows only
- low-confidence/outlier windows removed
- no variable-length model
- inference input remains waveform-derived channels only

The new CV folds are window-level folds stratified by record, class, and `original_region`. They are not record-heldout external tests. Their purpose is to test stable learnability after the clean-window rule is applied.

## Clean CV Fold Quality

Generated folds:

- `margin_ge_5s_drop_outlier_cv_seed20260619_fold0`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold1`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold2`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold3`
- `margin_ge_5s_drop_outlier_cv_seed20260619_fold4`

Each fold has about 4.3k test rows and balanced bad coverage:

- test good: about 2.24k-2.26k
- test medium: about 1.24k-1.26k
- test bad: about 815-818

Report:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_but_protocols/margin_ge_5s_drop_outlier_cv_seed20260619_report.md`

## Result A: Direct Clean-BUT Waveform Transformer CV

Model: `dualview_conformer_hier`

Result on 5-fold clean test:

- acc mean: 0.976893
- acc min: 0.964747
- macro-F1 mean: 0.979419
- good recall mean/min: 0.974335 / 0.934898
- medium recall mean/min: 0.966520 / 0.933065
- bad recall mean/min: 0.999756 / 0.998778

Interpretation: the cleaned BUT protocol is learnable by waveform Transformer well above the 0.94 target.

## Result B: PTB Synthetic-Only Transfer Check

Model: `ptb_mediumshell_b0p5_best`

This is the best PTB-generated medium-shell checkpoint, evaluated without clean-BUT fine-tuning.

5-fold clean test:

- acc mean: 0.915601
- acc min: 0.910599
- macro-F1 mean: 0.926475
- good recall mean/min: 0.882082 / 0.878801
- medium recall mean/min: 0.920850 / 0.909810
- bad recall mean/min: 0.999756 / 0.998778

Interpretation: pure PTB synthetic transfer is not yet a stable 0.94 solution. It solves bad and much of medium, but under-recognizes dirty-good clean-BUT windows.

Metrics:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_cv_checkpoint_eval/ptb_mediumshell_b0p5_best_on_clean_cv_metrics.csv`

## Result C: PTB-Init + Clean-Rule Fine-Tune CV

Model: `dualview_convtx_hier_ptbinit_b0p5`

Initialization:

- PTB medium-shell checkpoint:
  `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/runs/ptb_bad_alignment_cross_dataset/ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5/ckpt_best.pt`

Then fine-tuned on each clean-CV train split.

5-fold clean test:

- acc mean: 0.983773
- acc min: 0.980306
- macro-F1 mean: 0.985392
- good recall mean/min: 0.985484 / 0.980769
- medium recall mean/min: 0.970613 / 0.949004
- bad recall mean/min: 0.999266 / 0.997552

Interpretation: this satisfies the current 0.94+ goal with a clean, waveform-only Transformer chain:

1. PTB synthetic/medium-shell pretraining learns useful waveform structure and distribution priors.
2. Clean-BUT rule fine-tuning calibrates the good/medium boundary.
3. Low-confidence/outlier windows remain excluded from the main target by design.

Metrics:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_but_dualview_hier_transformer/clean_but_dualview_hier_transformer_metrics.csv`

Checkpoints:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/runs/clean_but_dualview_hier_transformer/margin_ge_5s_drop_outlier_cv_seed20260619_fold0/dualview_convtx_hier_ptbinit_b0p5/ckpt_best.pt`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/runs/clean_but_dualview_hier_transformer/margin_ge_5s_drop_outlier_cv_seed20260619_fold1/dualview_convtx_hier_ptbinit_b0p5/ckpt_best.pt`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/runs/clean_but_dualview_hier_transformer/margin_ge_5s_drop_outlier_cv_seed20260619_fold2/dualview_convtx_hier_ptbinit_b0p5/ckpt_best.pt`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/runs/clean_but_dualview_hier_transformer/margin_ge_5s_drop_outlier_cv_seed20260619_fold3/dualview_convtx_hier_ptbinit_b0p5/ckpt_best.pt`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/runs/clean_but_dualview_hier_transformer/margin_ge_5s_drop_outlier_cv_seed20260619_fold4/dualview_convtx_hier_ptbinit_b0p5/ckpt_best.pt`

## Remaining Research Caveat

The model still recovers some interpretable targets better than others:

- strong: `flatline_ratio`, `non_qrs_diff_p95`, `qrs_band_ratio`, `amplitude_entropy`
- decent: `qrs_visibility`, `baseline_step`, `sqi_basSQI`
- still weak: `detector_agreement`

This no longer blocks clean-CV classification, but it is the next feature-learning target if we want a more physiologically satisfying Transformer.
