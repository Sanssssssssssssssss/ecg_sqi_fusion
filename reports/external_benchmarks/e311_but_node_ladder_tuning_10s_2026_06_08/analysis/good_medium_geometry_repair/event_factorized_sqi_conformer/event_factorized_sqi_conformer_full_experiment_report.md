# Event-Factorized SQI Conformer Full Experiment Report

Generated: 2026-06-20

## Scope

This report covers the completed Event-Factorized SQI Conformer experiment matrix:

- Phase 0 correctness audit.
- Phase 1 architecture ablation: E0-E4, 5 folds x 3 seeds x 8 epochs.
- Phase 2 optimizer/branching ablation: O0-O3, 5 folds x 3 seeds x 8 epochs.
- Phase 3 pretraining ablation: P0-P3, 5 folds x 3 seeds x 8 epochs.

All formal candidates were trained from scratch. Inference uses waveform-derived channels only. SQI/factor columns are teacher/diagnostic targets, not formal inference inputs.

## Overall Decision

Best mean clean-test candidate by record-macro supported F1:

- Stage: `phase1`
- Candidate: `E1_query_only`
- Mean acc: `0.9696`
- Mean record-macro supported F1: `0.9527`
- Mean good / medium / bad recall: `0.9907 / 0.9333 / 0.4000`
- Bad-containing-record bad recall mean: `0.6667`

Interpretation: Phase 2 and Phase 3 did not beat the simpler Phase 1 query-only model. The extra high-res/local/artifact objectives did not solve the bad bottleneck. The persistent weak point remains bad generalization and bad-containing-record stability, not good/medium separation.

## All-Phase Summary

| stage | candidate | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_acc | record_macro_supported_f1 | bad_containing_record_bad_recall_mean | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 | E1_query_only | 15 | 0.9696 | 0.7726 | 0.9907 | 0.9333 | 0.4000 | 0.9680 | 0.9527 | 0.6667 | 0.0015 |
| phase1 | E0_noquery_nohi_nolocal_noart | 15 | 0.9653 | 0.7689 | 0.9885 | 0.9298 | 0.3989 | 0.9631 | 0.9487 | 0.6648 | 0.0028 |
| phase1 | E2_query_highres | 15 | 0.9573 | 0.7609 | 0.9750 | 0.9241 | 0.4000 | 0.9587 | 0.9461 | 0.6667 | 0.0017 |
| phase3 | P3_clean_noisy_physio_distill | 15 | 0.9639 | 0.7666 | 0.9902 | 0.9165 | 0.4000 | 0.9611 | 0.9457 | 0.6667 | 0.0014 |
| phase1 | E3_query_highres_local | 15 | 0.9585 | 0.7640 | 0.9774 | 0.9303 | 0.4000 | 0.9575 | 0.9455 | 0.6667 | 0.0018 |
| phase3 | P2_ecg_beat_rhythm_mask | 15 | 0.9637 | 0.7636 | 0.9845 | 0.9277 | 0.4000 | 0.9609 | 0.9450 | 0.6667 | 0.0012 |
| phase2 | O3_e4_upperblock_branch | 15 | 0.9587 | 0.7629 | 0.9871 | 0.9102 | 0.4000 | 0.9564 | 0.9418 | 0.6667 | 0.0025 |
| phase3 | P0_no_pretrain | 15 | 0.9567 | 0.7575 | 0.9803 | 0.9168 | 0.4000 | 0.9546 | 0.9403 | 0.6667 | 0.0017 |
| phase1 | E4_query_highres_local_art | 15 | 0.9492 | 0.7538 | 0.9654 | 0.9191 | 0.4000 | 0.9533 | 0.9394 | 0.6667 | 0.0020 |
| phase2 | O0_e4_ordinary | 15 | 0.9467 | 0.7544 | 0.9661 | 0.9156 | 0.4000 | 0.9514 | 0.9386 | 0.6667 | 0.0019 |
| phase3 | P1_generic_mask | 15 | 0.9553 | 0.7533 | 0.9819 | 0.9040 | 0.4000 | 0.9524 | 0.9382 | 0.6667 | 0.0006 |
| phase2 | O1_e4_pcgrad_class_artifact | 15 | 0.9465 | 0.7518 | 0.9688 | 0.9128 | 0.4000 | 0.9496 | 0.9380 | 0.6667 | 0.0026 |
| phase2 | O2_e4_cagrad_light | 15 | 0.9107 | 0.7260 | 0.9178 | 0.9230 | 0.4000 | 0.9272 | 0.9222 | 0.6667 | 0.0022 |

## Per-Phase Tables

### phase1

| stage | candidate | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_acc | record_macro_supported_f1 | bad_containing_record_bad_recall_mean | artifact_positive_nonbad_bad_fpr | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 | E1_query_only | 0.9696 | 0.7726 | 0.9907 | 0.9333 | 0.4000 | 0.9680 | 0.9527 | 0.6667 | 0.0015 | 15 |
| phase1 | E0_noquery_nohi_nolocal_noart | 0.9653 | 0.7689 | 0.9885 | 0.9298 | 0.3989 | 0.9631 | 0.9487 | 0.6648 | 0.0028 | 15 |
| phase1 | E2_query_highres | 0.9573 | 0.7609 | 0.9750 | 0.9241 | 0.4000 | 0.9587 | 0.9461 | 0.6667 | 0.0017 | 15 |
| phase1 | E3_query_highres_local | 0.9585 | 0.7640 | 0.9774 | 0.9303 | 0.4000 | 0.9575 | 0.9455 | 0.6667 | 0.0018 | 15 |
| phase1 | E4_query_highres_local_art | 0.9492 | 0.7538 | 0.9654 | 0.9191 | 0.4000 | 0.9533 | 0.9394 | 0.6667 | 0.0020 | 15 |
### phase2

| stage | candidate | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_acc | record_macro_supported_f1 | bad_containing_record_bad_recall_mean | artifact_positive_nonbad_bad_fpr | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2 | O3_e4_upperblock_branch | 0.9587 | 0.7629 | 0.9871 | 0.9102 | 0.4000 | 0.9564 | 0.9418 | 0.6667 | 0.0025 | 15 |
| phase2 | O0_e4_ordinary | 0.9467 | 0.7544 | 0.9661 | 0.9156 | 0.4000 | 0.9514 | 0.9386 | 0.6667 | 0.0019 | 15 |
| phase2 | O1_e4_pcgrad_class_artifact | 0.9465 | 0.7518 | 0.9688 | 0.9128 | 0.4000 | 0.9496 | 0.9380 | 0.6667 | 0.0026 | 15 |
| phase2 | O2_e4_cagrad_light | 0.9107 | 0.7260 | 0.9178 | 0.9230 | 0.4000 | 0.9272 | 0.9222 | 0.6667 | 0.0022 | 15 |
### phase3

| stage | candidate | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_acc | record_macro_supported_f1 | bad_containing_record_bad_recall_mean | artifact_positive_nonbad_bad_fpr | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase3 | P3_clean_noisy_physio_distill | 0.9639 | 0.7666 | 0.9902 | 0.9165 | 0.4000 | 0.9611 | 0.9457 | 0.6667 | 0.0014 | 15 |
| phase3 | P2_ecg_beat_rhythm_mask | 0.9637 | 0.7636 | 0.9845 | 0.9277 | 0.4000 | 0.9609 | 0.9450 | 0.6667 | 0.0012 | 15 |
| phase3 | P0_no_pretrain | 0.9567 | 0.7575 | 0.9803 | 0.9168 | 0.4000 | 0.9546 | 0.9403 | 0.6667 | 0.0017 | 15 |
| phase3 | P1_generic_mask | 0.9553 | 0.7533 | 0.9819 | 0.9040 | 0.4000 | 0.9524 | 0.9382 | 0.6667 | 0.0006 | 15 |

## Key Files

- Phase 1 report: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_sqi_conformer_phase1_report.md`
- Phase 2 report: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_sqi_conformer_phase2_report.md`
- Phase 3 report: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_sqi_conformer_phase3_report.md`
- All-phase summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_sqi_conformer_all_phase_summary.csv`
- Metrics CSVs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_metrics.csv`, `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase2_metrics.csv`, `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_metrics.csv`
- Feature recovery CSVs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_feature_recovery.csv`, `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase2_feature_recovery.csv`, `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_feature_recovery.csv`
- Record metrics CSVs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_record_metrics.csv`, `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase2_record_metrics.csv`, `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_record_metrics.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_sqi_conformer`

## Next Experimental Implication

The current evidence argues against simply adding more auxiliary heads or generic pretraining. The model can keep good/medium high, but the bad class remains sparse and record-dependent. The next useful experiment should focus on bad-containing records and event/artifact target quality, rather than another global architecture sweep.
