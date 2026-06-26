# GM Mechanism Repair: Subtype Fusion Decision (2026-06-25)

## Executive Summary

本轮在固定 `ptb_v112_gm_buffered_large_hybrid_s20260741` 数据、不改 encoder、不 warm-start 的条件下，新增并完整跑完 E21-E24：

- `E21_e6_subcls_consistency`: 只加 subtype-derived class consistency。
- `E22_e6_subtype_class_fusion`: 在 E6 上把 subtype-derived class probability 融进主 class probability。
- `E23_e14_subtype_class_fusion`: 在 E14 med-guard/low-gain 基础上做 subtype fusion。
- `E24_e6_subtype_fusion_pairrank`: E6 + subtype fusion + pairrank。

结论：**E24 是当前最强主线候选**。它同时拿到最高 mean clean-test accuracy 和最高 good/medium balanced recall；不是单靠牺牲某一类换来的提升。

## Main Result

| Candidate | Mean Acc | Macro-F1 | Good | Medium | Bad | GM Balanced | Good->Medium | Medium->Good | Record Macro Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E24 subtype fusion + pairrank | 0.924382 | 0.908527 | 0.902949 | 0.856647 | 0.972482 | 0.879798 | 76.0 | 116.7 | 0.930759 |
| E23 E14 + subtype fusion | 0.923968 | 0.907960 | 0.875661 | 0.866162 | 0.977958 | 0.870911 | 95.0 | 92.7 | 0.931097 |
| E21 subtype consistency | 0.922726 | 0.906753 | 0.881820 | 0.864165 | 0.974426 | 0.872992 | 92.0 | 100.3 | 0.928953 |
| E6 old mainline | 0.921981 | 0.905553 | 0.894739 | 0.853715 | 0.972295 | 0.874227 | 81.7 | 117.0 | 0.928187 |
| E14 balanced ablation | 0.921401 | 0.904896 | 0.890435 | 0.860098 | 0.970151 | 0.875266 | 89.0 | 115.3 | 0.926532 |

Best single folds:

- E24 fold 1: `acc=0.931180`, macro-F1 `0.919768`, good/medium/bad recall `0.921505 / 0.872617 / 0.970634`.
- E23 fold 1: `acc=0.929193`, macro-F1 `0.917283`, good/medium/bad recall `0.897849 / 0.876950 / 0.975270`.

## Mechanism Readout

What helped:

- **Subtype-to-class fusion is useful**, but unstable alone. E22 did not beat E6 on mean acc; E23 got higher peaks and mean acc when combined with E14's lower-gain/medium-guard setting.
- **Pairrank helps when paired with subtype fusion.** E24 has the best GM balanced recall and the best overall mean acc.
- **Subtype consistency alone is mildly useful.** E21 beats old E6 by about `+0.000745` acc and improves subtype-class alignment, but it is not enough to become the mainline by itself.

What did not solve the full problem:

- Direct subtype fusion can swing good/medium. E23 has higher acc than E6 but lower GM balanced than E14.
- Pairrank still has late-epoch instability in fold 2. The selected best checkpoint is good, but training curves show collapse risk after the peak.
- We are not near 0.95 yet. The new best is ~0.924 clean-test mean, so the remaining gap is real.

## Feature Recovery Check

E24 did not gain by breaking factor learning. Key decoded factors stayed comparable to E6/E14:

| Feature | E24 corr_all | E24 min class corr | Note |
|---|---:|---:|---|
| qrs_visibility | 0.9643 | 0.7175 | stable |
| qrs_band_ratio | 0.9520 | 0.8233 | stable |
| sqi_basSQI | 0.9666 | 0.7823 | stable |
| baseline_step | 0.9683 | 0.8710 | stable |
| non_qrs_rms_ratio | 0.9677 | 0.9128 | stable |
| detector_agreement | 0.7518 | 0.1336 | still weak class-wise |
| contact_loss_win_ratio | 0.2014 | -0.0393 | still weak |
| template_corr | 0.9496 | 0.2756 | weak class-wise |

Interpretation: encoder/factor recovery is no longer the main blocker for the easy SQI factors. The remaining hard part is **using detector/contact/template evidence consistently at the good/medium boundary and bad/nonbad artifact boundary**.

## Current Mainline Recommendation

Use `E24_e6_subtype_fusion_pairrank` as the next mainline:

- It keeps waveform-only inference.
- It preserves factor recovery.
- It adds a more interpretable path: waveform -> predicted mechanism/factor/subtype evidence -> subtype-derived class fusion -> hierarchical class decision.
- It reduces good->medium errors versus E6: `81.7 -> 76.0` average.
- It raises GM balanced recall versus E6/E14: `0.8742/0.8753 -> 0.8798`.

## Next Step Toward 0.95

Do not start another broad class-weight sweep. The highest-value next experiments are:

1. **Stabilize E24 checkpoint dynamics**
   - Lower LR or add cosine decay around the fold2 collapse point.
   - Keep E24 mechanism fixed.
   - Compare best epoch vs EMA/SWA checkpoint.

2. **Regularize subtype fusion**
   - Reduce `subtype_class_fusion_alpha` slightly, e.g. `0.12-0.18`.
   - Keep pairrank on.
   - Add entropy/gate regularization so subtype-derived probability cannot overpower the hierarchy.

3. **Target the remaining weak evidence**
   - Detector/template/contact are the only clearly weak class-wise features.
   - Add local-event/contact-specific validation panels and loss gating before adding more heads.

4. **Run seed confirmation only for E24-family**
   - Candidate set should be small now:
     - `E24_current`
     - `E24_low_alpha`
     - `E24_low_lr`
     - `E24_low_alpha_low_lr`
     - optional `E24_EMA/SWA`

## Artifacts

Runner:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_gm_mechanism_repair_suite.py`

Primary outputs:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\candidate_metrics_e21e24_subtypecal_20260625.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\factor_recovery_decoded_e21e24_subtypecal_20260625.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\confusion_by_candidate_e21e24_subtypecal_20260625.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\record_metrics_e21e24_subtypecal_20260625.csv`

Reports and waveform panels:

- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\gm_mechanism_repair_suite_report_e21e24_subtypecal_20260625.md`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\E24_e6_subtype_fusion_pairrank_fold0_waveform_panel.png`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\E24_e6_subtype_fusion_pairrank_fold1_waveform_panel.png`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\E24_e6_subtype_fusion_pairrank_fold2_waveform_panel.png`

Verification:

- `py_compile` passed for the updated runner.
- Training completed for E21-E24, 3 folds, 12 epochs, from scratch.
- stderr only contains `torch.load` FutureWarning during best-checkpoint evaluation.
