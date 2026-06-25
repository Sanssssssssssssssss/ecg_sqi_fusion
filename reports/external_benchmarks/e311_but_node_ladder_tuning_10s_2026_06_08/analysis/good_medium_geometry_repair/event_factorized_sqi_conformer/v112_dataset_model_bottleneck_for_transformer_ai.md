# v112 Dataset / Model / Bottleneck Brief for Transformer Redesign

Generated: 2026-06-25

This note freezes the current data and model state so another AI/researcher can reason about how to adjust the waveform Transformer. The core question is not whether feature-only MLP/tree models can separate the data. The formal target is still waveform-only inference: the model must infer ECG quality from waveform-derived channels, while SQI/geometry features are used only as synthetic-generation targets, training teachers, and diagnostics.

## 1. Current Fixed Dataset

### Protocol

- Main protocol: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Protocol alias path used by runners:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\ptb_v112_gm_buffered_large_hybrid_s20260741`
- Source protocol path:
  `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v112_gm_buffered_large_hybrid_s20260741`
- Main data report:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v112_gm_buffered_large_hybrid\v112_gm_buffered_large_hybrid_report.md`

### What v112 Is

v112 is derived from the v111 large hybrid synthetic PTB dataset. The goal is to keep the distribution-matched PTB synthetic data fixed for the next model-design discussion.

The logic is:

1. Start from v111 large hybrid synthetic PTB.
2. Keep all bad rows unchanged.
3. For good/medium rows only, remove rows when a BUT train+val good-vs-medium discriminator strongly predicts the opposite class.
4. The discriminator is only a generation/audit filter. It is not a model input, not a final classifier, and not used at inference.
5. BUT test rows are not used for dataset selection.

Dataset size:

| item | count |
| --- | ---: |
| rows before v112 filter | 13,968 |
| rows after v112 filter | 12,074 |
| removed good/medium contradictions | 1,894 |
| good rows | 2,615 |
| medium rows | 3,459 |
| bad rows | 6,000 |

Good/medium contradiction threshold: `0.85`.

BUT train+val good-vs-medium audit AUC: `0.966959`.

### Distribution Status

v112 is acceptable as the frozen working dataset for model research, but it is not a perfect domain match.

Class-level distribution audit:

| class | RBF-MMD | sliced-Wasserstein | quantile loss | domain AUC | PCA density overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| good | 0.3868 | 0.9999 | 0.9760 | 1.0000 | 0.2686 |
| medium | 0.2449 | 0.6327 | 0.5405 | 1.0000 | 0.3464 |
| bad | 1.3561 | 8.7570 | 6.2280 | 1.0000 | 0.0000 |

Interpretation:

- Good/medium are much closer than earlier versions but still distinguishable by a domain classifier.
- Bad remains the largest distribution mismatch, but the current model already reaches high bad recall on the synthetic diagnostic.
- The present bottleneck for synthetic/node accuracy is good/medium mutual confusion, not bad recall.

Key figures:

- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v112_gm_buffered_large_hybrid\v112_gm_buffered_shared_pca.png`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v112_gm_buffered_large_hybrid\v112_gm_buffered_key_feature_cdf_overlay.png`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v112_gm_buffered_large_hybrid\v112_gm_buffered_good_subtype_waveforms.png`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v112_gm_buffered_large_hybrid\v112_gm_buffered_medium_subtype_waveforms.png`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v112_gm_buffered_large_hybrid\v112_gm_buffered_bad_subtype_waveforms.png`

## 2. Label and Subtype System

### Main Quality Classes

- `good`
- `medium`
- `bad`

### Leaf Quality Subtypes

Good:

- `good_clean_core`
- `good_overlap_boundary`
- `good_isolated_low_purity`
- `good_mild_artifact_outlier`
- `good_hard_baseline_lowqrs`

Medium:

- `medium_clean_core`
- `medium_overlap_boundary`
- `medium_isolated_lowqrs`
- `medium_visible_qrs_detail`
- `medium_outlier_or_bad_boundary`
- `medium_hard_baseline_lowqrs`

Bad:

- `bad_dense_right_island`
- `bad_detector_template_disagree`
- `bad_baseline_wander_lowfreq`
- `bad_contact_reset_flatline`
- `bad_low_qrs_visibility`
- `bad_highfreq_detail_noise`
- `bad_other_boundary`

### Hard Good/Medium Boundary Families

The current model treats two good/medium boundary families explicitly:

| family | good leaf | medium leaf |
| --- | --- | --- |
| `isolated_lowqrs` | `good_isolated_low_purity` | `medium_isolated_lowqrs` |
| `mildartifact_hardbaseline` | `good_mild_artifact_outlier` | `medium_hard_baseline_lowqrs` |

Boundary evidence targets:

- Medium direction: `baseline_step`, `band_0p3_1`, `non_qrs_rms_ratio`
- Good direction: `sqi_basSQI`, `qrs_band_ratio`, `template_corr`

## 3. Current Formal Model

### Implementation

Runner:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_event_factorized_sqi_conformer.py`

Model class:

`EventFactorizedSQIConformer`

Formal inference input:

- waveform-derived channels only
- no SQI feature vector input
- no PCA/KNN/atlas feature input
- no MLP/tree/rule route at inference

SQI/geometry features are allowed only as:

- synthetic generation targets
- training-time teacher targets
- diagnostics/feature recovery reports

### Architecture

High-level structure:

1. High-resolution waveform stem:
   - Conv1d stride 2
   - GroupNorm
   - GELU
   - second Conv1d block
   - preserves local QRS/contact/flatline/detail evidence before heavy downsampling

2. Context downsampling:
   - Conv1d stride 4 on the high-resolution representation
   - produces context tokens for rhythm/global morphology reasoning

3. Query-token Conformer:
   - positional encoding
   - Conformer blocks with attention + depthwise temporal convolution
   - explicit quality-mechanism query tokens:
     - `[QRS]`
     - `[RR_TEMPLATE]`
     - `[BASELINE]`
     - `[CONTACT_RESET]`
     - `[DETAIL_NOISE]`
     - `[GLOBAL_MORPH]`
     - `[GM_BOUNDARY]`
     - `[BAD_STRESS]`

4. High-resolution cross-attention:
   - query tokens attend back to high-resolution tokens
   - intended to recover local bad evidence that mean pooling would erase

5. Local map heads:
   - `qrs_event_a`
   - `qrs_event_b`
   - `baseline`
   - `contact`
   - `reset`
   - `flatline`
   - `detail`

6. Factor head:
   - predicts waveform-computable SQI/factor targets from the first six mechanism query tokens
   - `detector_agreement` is not an arbitrary pooled regression output; it is computed from the supervised local QRS event maps via a soft detector-agreement statistic and reused consistently in feature recovery/reporting.

7. Hierarchical class head:
   - `BAD_STRESS` query predicts `bad_logit`
   - `GM_BOUNDARY` query predicts `medium_given_nonbad_logit`
   - class probabilities:
     - `P(bad) = sigmoid(bad_logit)`
     - `P(medium) = (1 - P(bad)) * sigmoid(medium_given_nonbad_logit)`
     - `P(good) = (1 - P(bad)) * (1 - sigmoid(medium_given_nonbad_logit))`

8. Unified subtype head:
   - one `quality_subtype_head` predicts all good/medium/bad leaf subtypes
   - subtype probabilities can be summed back to class probabilities
   - this replaced the older separated `gm_subtype_head` / `bad_subtype_head` as the active formal line

9. Boundary heads:
   - `boundary_family_head`
   - `boundary_label_head`
   - supervise the two hard good/medium boundary families

10. Artifact head:
    - artifact presence/type/severity supervision is separated from the bad-quality decision
    - artifact-positive nonbad rows are explicitly allowed to be artifact-positive and bad-negative

## 4. Teacher / Diagnostic Factors

Current factor targets:

- `qrs_visibility`
- `detector_agreement`
- `baseline_step`
- `flatline_ratio`
- `sqi_basSQI`
- `non_qrs_diff_p95`
- `non_qrs_rms_ratio`
- `qrs_band_ratio`
- `template_corr`
- `amplitude_entropy`
- `contact_loss_win_ratio`
- `band_0p3_1`

These are intended to be waveform-computable or waveform-SQI-like. PCA/KNN/atlas purity features are not formal model inputs and should not be treated as final model claims.

## 5. Losses

The active training objective includes:

- hierarchical class NLL
- SQI/factor recovery loss
- local map loss
- artifact specificity/type/severity loss
- unified subtype leaf CE
- subtype-to-class consistency loss
- boundary family/label loss
- boundary evidence direction loss
- optional class weights in the latest tuning candidates
- optional pretraining modes for masked waveform or factorized proxy reconstruction

The current best candidate is:

`E4_query_highres_local_art_unified_lowaux_lr15e4`

Core settings:

- EventFactorizedSQIConformer
- high-res fusion on
- local supervision on
- artifact aux on
- unified subtype head on
- reduced auxiliary pressure relative to earlier unified-subtype runs
- learning rate `1.5e-4`

## 6. Current Results

Main report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\v112_accuracy_tuning_report.md`

### Baseline

Previous v112 quick baseline, 3 folds, 6 finetune epochs:

| candidate | mean acc | macro-F1 | good | medium | bad |
| --- | ---: | ---: | ---: | ---: | ---: |
| `E4_query_highres_local_art` | 0.9077 | 0.8867 | 0.8056 | 0.8829 | 0.9655 |
| `E4_query_highres_local_art_unified_subtype` | 0.9140 | 0.8951 | 0.8394 | 0.8676 | 0.9727 |

### 10-Epoch Tuning

| candidate | mean acc | macro-F1 | good | medium | bad | best fold acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `E4_query_highres_local_art_unified_lowaux_lr15e4` | 0.9238 | 0.9075 | 0.9172 | 0.8369 | 0.9768 | 0.9289 |
| `E4_query_highres_local_art_unified_factorprotect_lr15e4` | 0.9223 | 0.9058 | 0.9025 | 0.8386 | 0.9796 | 0.9250 |
| `E4_query_highres_local_art_unified_gmweighted_lr15e4` | 0.9201 | 0.9034 | 0.8847 | 0.8629 | 0.9677 | 0.9270 |

Best fold confusion for the current tuned candidate:

```text
[[847,  76,   7],
 [106, 1004, 44],
 [ 23,  30, 1888]]
```

Interpretation:

- Bad recall is high.
- Most remaining errors are good/medium boundary errors.
- Increasing bad modeling is unlikely to be the main path to synthetic/node 0.95 on v112.

### Calibration

Val-tuned class-prior/subtype-head calibration improved the best mean acc only to about `0.9260`, with best fold around `0.9324`. This suggests the remaining gap is not mainly a threshold/calibration issue.

### Pretraining

| candidate | mean acc | macro-F1 | good | medium | bad | best fold acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P2_ecg_mask_unified_lowaux_v112` | 0.9211 | 0.9045 | 0.8734 | 0.8782 | 0.9657 | 0.9289 |
| `P3_factorized_proxy_unified_gmweighted_v112` | 0.9212 | 0.9050 | 0.9094 | 0.8416 | 0.9716 | 0.9245 |

Light pretraining did not beat the best phase1 tuned model.

## 7. Seed Sweep

Active seed sweep command:

```powershell
.\.venv\Scripts\python.exe E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_event_factorized_sqi_conformer.py --stage phase1 --policy ptb_v112_gm_buffered_large_hybrid_s20260741 --folds 3 --split-seed 20260741 --seed 20260841 --seeds 3 --epochs 10 --batch-size 128 --num-workers 0 --candidates E4_query_highres_local_art_unified_lowaux_lr15e4
```

Completed: 2026-06-25.

The seed sweep retrained the same candidate from scratch over 3 folds x 3 seeds. It did not find a lucky seed that reaches 0.95.

Clean-test summary over 9 runs:

| metric | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| acc | 0.916183 | 0.002821 | 0.911034 | 0.920000 |
| record-macro supported F1 | 0.921247 | 0.003543 | 0.913602 | 0.924285 |
| sklearn macro-F1 | 0.898271 | 0.005358 | 0.890476 | 0.906501 |
| good recall | 0.850051 | 0.024794 | 0.811834 | 0.877381 |
| medium recall | 0.870442 | 0.014847 | 0.846019 | 0.898614 |
| bad recall | 0.971300 | 0.005451 | 0.961783 | 0.978692 |
| bad FPR on nonbad | 0.031694 | 0.007594 | 0.022189 | 0.041854 |

Best clean-test rows by accuracy:

| acc | record-macro supported F1 | good | medium | bad | bad FPR nonbad | confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.920000 | 0.923193 | 0.868817 | 0.877816 | 0.969603 | 0.029750 | `[[808,115,7],[86,1013,55],[23,36,1882]]` |
| 0.920000 | 0.924078 | 0.872043 | 0.863951 | 0.976301 | 0.041747 | `[[811,107,12],[82,997,75],[17,29,1895]]` |
| 0.917516 | 0.924285 | 0.871006 | 0.860585 | 0.969772 | 0.026906 | `[[736,105,4],[112,1000,50],[13,48,1957]]` |

Interpretation:

- Changing random seed changes the good/medium preference, but it does not remove the good/medium tradeoff.
- Bad remains consistently strong.
- The current architecture/loss/data combination is a stable ~0.916-0.920 clean-test model, not a hidden 0.95 model waiting for a better seed.
- This strengthens the conclusion that the next step should be structural Transformer/token/loss redesign for hard good/medium waveform evidence.

## 8. Feature Recovery: What Is Learned vs Weak

Seed-sweep averaged feature recovery:

| feature | corr all | corr good | corr medium | corr bad | min supported-class corr | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_step` | 0.9328 | 0.8975 | 0.8539 | 0.6628 | 0.6628 | 0.2554 |
| `non_qrs_diff_p95` | 0.8862 | 0.7916 | 0.8693 | 0.9435 | 0.7916 | 0.3623 |
| `template_corr` | 0.8457 | 0.2170 | 0.2524 | 0.6230 | 0.2055 | 1.3292 |
| `detector_agreement` | 0.8118 | 0.2793 | 0.3261 | 0.7516 | 0.2727 | 0.3255 |
| `flatline_ratio` | 0.7954 | 0.6911 | 0.6094 | 0.8110 | 0.6081 | 3.1084 |
| `non_qrs_rms_ratio` | 0.6157 | 0.5829 | -0.0242 | 0.7274 | -0.0242 | 1.6805 |
| `sqi_basSQI` | 0.6100 | 0.3886 | -0.3134 | 0.3951 | -0.3134 | 0.5813 |
| `qrs_visibility` | 0.3774 | n/a | 0.2540 | 0.2272 | 0.1976 | 3.6614 |
| `amplitude_entropy` | 0.2358 | 0.2947 | 0.1656 | 0.2233 | 0.1557 | 0.5967 |
| `contact_loss_win_ratio` | 0.2000 | 0.0107 | 0.2664 | 0.3250 | 0.0107 | 5.6199 |
| `qrs_band_ratio` | 0.0000 | n/a | n/a | n/a | n/a | 3.8550 |

Strong or usable recovery:

- `baseline_step`
- `non_qrs_diff_p95`
- `detector_agreement` at all-class level, though class-wise recovery is weaker
- `flatline_ratio`

Weak or unstable recovery:

- `qrs_visibility`
- `amplitude_entropy`
- `contact_loss_win_ratio`
- medium-class `sqi_basSQI`
- medium-class `non_qrs_rms_ratio`
- `qrs_band_ratio`, which is currently suspicious/degenerate in recovery and should be audited for target variance, transform, and normalization effects

Important interpretation:

The model can see coarse artifacts and many global quality signals, but it still does not reliably learn the subtle waveform evidence that separates hard good from hard medium.

## 9. Current Bottleneck

The present bottleneck is not:

- lack of a subtype head
- lack of a bad head
- threshold calibration
- simple learning-rate tuning
- light masked pretraining
- feature-only classifier capacity

The present bottleneck appears to be:

1. Good/medium boundary rows contain evidence that is local, subtle, and subtype-dependent.
2. The model learns gross bad evidence well but still confuses:
   - good-like medium
   - medium-like good
   - low-QRS but still acceptable-ish windows
   - mild artifact that should not become bad
   - baseline/non-QRS dominance that should push medium
3. Some teacher factors that should help the boundary are not recovered robustly from waveform.
4. Unified subtype supervision improves class accuracy but can compete with some factor recovery objectives.

## 10. Questions for Transformer Redesign

The next AI/researcher should focus on these questions:

1. How should the tokenizer expose `qrs_visibility`, `contact_loss`, and `amplitude_entropy` better without using explicit SQI features as inference input?
2. Should the model use beat-centered tokens or R-peak candidate tokens in addition to fixed patch tokens?
3. Should local map supervision become stronger for QRS visibility and contact/reset/dropout, instead of being a small auxiliary loss?
4. Should good/medium boundary training use pairwise or ranking losses between matched hard subtypes?
5. Should the hierarchical class head be directly derived from the unified subtype probabilities, or should the current main class head remain separate?
6. How can subtype supervision be prevented from weakening factor recovery?
7. Would a local worst-token / top-k artifact pooling mechanism help more than attention-pooling for quality degradation?
8. Should `sqi_basSQI`, `qrs_band_ratio`, and `template_corr` be represented through differentiable waveform modules rather than generic regression from query embeddings?
9. How can the model learn contact/reset/flatline evidence as a structured event-detection task rather than a scalar feature?
10. Is the current hierarchical probability formulation too restrictive for ambiguous good/medium rows, or does it help enough to keep?

## 11. Do Not Do

For the next model-design step:

- Do not use route/rule artifacts as final model results.
- Do not use MLP/tree/47-feature classifiers as formal inference models.
- Do not feed PCA/KNN/atlas purity features into the model.
- Do not use BUT test for training, feature selection, or model selection.
- Do not keep sweeping class weights blindly; previous sweeps show the gap is structural.

## 12. Most Useful Files for Review

Code:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_event_factorized_sqi_conformer.py`

Dataset report:

- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v112_gm_buffered_large_hybrid\v112_gm_buffered_large_hybrid_report.md`

Model result report:

- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\v112_accuracy_tuning_report.md`

Metrics CSVs:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_metrics.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_feature_recovery.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_record_metrics.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_metrics.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_feature_recovery.csv`

Seed sweep logs:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_logs\run_v112_best_seed_sweep_20260625_054253.log`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_logs\run_v112_best_seed_sweep_20260625_054253.err.log`
