# Waveform Transformer Research Status - 2026-06-17

## Executive Summary

当前结论需要换思路：47-feature / reduced-feature MLP 能在 held-out original BUT test 上很高，是因为它直接吃到了 PCA shell、boundary margin、label-neighborhood purity 这类“地图特征”。这些特征证明 good/medium/bad 在规则空间里是可分的，但它们不是最终 waveform-only Transformer 推理时天然可见的输入。

截至本报告，正式 waveform-only Transformer 的 best frontier 仍在 original_test_all_10s+ 约 `0.84-0.86`，不是 `0.96`。`0.96` 是 47-feature tabular upper bound，不是 waveform-only 结果。

最重要的新证据是 aux-pred upper-bound audit：把当前 Transformer 预测出来的 auxiliary features 再喂给 HGB / ExtraTrees / Logistic Regression，original_test 仍只有约 `0.846`。这说明瓶颈不是最后分类头，也不是“换一个浅层 classifier 就能解决”，而是 encoder/tokenizer 没有稳定恢复关键 ECG quality features。

下一步不应该继续盲目加 loss 或扩大普通 Transformer；应该转向 literature-guided, SQI-aware waveform representation：RR interval / beat reliability / detector agreement / band-power / baseline drift / local worst-case artifact token 都要成为 waveform encoder 的结构性任务。

## Formal Evaluation Contract

- Training / selection: PTB-generated synthetic / Clean-SemiClean node diagnostic only.
- BUT usage: held-out report-only, never used for training, feature selection, threshold selection, or promotion.
- Final model target: inference input must be ECG waveform only.
- 47 SQI/geometry features: allowed only as training-time teacher targets and diagnostic upper bound.
- Main BUT buckets:
  - `original_test_all_10s+`: 8477 windows = good 3640, medium 4426, bad 411.
  - `original_all_10s+`: 32956 windows = good 17043, medium 10628, bad 5285.
  - `bad_core_nearboundary`: 119 windows.
  - `bad_outlier_stress`: 292 windows, mostly record 111001.

## Current Best Results

| Model / diagnostic | Formal waveform-only? | Inference input | original_test acc | good | medium | bad | Main interpretation |
|---|---:|---|---:|---:|---:|---:|---|
| 47-feature tabular upper bound | No | 47 SQI/geometry incl. target PCA/neighborhood | 0.963548 | 0.956 | 0.973 | 0.927 | Shows the rule/geometry space can separate BUT. Not a waveform model. |
| reduced top14 MLP | No | selected SQI/geometry features | 0.898431 | 0.892 | 0.897 | 0.973 | Stronger interpretability, still not waveform-only. |
| best waveform-only Transformer `featurefirst_top20_hardrec_qfeatbin_qrsbase_a050` | Yes | waveform-derived tokens only | 0.855963 | 0.890 | 0.884 | 0.253 | Current formal frontier. Better good/medium balance, but bad outlier stress still not learned. |
| prior waveform-only Transformer `featurefirst_top20_hardrec_a050` | Yes | waveform-derived tokens only | 0.844520 | 0.848 | 0.894 | 0.275 | Previous best before quantile auxiliary binning. |
| aux-pred upper bound on same Transformer | Yes for model input; diagnostic classifier on predicted aux | predicted auxiliary features | 0.846172 | 0.860 | 0.885 | 0.307 | Final classifier is not the bottleneck; feature recovery is. |
| waveform primitive proxy | Yes | deterministic waveform stats | ~0.827-0.831 | mixed | mixed | high in some configs | Some bad stress is waveform-visible, but good/medium balance degrades. |

## What We Tried

### Data / target construction

- N6800/N7000/N7100 trim-bad and geometry-repair node ladder promoted on Clean/SemiClean/node diagnostic.
- N7200 transparent qrs-low rule-mode artifact passed Clean/SemiClean node gates, but transferred poorly to original BUT.
- Boundary block generator added:
  - `gm_clean_overlap_body`
  - `gm_good_rescue_pc1flat_qrsvisible`
  - `gm_medium_qrslow_hardneg`
  - `gm_visible_qrs_medium_detail`
  - `bad_core_guard`
  - `bad_controlled_outlier`
  - `bad_extreme_stress_holdout`
- Full-boundary / N17043 experiments widened synthetic coverage and clarified domain gaps.

### Model attempts

- UFormer / waveform-only Transformer variants.
- SQI-query token Transformer.
- Filterbank / primitive-channel patch variants.
- Event-QRS and qrs-heavy tokenization.
- Predicted-feature fusion from waveform only.
- Bad-stress heads, specificity penalties, bad gating.
- Hard-feature auxiliary losses.
- Long auxiliary pretraining, wide auxiliary heads, qrsbase variants.
- Quantile auxiliary binning for hard features (`qfeatbin`) completed. It improves original_test acc to `0.855963` in the qrsbase variant, but bad outlier stress remains weak (`0.003425` raw, `0.068493` bad-calibrated). This is a small frontier improvement, not a structural breakthrough.

### Key negative result

More auxiliary losses can make synthetic validation look excellent, but that has not solved original BUT transfer. The recurring failure is:

- synthetic/node acc very high,
- original_test stuck around `0.80-0.86`,
- bad_core near-boundary often recovered,
- bad_outlier_stress still missed,
- trying to recover bad stress usually creates false bad on non-bad windows.

## Feature Recovery: Learned vs Not Learned

### Features the waveform Transformer can partially learn

These are relatively aligned with local/global waveform statistics:

- `pc1`
- `pca_margin`
- `flatline_ratio`
- `non_qrs_diff_p95`
- `band_15_30`
- `band_30_45`
- `pc3` to a moderate degree

Example from `featurefirst_top20_hardrec_a050` aux recovery:

| Feature | Synthetic corr | Original corr pattern |
|---|---:|---|
| `pca_margin` | ~0.748 | high on original |
| `non_qrs_diff_p95` | ~0.716 | very high on original |
| `band_15_30` | ~0.713 | useful |
| `band_30_45` | ~0.690 | useful |
| `flatline_ratio` | ~0.555 | useful but not enough |

### Hard features still not learned well enough

These are the blockers:

| Feature | Current issue | Why it is hard from raw waveform |
|---|---|---|
| `qrs_visibility` | synthetic corr ~0.17, original moderate only after domain shift | It is not just high-frequency energy; it requires knowing where QRS should be and whether QRS remains reliable under noise. |
| `detector_agreement` | synthetic corr ~0.26 | It is an algorithm agreement concept; model needs to emulate multiple detector views or RR consistency. |
| `baseline_step` | synthetic corr ~0.32 | Local baseline jumps can be sparse and are diluted by mean pooling. |
| `sqi_basSQI` | weak | Requires stable low-frequency vs cardiac band integration, not just patch morphology. |
| `pc2 / pc3` | pc2 especially poor | PCA axes are target-distribution geometry, not pure physiology. |
| `boundary_confidence` / `knn_label_purity` | weak | These are label-neighborhood geometry features, not directly observable waveform properties. |

The important distinction: some features are waveform-computable but need better inductive bias; some are target/atlas geometry and should not be treated as recoverable waveform properties.

## Why The 0.96 Feature Model Does Not Transfer Into A Waveform Claim

The 47-feature model uses target-aware geometry:

- PCA coordinates from the target shell.
- Boundary margin / region confidence.
- kNN label-neighborhood purity.
- QRS / detector / SQI statistics already computed outside the neural waveform model.

This is a valid diagnostic oracle and a useful data-generation guide. It is not a clean final waveform model. The useful takeaway is not “MLP is the final answer”; it is:

> The labels are separable if the model can recover beat reliability, baseline/frequency quality, and target-shell geometry proxies.

Our next model has to learn the waveform-computable part of that map instead of being handed the map.

## Why Original BUT Still Breaks The Model

### 1. Record/domain shift is real

Original train->test capacity diagnostic on BUT labels already shows severe generalization difficulty:

- train acc ~0.9068
- val acc ~0.8712
- test acc ~0.7197
- test bad recall ~0.290

So original BUT is not merely “same distribution plus noise”; the test slice has record/domain structure that can break even BUT-trained diagnostics.

### 2. Bad has two different domains

- `bad_core_nearboundary` is learnable and often recovered well.
- `bad_outlier_stress` is mostly record 111001 and behaves like a different stress domain.

The current best waveform model has strong bad behavior on original_all because original_all contains many bad-like windows, but original_test bad_outlier_stress remains weak. The latest qfeatbin qrsbase model reaches original_all bad recall `0.928666`, while original_test bad recall is only `0.253041` and bad_outlier_stress is almost absent. This is why a single bad recall number can look contradictory across buckets.

### 3. Good/medium requires beat-level semantics

Visual audits show many good/medium errors are not SNR-only:

- Some medium windows still have visible QRS but degraded detail.
- Some good rescue windows look slightly contaminated but preserve QRS reliability.
- The meaningful boundary is closer to “what remains clinically measurable” than to global noise strength.

## Literature-Guided Direction

The next architecture should borrow the useful pieces from time-series and ECG representation learning:

- Patch-level time-series Transformer design from PatchTST: patching preserves local semantics and reduces attention cost, but ECG quality needs more than generic patch tokens. Reference: https://arxiv.org/abs/2211.14730
- Masked ECG Transformer / MTECG: ECG-specific masked reconstruction plus positional embeddings is a better pretraining path than pure supervised classification. Reference: https://arxiv.org/abs/2309.07136
- ECG self-supervised learning: contrastive / predictive pretraining can improve robustness and representation transfer. Reference: https://arxiv.org/abs/2103.12676
- SQI fusion literature: useful ECG quality signals include spectral distribution, higher moments, and detector agreement; these should be architectural targets, not only post-hoc features. References: https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.00727/full and https://www.robots.ox.ac.uk/~gari/papers/IOP_ECG_SQI_Clifford_et_al.pdf

## Recommended New Architecture Direction

### 1. RR / beat reliability branch

Add a waveform-only beat token stream:

- lightweight R-peak candidate detector,
- beat-centered windows,
- RR interval consistency token,
- peak amplitude / local contrast,
- QRS width / prominence proxy,
- detector-agreement emulation using multiple simple differentiable or fixed candidate views.

This directly targets:

- `qrs_visibility`
- `detector_agreement`
- good vs medium boundary
- bad stress that destroys QRS reliability

### 2. Frequency / baseline SQI branch

Add fixed waveform-derived channels before patching:

- low-frequency baseline channel,
- high-pass morphology channel,
- derivative channel,
- band energies for roughly 0-1, 1-5, 5-15, 15-30, 30-45 Hz,
- local wavelet/detail tokens.

This targets:

- `sqi_basSQI`
- `baseline_step`
- `non_qrs_diff_p95`
- `band_15_30`
- `band_30_45`
- bad outlier stress

### 3. Query-token Transformer with worst-case pooling

Do not rely on mean pooling. Use:

- `[QRS]`, `[RR]`, `[BASELINE]`, `[BAND]`, `[DETAIL]`, `[BAD_STRESS]`, `[GM_BOUNDARY]` query tokens,
- mean + max + top-k artifact pooling,
- class head from structured query tokens.

Quality labels often depend on a small corrupted segment; mean pooling hides exactly the information we need.

### 4. Module-first training checks

Before full classification search, each candidate must pass module tests:

| Module target | Minimum useful diagnostic |
|---|---:|
| `qrs_visibility` recovery | corr >= 0.45 |
| `detector_agreement` recovery | corr >= 0.45 |
| `baseline_step` recovery | corr >= 0.55 |
| `sqi_basSQI` recovery | corr >= 0.45 |
| `flatline_ratio` recovery | corr >= 0.60 |
| bad stress detection specificity | bad stress up without non-bad false-bad explosion |

If these do not move, full classification acc will probably stay in the 0.84 wall.

## Data Coverage Still Missing

The uncovered pieces are not just “more samples”; they are specific waveform regimes:

1. Record-111-style bad outlier stress:
   - intermittent contact loss,
   - low-frequency plateau / drift,
   - preserved spike-like segments that confuse bad/non-bad,
   - long-duration bad morphology not represented by current bad core.

2. Good/medium boundary with visible QRS:
   - QRS measurable but P/T/detail degraded,
   - low QRS visibility but not fully bad,
   - local bursts that do not dominate the full 10s window.

3. Baseline and detector-agreement gaps:
   - synthetic data does not yet force the model to emulate R-peak consistency or detector disagreement.

4. Target geometry features:
   - `boundary_confidence` and `knn_label_purity` are still useful for diagnostics, but they are not waveform facts. We should stop optimizing as if they must be exactly reconstructed from raw ECG.

## Immediate Next Plan

1. Freeze this report and keep current waveform experiments as baseline.
2. Stop spending large runs on generic aux-loss escalation unless it improves hard feature recovery.
3. Implement a new experiment-only `BeatReliabilitySQITransformer`:
   - raw waveform patch stream,
   - RR/beat candidate stream,
   - filterbank/baseline stream,
   - structured SQI query tokens,
   - top-k artifact pooling.
4. Pretrain on PTB synthetic with waveform-computable teacher targets only:
   - QRS visibility proxy,
   - detector agreement proxy,
   - RR consistency,
   - baseline step / basSQI,
   - band/detail reconstruction.
5. Fine-tune classification only after module recovery passes.
6. Run BUT bucketed report-only:
   - original_test_all_10s+,
   - original_all_10s+,
   - bad_core_nearboundary,
   - bad_outlier_stress.

## Key Artifacts

- Feasibility audit: `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_feature_feasibility_audit.md`
- Primitive learnability: `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_primitive_learnability_report.md`
- Current iteration compare: `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/current_waveform_transformer_iteration_compare.md`
- Aux-pred upper bound: `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/aux_pred_upper_bound/aux_pred_upper_bound_report.md`
- Aux-pred metrics: `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/aux_pred_upper_bound/aux_pred_upper_bound_metrics.csv`
- Aux feature recovery: `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/aux_pred_upper_bound/aux_pred_feature_recovery.csv`

## Bottom Line

We are not blocked because ECG quality is impossible. The feature upper bound proves the opposite. We are blocked because the current waveform Transformer does not have the right structural pressure to learn RR reliability, detector agreement, baseline/frequency SQIs, and local worst-case artifacts from raw waveform.

The next strong attempt should be a beat/RR-aware SQI Transformer, not another broad class-weight or auxiliary-loss sweep.
