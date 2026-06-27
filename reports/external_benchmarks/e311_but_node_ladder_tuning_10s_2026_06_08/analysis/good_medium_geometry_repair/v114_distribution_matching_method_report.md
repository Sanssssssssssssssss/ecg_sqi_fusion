# v114 Distribution Matching Method Report

Date: 2026-06-27

## Executive Summary

Current conclusion:

- The correct formal objective is **class-balanced augmentation with class-conditional distribution preservation**: balance `P(class)`, but preserve the original BUT-like `P(features | class)` as much as possible.
- We should **not force every subtype to the same count**. Equal subtype quotas can damage the natural class-conditional distribution because rare subtype islands are artificially amplified.
- MMD is lower-is-better; PCA density overlap is higher-is-better.
- The previous very-low MMD line proves that low MMD is possible when BUT-native/non-clean support is allowed, but it is an **upper bound / floor diagnostic**, not a clean-only formal synthetic claim.
- Under the current strict clean-only rule, PTB clean carriers + BUT clean style anchors + BUT non-clean feature targets do **not yet** reach the target `all_labels` MMD around `0.08`. The blocker is not the selector alone; it is missing waveform support for BUT-like bad morphology.

## Current Metric Contract

All comparisons use the v110 metric contract:

- Target reference: BUT train+val only.
- Features: unified 47 waveform/SQI feature extractor.
- Main metric: subtype-level RBF-MMD by `class x subtype`.
- Summary metric: class-wise subtype-median RBF-MMD and PCA density overlap.
- Supplement: global `all_labels`, `class_good`, `class_medium`, `class_bad` RBF-MMD.
- PCA is for visualization/density overlap audit, not model input.
- BUT test is never used for generation, target selection, MMD optimization, or model selection.

## Method Lines

### 1. Upper-Bound Native/Bayesian Match

Purpose:

This line answers: "If the candidate support already contains BUT-native style and non-clean support, how low can the v110 MMD gox"

Status:

- Diagnostic upper bound only.
- Not a formal PTB clean-only synthetic result.
- It may include BUT-native/non-clean support, so it is useful for understanding the metric floor but not for final data-generation claims.

Key result:

| Method | all-label MMD | good MMD | medium MMD | bad MMD | PCA overlap |
|---|---:|---:|---:|---:|---:|
| `hybrid_v114_bayesian_match_natural_prior_eval_s20260793` | 0.0025 | 0.0012 | 0.0016 | 0.0010 | 1.0000 |

Interpretation:

This explains why earlier runs could reach very low MMD. The metric can be made almost perfect when the candidate support already contains BUT-like native/non-clean morphology. This is an upper-bound sanity check, not the formal clean-only path.

Report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v114_but_style_residual_hybrid\s20260793\hybrid_v114_bayesian_match_natural_prior_eval_s20260793`

### 2. Constrained MMD / KMM Hybrid Line

Purpose:

This line optimizes the selected dataset distribution directly under a minimum PTB/generated fraction constraint. It uses MMD/KMM-style constrained selection and is useful for studying the tradeoff between "more formal PTB support" and "distribution similarity."

Status:

- Diagnostic/transition line.
- Not the final clean-only claim if BUT-native/non-clean support remains in the candidate pool.

Key results:

| Method | PTB/generated min target | all-label MMD | good MMD | medium MMD | bad MMD | max issue |
|---|---:|---:|---:|---:|---:|---|
| `constrained_ptbmin20_s20260810` | 20% | 0.0082 | 0.0124 | 0.0108 | 0.0043 | very good distribution, but not clean-only formal |
| `constrained_ptbmin50_s20260811` | 50% | 0.0298 | 0.0292 | 0.0368 | 0.0160 | still strong distribution |
| `constrained_ptbmin100_s20260812` | 100% requested, actual mixed/generated support about 55% | 0.0594 | 0.0418 | 0.0912 | 0.0722 | class MMD good, but subtype locality degrades |

Interpretation:

This line shows the optimization machinery can drive global MMD low when the candidate pool has enough BUT-like support. It also shows why "just sample harder" is not enough for strict clean-only: once native/non-clean support is removed, the candidate support changes and MMD rises sharply.

Code:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_constrained_mmd_kmm.py`

### 3. Formal Clean-Only Style Mechanism

Purpose:

Formal clean-only attempt:

- PTB clean carrier supplies physiology.
- BUT clean anchors supply acquisition style.
- BUT non-clean rows provide target feature distributions only.
- No BUT medium/bad waveform donor.

Key result:

| Method | all-label MMD | good MMD | medium MMD | bad MMD | bad PCA overlap |
|---|---:|---:|---:|---:|---:|
| `ptb_v114_cleanonly_style_mechanism_s20260800` | 0.2963 | 0.3408 | 0.2098 | 0.7127 | 0.0253 |

Interpretation:

Good/medium moved somewhat toward BUT, but bad remained far away. This suggests the hand-coded bad mechanisms were not expressive enough.

Code:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_cleanonly_style_generator.py`

### 4. Formal Clean-Only Target-Param Generator

Purpose:

For each generated synthetic row, sample a BUT target feature vector and transform a PTB clean carrier using feature-driven parameters:

- baseline drift/step/ramp;
- high-frequency/detail noise;
- QRS attenuation;
- detector/template instability;
- contact/flatline/reset segments.

Key result:

| Method | all-label MMD | good MMD | medium MMD | bad MMD | bad subtype median MMD |
|---|---:|---:|---:|---:|---:|
| `ptb_v114_cleanonly_target_param_s20260820` | 0.2807 | 0.3497 | 0.2256 | 0.8125 | 1.2303 |

Interpretation:

The target-param transform did not solve the support gap. It slightly improved global MMD versus the first clean-only line, but bad got worse. PCA showed generated bad collapsing into an artificial vertical band rather than covering BUT bad islands.

Code:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_cleanonly_target_param_generator.py`

### 5. Formal Clean-Only ABC-SMC Generator

Purpose:

Replace one-shot hand-tuned generation with parameter-space accept/update:

1. Generate candidate perturbation parameters.
2. Recompute unified 47 features.
3. Score against BUT train+val target distribution.
4. Keep elite candidates.
5. Resample parameters around elites in later rounds.

This follows an approximate Bayesian computation / sequential Monte Carlo style loop, but remains formal clean-only:

- PTB clean carriers only;
- BUT clean style anchors only;
- BUT non-clean rows only provide feature targets, not waveform donors.

#### 5a. Subtype-equal allocation

This is now considered the wrong default for the formal objective, because it equalizes rare and common subtypes.

| Method | all-label MMD | good MMD | medium MMD | bad MMD |
|---|---:|---:|---:|---:|
| `ptb_v114_cleanonly_abc_smc_s20260830` | 0.2839 | 0.4083 | 0.3275 | 0.6366 |

#### 5b. Natural subtype allocation

Class is balanced, subtype counts follow natural BUT proportions within each class.

| Method | all-label MMD | good MMD | medium MMD | bad MMD |
|---|---:|---:|---:|---:|
| `ptb_v114_cleanonly_abc_smc_s20260831` | 0.2996 | 0.4046 | 0.3190 | 0.7898 |

#### 5c. Class-level selector

Class is balanced, but subtype count is not fixed. The selector chooses from the entire class candidate pool to match `P(features | class)`.

| Method | all-label MMD | good MMD | medium MMD | bad MMD | bad PCA overlap |
|---|---:|---:|---:|---:|---:|
| `ptb_v114_cleanonly_abc_smc_s20260832` | 0.2899 | 0.4111 | 0.3508 | 0.6424 | 0.0346 |

Interpretation:

Changing subtype allocation did not fix the clean-only gap. Therefore the bottleneck is not merely "how to allocate subtype counts." The stricter conclusion is:

> The current clean-only generator cannot yet create enough BUT-like bad support. Selector/MMD optimization cannot select samples that do not exist in the candidate bank.

Code:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_cleanonly_abc_smc_generator.py`

## Why Subtype Equalization Is Wrong for the Current Goal

If the goal is "add synthetic data without changing the original class morphology," the mathematical target is:

```text
Make P_train(class) balanced,
but preserve P_synthetic(features | class) ~= P_BUT(features | class).
```

That means subtype is a diagnostic/stratification variable, not necessarily a balancing variable. Equal subtype balancing changes the mixture weights:

```text
P(features | class) = sum_subtype P(features | class, subtype) * P(subtype | class)
```

If we force `P(subtype | class)` to uniform, we can destroy the original class distribution even when each subtype looks locally reasonable.

Therefore future formal generation should use:

- class-balanced output;
- class-level MMD/KMM/herding objective;
- subtype-level diagnostics only as guardrails;
- no forced equal subtype quota unless explicitly studying rare subtype robustness.

## Why Clean-Only Still Fails

The biggest failure is bad morphology.

BUT bad often has:

- very low `detector_agreement` / `sqi_iSQI`;
- surprisingly high or moderate `sqi_bSQI`;
- not necessarily huge `band_30_45`;
- dense/periodic/structured contamination rather than simple high-frequency noise;
- class-specific islands in PCA space.

The current clean-only generator often creates:

- too much high-frequency energy;
- too much artificial spike/edge noise;
- QRS visibility that remains too high for some bad subtypes;
- PCA vertical bands instead of BUT bad islands.

This is why clean-only MMD stays around `0.28-0.30` globally and `0.64+` for bad class.

## Current Implementation Map

Primary scripts:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\build_v114_but_style_residual_hybrid.py`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_constrained_mmd_kmm.py`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_cleanonly_style_generator.py`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_cleanonly_target_param_generator.py`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_v114_cleanonly_abc_smc_generator.py`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\audit_v110_metric_distribution.py`

Machine-readable summary:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v114_distribution_method_summary_metrics.csv`

Key report folders:

- Upper bound:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v114_but_style_residual_hybrid\s20260793`
- Clean-only style:
  `E:\GPTProject2\ecg_keep_20260528_172844\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v114_cleanonly_style_generator\s20260800`
- Clean-only target-param:
  `E:\GPTProject2\ecg_keep_20260528_172844\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v114_cleanonly_target_param\s20260820`
- Clean-only ABC-SMC:
  `E:\GPTProject2\ecg_keep_20260528_172844\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\v114_cleanonly_abc_smc`

## Recommended Next Step

Do not continue tuning subtype quotas.

The next useful step is to improve the clean-only bad generator support:

1. Keep the formal contract:
   PTB clean carrier + BUT clean style anchors + BUT non-clean feature targets only.
2. Generate bad candidate mechanisms that match the observed BUT facts:
   low detector agreement without exploding high-frequency bands.
3. Optimize at class-level MMD:
   balance class counts, not subtype counts.
4. Use subtype-level metrics only to diagnose which bad mechanism remains missing.

If that still cannot reduce bad MMD, then the honest conclusion is:

> Strict clean-only PTB carrier generation lacks support for the BUT bad morphology; reaching low MMD requires either a richer physiology/artifact generator or allowing controlled BUT-derived residual/native support as a separate semi-synthetic line.

