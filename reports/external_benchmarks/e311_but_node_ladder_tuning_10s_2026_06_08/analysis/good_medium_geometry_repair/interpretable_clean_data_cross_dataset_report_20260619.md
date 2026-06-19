# Interpretable Clean-Data Rule and PTB/BUT Cross-Dataset Check

Date: 2026-06-19

## Decision

We should keep the task fixed-length 10s for now and clean the supervision, not move to variable-length input.

The interpretable rule is:

1. A window is eligible for training only if the whole 10s window lies inside one consensus label segment and has enough margin from both segment boundaries.
2. The recommended high-confidence main training rule is `min_margin_sec >= 5`.
3. `outlier_low_confidence` should not be mixed blindly into the main clean-body training set. It is retained as a named stress bucket or used in a controlled stress curriculum only.
4. `good_medium_overlap` is not discarded; it is a real boundary bucket and should remain visible in metrics and waveform panels.
5. PCA/KNN/atlas geometry can be used for analysis, but formal model auxiliary targets should be waveform-computable SQI/QRS/baseline/detail features.

This makes the data rule explainable: we are not deleting examples to inflate metrics; we are separating stable supervision, boundary supervision, and stress/outlier supervision.

## Literature Anchors

- BUT QDB is annotated sample-by-sample and explicitly targets ECG quality assessment under varying quality over time. This supports using segment-consensus and boundary-margin rules instead of treating every centered crop as equally reliable. Source: [PhysioNet BUT QDB](https://physionet.org/content/butqdb/1.0.0/).
- PTB-XL uses fixed 10s ECG records and recommended benchmark splits, reinforcing the value of fixed-length comparability when the label covers the whole record/window. Source: [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/).
- Recent ECG SQA work argues that quality should be defined by whether reliable physiological measurements can be derived, not just by visual cleanness or SNR. It uses QRS/PQRST template consistency and RR plausibility, and it adds context around segment boundaries to capture complete complexes. Source: [Scientific Reports 2025 ECG SQA](https://www.nature.com/articles/s41598-025-25365-x).
- PhysioNet/MIMIC SQI notes that ECG quality metrics combine time/frequency statistics, QRS detector performance, and correlation/context measures; some artifact types remain hard and should be reported carefully. Source: [PhysioNet Signal Quality](https://archive.physionet.org/mimic2/UserGuide/node42.html).
- Cross-dataset ECG noise work emphasizes evaluating generalization across data sources/noise types, not only within one dataset. Source: [Kalpande et al. 2025](https://arxiv.org/html/2502.14522v1).

## Materialized Protocols

All protocols are fixed 10s. No variable-length input was created.

| policy | total | good | medium | bad | intended use |
| --- | ---: | ---: | ---: | ---: | --- |
| `margin_ge_5s_drop_outlier` | 21,575 | 11,228 | 6,265 | 4,082 | high-confidence clean-body training |
| `margin_ge_5s_keep_outlier` | 29,410 | 15,042 | 9,212 | 5,156 | stress-inclusive diagnostic |
| `clean_core_plus_overlap_margin2` | 17,823 | 11,458 | 6,365 | 0 | good/medium boundary sanity |
| `clean_core_only_margin2` | 4,944 | 3,196 | 1,748 | 0 | easiest clean-core sanity |

Protocol report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\clean_but_protocols_report.md`

## Cleaned Data Visuals

High-confidence clean-body, outliers removed:

![margin_ge_5s_drop_outlier](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocol_visuals\margin_ge_5s_drop_outlier_waveform_examples.png)

Strict-margin protocol, outliers retained as stress:

![margin_ge_5s_keep_outlier](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocol_visuals\margin_ge_5s_keep_outlier_waveform_examples.png)

Visual interpretation:

- Good and medium clean/overlap windows are meaningful and learnable under fixed 10s.
- Bad core and bad outlier stress are visually high-artifact, but outlier stress is not a free add-on; it disrupts good/medium classification if mixed into one ordinary 3-class objective.

## Model Used For This Round

Runner:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_clean_but_dualview_hier_transformer.py`

Input:

- fixed 10s waveform-derived channels only;
- physical/global-normalized waveform;
- robust waveform;
- derivative;
- long baseline/trend;
- high-pass residual;
- local detail/envelope channels.

Training-only auxiliary targets:

- QRS visibility/prominence/band ratio;
- detector agreement and template correlation;
- baseline step, flatline/contact loss;
- non-QRS/detail/band energy;
- entropy, Hjorth, wavelet;
- 7 SQI including `sqi_basSQI`.

Excluded from formal aux:

- `pc*`;
- `pca_margin`;
- `boundary_confidence`;
- `region_confidence`;
- `knn_label_purity`.

## Cross-Dataset Results

### Policy: `margin_ge_5s_drop_outlier`

| train domain | eval bucket | acc | good R | medium R | bad R |
| --- | --- | ---: | ---: | ---: | ---: |
| PTB synthetic | PTB synthetic test | 0.9682 | 0.9184 | 0.9854 | 0.9793 |
| PTB synthetic | BUT clean test | 0.8653 | 0.9602 | 0.8676 | 0.0169 |
| PTB synthetic | BUT original test | 0.6928 | 0.6687 | 0.7741 | 0.0316 |
| BUT clean | BUT clean test | 0.9300 | 0.9990 | 0.9047 | 0.7881 |
| BUT clean | PTB synthetic test | 0.7883 | 0.9017 | 0.8985 | 0.0000 |
| BUT clean | BUT original test | 0.7975 | 0.9217 | 0.7481 | 0.2287 |

Metrics:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\interpretable_clean_cross_dataset\interpretable_clean_cross_dataset_margin_ge_5s_drop_outlier_dualview_convtx_hier_metrics.csv`

### Policy: `margin_ge_5s_keep_outlier`

| train domain | eval bucket | acc | good R | medium R | bad R |
| --- | --- | ---: | ---: | ---: | ---: |
| PTB synthetic | PTB synthetic test | 0.9815 | 0.9414 | 0.9976 | 0.9793 |
| PTB synthetic | BUT clean/stress test | 0.6676 | 0.6513 | 0.7003 | 0.4180 |
| PTB synthetic | BUT original test | 0.6518 | 0.6327 | 0.6941 | 0.3650 |
| PTB synthetic | BUT bad core | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| PTB synthetic | BUT bad outlier stress | 0.1062 | 0.0000 | 0.0000 | 0.1062 |
| BUT keep-outlier | BUT clean/stress test | 0.7455 | 0.9756 | 0.5935 | 0.3794 |
| BUT keep-outlier | PTB synthetic test | 0.7155 | 0.6632 | 0.8758 | 0.0000 |
| BUT keep-outlier | BUT original test | 0.7302 | 0.9747 | 0.5700 | 0.2895 |

Metrics:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\interpretable_clean_cross_dataset\interpretable_clean_cross_dataset_margin_ge_5s_keep_outlier_dualview_convtx_hier_metrics.csv`

## Main Finding

The clean rule works: `margin_ge_5s_drop_outlier` gives a learnable fixed-10s BUT clean-body task, and a waveform-only Transformer reaches clean test acc 0.93 in the quick cross run and 0.988 in the Conformer clean-body run.

The cross-domain gap is the blocker:

- PTB synthetic learns itself very well.
- BUT clean learns itself reasonably well.
- PTB -> BUT fails mainly on BUT bad.
- BUT -> PTB fails mainly on PTB bad.

So the current PTB synthetic bad morphology and BUT bad/stress morphology are not aligned enough. This is not solved by variable-length input.

## Proposed Clean Data Rule

Use three tiers:

### Tier A: high-confidence training

Criteria:

- fixed 10s window;
- fully inside one consensus quality segment;
- `min(left_margin, right_margin) >= 5s`;
- no `outlier_low_confidence`.

Use:

- main 3-class clean-body training;
- model selection on clean-body diagnostic;
- good/medium boundary analysis.

Current policy:

- `margin_ge_5s_drop_outlier`.

### Tier B: boundary/stress diagnostic

Criteria:

- fixed 10s;
- segment margin >= 5s;
- includes `outlier_low_confidence`, but tracked as a separate named bucket.

Use:

- stress report;
- controlled stress curriculum only after clean-body model is stable;
- never mix blindly into ordinary CE as if it were equally reliable.

Current policy:

- `margin_ge_5s_keep_outlier`.

### Tier C: exclude from training

Criteria:

- unmatched consensus segment;
- too close to label transition boundary;
- ambiguous/cropped segment with insufficient margin;
- label segment too short to support a stable fixed 10s interpretation.

Use:

- not in training;
- can be counted in audit only.

## PTB Synthetic Fitting Rule Update

PTB synthetic generation should become interpretable in the same tiered language:

1. `good`: stable QRS/PQRST, high detector/template agreement, low baseline/contact loss.
2. `medium`: QRS still usable, but detail/P/T/baseline morphology partly degraded.
3. `bad_core`: rhythm/QRS extraction unreliable but morphology resembles BUT bad core.
4. `bad_stress`: BUT record-111-like artifact/contact stress; train only with controlled ratio and dedicated stress head.

Do not fit PTB using atlas-only or KNN-only geometry as a formal claim. Use those only to discover candidate regions, then translate them into waveform-computable targets such as RR plausibility, detector agreement, baseline/contact, band/detail, and template consistency.

## Immediate Next Experiment

1. Keep `margin_ge_5s_drop_outlier` as the main clean-body training set.
2. Build a small controlled `bad_stress` synthetic block, not a broad bad outlier dump.
3. Train with a separate bad-stress auxiliary head or curriculum:
   - stage 1: clean-body;
   - stage 2: small stress mix;
   - stage 3: evaluate full BUT buckets.
4. Selection remains clean-body + bucketed stress, not a single inflated filtered score.

This keeps the story coherent: clean supervised windows train the classifier; stress windows test and gradually harden robustness.
