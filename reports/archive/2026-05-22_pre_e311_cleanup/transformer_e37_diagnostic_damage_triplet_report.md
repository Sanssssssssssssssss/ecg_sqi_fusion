# E3.7 Diagnostic-Damage Triplet Report

## Goal

E3.7 tested whether a single monotonic diagnostic damage axis could replace the more rule-heavy E3.6 critical-damage labels.

The intended simplification was:

- keep the E3.5/E3.6 triplet structure and matched measured-SNR design;
- add `diagnostic_damage_score = max(qrs_axis, 0.85 * tst_axis, crit_axis, corr_axis, 0.75 * beat_axis)`;
- label only clear margin regions: good `<= 0.32`, medium `0.55-0.82`, bad `>= 1.00`;
- keep model structure simple: CLS pooling, raw input, no local heads, no SQI teacher, no noise-type head.

## Dataset

Artifact: `outputs/transformer_e37_diagnostic_damage_triplet`

The full E3.7 generation produced:

| split | good | medium | bad |
|---|---:|---:|---:|
| train | 2328 | 2328 | 2328 |
| val | 453 | 453 | 453 |
| test | 467 | 467 | 467 |

Total: 9744 samples from 3248 triplets.

The data passed the shortcut audits:

| audit | value |
|---|---:|
| measured-SNR oracle accuracy | 0.3320 |
| max measured-SNR class mean gap | 0.0391 dB |
| max proxy pSQI class mean gap | 0.0013 |
| max proxy basSQI class mean gap | 0.0017 |
| global noise score mean, good | 0.3108 |
| global noise score mean, medium | 0.3108 |
| global noise score mean, bad | 0.3093 |

The diagnostic score itself separated classes cleanly by construction:

| class | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| good | 0.2511 | 0.1788 | 0.2621 | 0.3102 |
| medium | 0.6908 | 0.5900 | 0.6854 | 0.7910 |
| bad | 1.2345 | 1.0494 | 1.1491 | 1.5148 |

## SQI Baselines

Artifact: `outputs/transformer_e37_diagnostic_damage_triplet_sqi_ml`

| model | test acc | macro F1 | good recall | medium recall | bad recall |
|---|---:|---:|---:|---:|---:|
| SQI-SVM | 0.4090 | 0.4078 | 0.4261 | 0.4540 | 0.3469 |
| SQI-MLP | 0.3633 | 0.3626 | 0.3319 | 0.3469 | 0.4111 |

Confusion matrices:

```text
SQI-SVM
[[199, 160, 108],
 [148, 212, 107],
 [150, 155, 162]]

SQI-MLP
[[155, 134, 178],
 [133, 162, 172],
 [138, 137, 192]]
```

This is good for benchmark design: E3.7 did not become an SQI-summary shortcut task.

## Transformer Runs

All transformer runs used raw input and `cls_pool=cls`.

| run | method | best val | test acc | good recall | medium recall | bad recall |
|---|---|---:|---:|---:|---:|---:|
| B1 | E3.7 CLS from scratch | 0.7572 | 0.7495 | 0.7473 | 0.7066 | 0.7944 |
| B2 | E3.7 CLS, E3.5 CLS warm-start | 0.7873 | 0.7702 | 0.7730 | 0.7323 | 0.8051 |
| B3 | B2 + rank loss 0.03 / margin 0.10 | 0.7962 | 0.7666 | 0.7923 | 0.7238 | 0.7837 |
| B4 | B2, classification-only | 0.7896 | 0.7702 | 0.7580 | 0.7409 | 0.8116 |

Confusion matrices:

```text
B1: E3.7 CLS from scratch
[[349,  98,  20],
 [ 81, 330,  56],
 [ 13,  83, 371]]

B2: E3.5 warm-start
[[361,  85,  21],
 [ 78, 342,  47],
 [ 12,  79, 376]]

B3: E3.5 warm-start + small rank
[[370,  77,  20],
 [ 89, 338,  40],
 [ 22,  79, 366]]

B4: E3.5 warm-start, classification-only
[[354,  92,  21],
 [ 75, 346,  46],
 [ 12,  76, 379]]
```

## Readout

E3.7 first version did not meet the target. It is far below:

- E3.6 margin-tuned R4: 0.9255 test acc;
- E3.5 CLS-only: 0.9387 test acc;
- target range for this round: 0.94+.

Warm-start helped from 0.7495 to 0.7702, but not nearly enough. Small rank loss did not improve overall test accuracy. Removing denoise and level losses also did not improve test accuracy, so the failure is not mainly due to auxiliary-loss interference.

## Failure Audit

The important audit is the dominant axis inside `diagnostic_damage_score`.

Dominant axis by class:

| class | beat | corr | qrs | tst |
|---|---:|---:|---:|---:|
| good | 3137 | 0 | 6 | 105 |
| medium | 1517 | 1 | 502 | 1228 |
| bad | 195 | 0 | 1114 | 1939 |

The main problem is `medium + noncritical`:

- `medium + noncritical` count: 1343 samples.
- Of these, 1333 are dominated by `beat_axis`.
- These samples have low QRS/T-ST damage but become medium because `max_beat_nprd` is moderate.

That means E3.7 accidentally reintroduced a mixed semantic rule:

- QRS/T-ST/critical damage means diagnostic local damage.
- Beat-axis-only damage can mark noncritical samples as medium.

So the labels are clean on the numeric score axis, but not clean on the clinical/local interpretation axis. The transformer is being asked to separate many samples whose global SQI and measured SNR are matched, whose critical regions may be mostly intact, but whose label changes due to a max single-beat term. This is harder and less aligned with the intended "critical diagnostic damage" story.

## Conclusion

E3.7 as implemented is not the right mainline dataset. The single-axis idea is good for reporting and auditing, but the current max score puts too much authority on `beat_axis`.

Keep these outcomes:

- `diagnostic_damage_score` is useful as an audit column.
- E3.5 warm-start helps a little, but does not fix label semantics.
- Rank loss is not worth keeping for E3.7 v1.
- Classification-only does not rescue the run.

Recommended next patch:

- make the primary single-axis score focus on `qrs_axis`, `tst_axis`, `crit_axis`, and possibly `corr_axis`;
- demote `beat_axis` to an audit/tie-breaker, not a primary medium trigger;
- disallow beat-axis-only `noncritical` medium in the main benchmark;
- keep the same simple CLS transformer and rerun only the data patch first.
