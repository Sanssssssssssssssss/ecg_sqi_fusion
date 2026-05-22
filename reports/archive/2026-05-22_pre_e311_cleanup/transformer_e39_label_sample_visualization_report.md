# E3.9 D1 Label Sample Visualization

This visualization checks the current transformer-line main dataset:

```text
outputs/transformer_e39a_smooth_morph_triplet
```

The active label source is `e39a_smooth_morph_margin`.

## Label Rule

```text
smooth_morph_score =
    0.45*qrs_nprd
  + 0.25*tst_nprd
  + 0.20*(1 - beat_corr)
  + 0.10*max_beat_nprd

good:
  smooth_morph_score <= 0.10
  qrs_nprd <= 0.10
  beat_corr >= 0.95

medium:
  0.27 <= smooth_morph_score <= 0.40
  qrs_nprd < 0.35
  beat_corr >= 0.80

bad:
  smooth_morph_score >= 0.58
  or qrs_nprd >= 0.45
  or beat_corr <= 0.70

gray:
  otherwise
```

## Outputs

Figures were written under the dataset artifact so the output directory stays
local and contained:

```text
outputs/transformer_e39a_smooth_morph_triplet/figs_label_samples
```

Generated files:

```text
e39_d1_counterfactual_triplets.png
e39_d1_counterfactual_triplets.pdf
e39_d1_label_representative_boundary_samples.png
e39_d1_label_representative_boundary_samples.pdf
e39_d1_label_metric_distributions.png
e39_d1_label_metric_distributions.pdf
selected_label_samples.csv
label_sample_visualization_summary.json
```

## What To Look At

The most useful figure is:

```text
outputs/transformer_e39a_smooth_morph_triplet/figs_label_samples/e39_d1_counterfactual_triplets.png
```

It shows six held-out test counterfactual groups. Each row keeps the same clean
morphology/noise source and compares:

```text
good -> medium -> bad
```

The panels overlay:

- solid black: noisy ECG
- dashed gray: clean ECG
- red: residual noise
- blue shade: QRS mask
- orange shade: T-ST mask
- red shade: injected noise placement

The second figure:

```text
outputs/transformer_e39a_smooth_morph_triplet/figs_label_samples/e39_d1_label_representative_boundary_samples.png
```

shows representative and boundary examples:

- low-score good
- near-threshold good
- noncritical-hard good
- low / center / high medium
- QRS bad
- score bad
- low-score bad trigger

The third figure:

```text
outputs/transformer_e39a_smooth_morph_triplet/figs_label_samples/e39_d1_label_metric_distributions.png
```

summarizes the test-set metric distributions by class.

## Selected Triplet Groups

The counterfactual figure uses test groups:

```text
4800, 4802, 4805, 4806, 4808, 4809
```

All selected rows and their metrics are saved in:

```text
outputs/transformer_e39a_smooth_morph_triplet/figs_label_samples/selected_label_samples.csv
```

## Readout

The current rule is visually coherent in the counterfactual triplets:

- good samples are mostly noncritical placements with low QRS damage and low
  smooth score;
- medium samples are usually uniform or T-ST overlap cases where the residual
  visibly touches diagnostically relevant morphology without reaching the bad
  QRS trigger;
- bad samples are mostly QRS-overlap failures, and can have smooth scores that
  overlap medium because `qrs_nprd >= 0.45` is an explicit severe-failure rule.

The metric-distribution figure explains why D1 works better than the E3.7/E3.8
max-score line: global noise and measured SNR remain similar across classes,
while QRS/T-ST damage and the smooth morphology score provide the class
structure.
