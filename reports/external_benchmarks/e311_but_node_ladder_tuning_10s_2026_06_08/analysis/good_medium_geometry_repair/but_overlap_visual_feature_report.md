# BUT Good/Medium Overlap Visual Feature Deep Dive

This report only analyzes the existing Clean/SemiClean/node diagnostic overlap rows. It does not use original BUT for model selection.

## Error Counts

- waveform groups: `{'best_correct_good_boundary': 6906, 'best_correct_medium_boundary': 5714, 'best_medium_to_good': 335, 'best_good_to_medium': 204}`

Current best N7110 confusion on this overlap table:

```
best_pred_class  good  medium
class_name                   
good             6906     204
medium            335    5714
```

## Visual Interpretation

The visually obvious split is real but local:

- good rows eaten by medium usually still have visible, regular QRS morphology and lower non-QRS/detail energy.
- medium rows eaten by good usually have weaker QRS/template agreement, PC3-high geometry, and stronger non-QRS derivative/high-frequency detail.
- this explains why broad qrs-low or one-sided medium sweeps failed: they move the whole decision surface instead of only protecting the local medium-hard-negative island.

## Top Separating Features

- `pc1`: KS 0.991; good->medium median -3.657, medium->good median -0.6974
- `pc3`: KS 0.959; good->medium median -1.808, medium->good median 2.97
- `qrs_visibility`: KS 0.926; good->medium median 0.5978, medium->good median 0.238
- `flatline_ratio`: KS 0.902; good->medium median 0.2826, medium->good median 0.09367
- `template_corr`: KS 0.883; good->medium median 0.7226, medium->good median 0.5457
- `non_qrs_diff_p95`: KS 0.792; good->medium median 0.04405, medium->good median 0.1119
- `raw_non_qrs_diff_p95`: KS 0.777; good->medium median 0.332, medium->good median 0.9606
- `band_30_45`: KS 0.730; good->medium median 0.01574, medium->good median 0.03003
- `amplitude_entropy`: KS 0.700; good->medium median 0.7222, medium->good median 0.6353
- `raw_slow_drift_ratio`: KS 0.625; good->medium median 0.2319, medium->good median 0.0859

## Rule Candidates From Train+Val Thresholds

The rules below are diagnostic candidates. They should guide generator/rule-mode design, not become hidden selection on original BUT.


### good_rescue

- `pc1`: fixed 192, lost 7, precision 0.965, current-error recall 0.941
- `qrs_visibility >= 0.432896 AND non_qrs_diff_p95 <= 0.0589044`: fixed 169, lost 7, precision 0.960, current-error recall 0.828
- `qrs_visibility`: fixed 180, lost 16, precision 0.918, current-error recall 0.882
- `qrs_visibility >= 0.395938 AND non_qrs_diff_p95 <= 0.0589044`: fixed 172, lost 16, precision 0.915, current-error recall 0.843
- `qrs_visibility >= 0.432896 AND flatline_ratio >= 0.135308`: fixed 201, lost 22, precision 0.901, current-error recall 0.985
- `qrs_visibility >= 0.432896 AND raw_non_qrs_diff_p95 <= 0.423036`: fixed 155, lost 14, precision 0.917, current-error recall 0.760
- `pc3 <= -0.0623558 AND flatline_ratio >= 0.135308`: fixed 175, lost 20, precision 0.897, current-error recall 0.858
- `qrs_visibility >= 0.432896 AND non_qrs_diff_p95 <= 0.0717702`: fixed 189, lost 25, precision 0.883, current-error recall 0.926

### medium_rescue

- `qrs_visibility <= 0.338125 AND non_qrs_diff_p95 >= 0.0699296`: fixed 168, lost 2, precision 0.988, current-error recall 0.501
- `qrs_visibility <= 0.338125 AND flatline_ratio <= 0.184948`: fixed 199, lost 4, precision 0.980, current-error recall 0.594
- `qrs_visibility <= 0.338125 AND flatline_ratio <= 0.213771`: fixed 206, lost 7, precision 0.967, current-error recall 0.615
- `qrs_visibility <= 0.338125 AND non_qrs_diff_p95 >= 0.0655117`: fixed 174, lost 6, precision 0.967, current-error recall 0.519
- `pc1 >= -3.13881 AND qrs_visibility <= 0.338125`: fixed 216, lost 8, precision 0.964, current-error recall 0.645
- `qrs_visibility <= 0.338125 AND flatline_ratio <= 0.244996`: fixed 211, lost 9, precision 0.959, current-error recall 0.630
- `pc1 >= -3.44731 AND qrs_visibility <= 0.338125`: fixed 216, lost 16, precision 0.931, current-error recall 0.645
- `raw_non_qrs_diff_p95 >= 0.918193 AND amplitude_entropy >= 0.631451`: fixed 72, lost 8, precision 0.900, current-error recall 0.215

## Two-Sided Visual Gate Simulation

This local simulation combines the strongest interpretable train+val rules:

- good rescue: predicted-medium rows with `qrs_visibility >= 0.432896` and `non_qrs_diff_p95 <= 0.0589044`.
- medium rescue: predicted-good rows with `qrs_visibility <= 0.338125` and `non_qrs_diff_p95 >= 0.0699296`.

It is still a diagnostic rule probe, not a promoted checkpoint.

```
    split        prediction     n      acc  good_to_medium  medium_to_good  good_recall  medium_recall
train_val   best_pred_class  9597 0.955299             199             230     0.965271       0.940522
train_val  visual_gate_pred  9597 0.982599              37             130     0.993543       0.966382
train_val visual_gate_delta  9597 0.027300            -162            -100     0.028272       0.025860
     test   best_pred_class  3562 0.969118               5             105     0.996377       0.951879
     test  visual_gate_pred  3562 0.987647               0              44     1.000000       0.979835
     test visual_gate_delta  3562 0.018529              -5             -61     0.003623       0.027956
      all   best_pred_class 13159 0.959039             204             335     0.971308       0.944619
      all  visual_gate_pred 13159 0.983965              37             174     0.994796       0.971235
      all visual_gate_delta 13159 0.024926            -167            -161     0.023488       0.026616
```

## Generated Figures

- `but_overlap_visual_feature_boxplots.png`
- `but_overlap_visual_rule_scatter.png`
- `but_overlap_visual_error_waveforms_ranked.png`

## Next Experimental Move

The next promising direction is a two-sided local boundary treatment:

1. good rescue: only rescue predicted-medium rows that are PC3-low / QRS-visible / template-consistent / low non-QRS-detail.
2. medium protection: only protect predicted-good rows that are PC3-high or QRS/template-weak with elevated non-QRS detail.
3. avoid broad synthetic row additions. The earlier qrs-low single-checkpoint conversion failed because it changed class balance globally.
