# Record111 Physio Teacher Alignment Update - 2026-06-18

## What Changed

We found a training-target conflict in the waveform student runner: record111-like bad augmentations could make the waveform visibly worse while the auxiliary SQI/geometry target still came from the original base PTB row. That can teach the Transformer to map bad-looking augmented waveforms back toward non-bad feature targets.

The runner now supports `aug_bad_aux_pseudo_mode="physio_record111"`. This mode adjusts only waveform-computable targets for augmented bad rows:

- QRS/RR reliability: `qrs_visibility`, `detector_agreement`, `qrs_band_ratio`, `template_corr`, `qrs_prom_p90`
- baseline/contact/flatline: `sqi_basSQI`, `baseline_step`, `flatline_ratio`, `contact_loss_win_ratio`, `fatal_or_score`, `low_amp_ratio`
- detail/stress: `non_qrs_rms_ratio`, `non_qrs_diff_p95`, `band_15_30`, `band_30_45`, `sqi_bSQI`, `sqi_pSQI`

This path intentionally avoids formal training targets such as `knn_label_purity`, `region_confidence`, `boundary_confidence`, `pc2`, or `pc3`. Those remain diagnostic-only.

## Best Result This Round

Best candidate: `featurefirst_wavecomp_record111physio_guard_a050`

Checkpoint:
`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_wavecomp_record111physio_guard_a050\ckpt_best.pt`

| Bucket | Acc | Good R | Medium R | Bad R | Notes |
|---|---:|---:|---:|---:|---|
| synthetic_test | 0.9928 | 0.9937 | 0.9951 | 0.9793 | clean/node useful gate passes |
| original_test_all_10s+ | 0.8601 | 0.8635 | 0.8771 | 0.6472 | best waveform-only bad recovery so far in this branch |
| original_all_10s+ | 0.8692 | 0.8264 | 0.8926 | 0.9601 | strong bad on full original pool, good still shifted |
| bad_core_nearboundary | 0.9076 | - | - | 0.9076 | core bad remains mostly covered |
| bad_outlier_stress | 0.5411 | - | - | 0.5411 | first useful record111 bad-outlier recovery |

## Failed Follow-Ups

| Candidate | Main outcome |
|---|---|
| `featurefirst_wavecomp_record111physio_recall_a050` | More bad pressure reduced original_test to 0.8445 and bad_outlier to 0.4760. |
| `featurefirst_localhead_record111physio_a050` | Local baseline/QRS heads made bad_core perfect but missed bad_outlier; original_test 0.7951, bad_outlier 0.1678. |
| `featurefirst_wavecomp_record111physio_guard_long_a050` | Longer training overfit synthetic stress; original_test fell to 0.8162, bad_outlier to 0.2192 raw. |
| `featurefirst_wavecomp_record111physio_mildwide_*` | Mild/wide stress did not cover original bad_outlier; best original_test around 0.8299 and bad_outlier below 0.25. |

## Feature Recovery Readout

For the best guard model on original rows:

| Feature | Correlation | Interpretation |
|---|---:|---|
| `pc1` | 0.962 | detail/high-frequency geometry is learned, but this is diagnostic-only |
| `non_qrs_diff_p95` | 0.922 | local detail/noise tail is learned |
| `sqi_bSQI` | 0.927 | beat agreement-like information is partly learnable |
| `pca_margin` | 0.864 | diagnostic geometry recovery is strong |
| `qrs_visibility` | 0.463 | barely above target threshold; promising but not robust |
| `flatline_ratio` | 0.172 | weak |
| `detector_agreement` | -0.126 | not learned in the right direction |
| `baseline_step` | -0.226 | not learned in the right direction |
| `sqi_basSQI` | -0.141 | not learned in the right direction |
| `qrs_band_ratio` | -0.404 | actively inverted on original |

## Current Interpretation

The useful mechanism is not more bad class weight. It is target consistency: if the PTB synthetic waveform is changed to look like record111 bad, its teacher targets must also become bad in waveform-computable QRS/baseline/detail dimensions.

The remaining blocker is baseline/contact/domain orientation. The model can detect detail/noise and some QRS visibility, but it does not yet learn the low-frequency baseline and qrs-band axes in a way that transfers to original BUT. More epochs and broader stress both made that worse, suggesting synthetic stress is still too separable from original record111 morphology.

## Next Research Move

Do not continue broad bad-pressure or longer-training sweeps. The next useful experiment should either:

1. add a waveform-derived low-frequency/baseline token bank that directly exposes baseline wander/contact morphology to the Transformer, then keep the physio pseudo-target alignment; or
2. recalibrate the synthetic record111 generator itself by matching the distribution of `sqi_basSQI`, `baseline_step`, `qrs_band_ratio`, and `detector_agreement` against original bad_outlier audit summaries before training.

Selection must remain synthetic/node diagnostic only. Original BUT remains bucketed report-only.
