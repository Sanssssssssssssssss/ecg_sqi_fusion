# Final Visuals and CINC2017 Probe

## Generated figures

- Synthetic examples: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\final_visuals_cinc_probe\figure_synthetic_examples_10_per_label.png`
- Training curves: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\final_visuals_cinc_probe\figure_training_curves_final_uformer_geometry.png`
- Synthetic vs BUT PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\final_visuals_cinc_probe\figure_synthetic_vs_but_pca_geometry.png`
- Synthetic vs BUT class mix: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\final_visuals_cinc_probe\figure_synthetic_vs_but_class_mix.png`
- CINC2017 quality mix: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\final_visuals_cinc_probe\figure_cinc2017_quality_prediction_mix.png`
- CINC2017 examples: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\final_visuals_cinc_probe\figure_cinc2017_examples_by_predicted_quality.png`

## CINC2017 note

CINC2017 labels are rhythm labels (N/A/O/~), not SQI good/medium/bad labels. The CINC probe is therefore report-only and should be interpreted as predicted quality distribution plus a weak noisy-label sanity check for `~` records.

## CINC2017 summary by rhythm label

rhythm_label    n  pred_good_share  mean_p_good  pred_medium_share  mean_p_medium  pred_bad_share   mean_p_bad
           A  758              1.0     0.998725                0.0       0.001274             0.0 1.134651e-06
           N 5076              1.0     0.999300                0.0       0.000699             0.0 6.022989e-07
           O 2415              1.0     0.999130                0.0       0.000869             0.0 7.489884e-07
           ~  279              1.0     0.999725                0.0       0.000274             0.0 2.423917e-07
