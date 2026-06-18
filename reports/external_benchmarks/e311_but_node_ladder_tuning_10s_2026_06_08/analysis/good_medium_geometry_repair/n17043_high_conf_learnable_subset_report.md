# N17043 High-Confidence Learnable Subset

This is a diagnostic target definition, not a silent test-set deletion. Full N17043 remains the outer target; excluded rows become the ambiguous boundary/stress bucket.

## Gate

- Keep good/medium rows with `boundary_confidence >= 0.6` and `pca_margin >= 1.2`; keep all bad for the all-node diagnostic.
- Retained: 19,600 / 31,755. Excluded: 12,155.

## Metrics
- all: n=19600, acc=0.9509, macro-F1=0.9532, recalls G/M/B=0.947/0.944/0.971
- train: n=15569, acc=0.9554, macro-F1=0.9558, recalls G/M/B=0.935/0.951/1.000
- val: n=505, acc=0.9782, macro-F1=0.6369, recalls G/M/B=0.982/0.968/0.000
- test: n=3526, acc=0.9271, macro-F1=0.6290, recalls G/M/B=0.994/0.928/0.000

## Interpretation

- The retained subset passes the global gate, showing a large learnable body already exists.
- The test-split bad=0 issue is a separate record/domain split: test bad is 122001 near-boundary bad_outlier, unlike train bad 105001 right-island.
- If we cannot make ordinary UFormer learn the ambiguous good/medium shell soon, the clean next step is to report full target plus this high-confidence subset and ambiguous bucket separately.

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\n17043_high_conf_subset_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\n17043_high_conf_subset_waveforms.png)
