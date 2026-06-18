# Flatline Bad Veto Cascade

Base expert keeps good/medium; flatline-bad detector can override to bad. Thresholds ranked on synthetic clean+stress validation only.

## Best Original Test Report-Only

| Candidate | Thr | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R |
|---|---:|---:|---:|---:|---:|---:|---:|
| flatveto_mild_extra | 0.780 | 0.808777 | 0.682748 | 0.832418 | 0.810212 | 0.583942 | 0.417808 |
| flatveto_mild_extra | 0.770 | 0.807951 | 0.683339 | 0.832143 | 0.807727 | 0.596107 | 0.434932 |
| flatveto_mild_extra | 0.760 | 0.807361 | 0.683836 | 0.831593 | 0.806146 | 0.605839 | 0.448630 |
| flatveto_mild_extra | 0.750 | 0.807125 | 0.684437 | 0.831593 | 0.805016 | 0.613139 | 0.458904 |
| flatveto_mild_extra | 0.740 | 0.806063 | 0.684092 | 0.831593 | 0.802305 | 0.620438 | 0.469178 |
| flatveto_mild_extra | 0.730 | 0.804766 | 0.683836 | 0.830769 | 0.799593 | 0.630170 | 0.482877 |
| flatveto_mild_extra | 0.720 | 0.803822 | 0.683614 | 0.829670 | 0.798012 | 0.637470 | 0.493151 |
| flatveto_mild_extra | 0.710 | 0.803114 | 0.683618 | 0.828846 | 0.796656 | 0.644769 | 0.503425 |
| flatveto_mild_extra | 0.700 | 0.801935 | 0.683715 | 0.828297 | 0.793719 | 0.656934 | 0.517123 |
| flatveto_mild_extra | 0.690 | 0.800519 | 0.682162 | 0.828297 | 0.791008 | 0.656934 | 0.517123 |
| flatveto_mild_extra | 0.680 | 0.799221 | 0.680723 | 0.827473 | 0.789200 | 0.656934 | 0.517123 |
| flatveto_mild_extra | 0.660 | 0.795918 | 0.679148 | 0.824725 | 0.783326 | 0.676399 | 0.544521 |
| flatveto_balanced_extra | 0.740 | 0.795564 | 0.674867 | 0.820055 | 0.789200 | 0.647202 | 0.506849 |
| flatveto_balanced_extra | 0.730 | 0.793913 | 0.673938 | 0.819780 | 0.785585 | 0.654501 | 0.517123 |
| flatveto_strong_extra | 0.760 | 0.793323 | 0.675073 | 0.814835 | 0.786489 | 0.676399 | 0.547945 |
| flatveto_balanced_extra | 0.720 | 0.792969 | 0.674505 | 0.818407 | 0.783552 | 0.669100 | 0.537671 |
| flatveto_balanced_extra | 0.710 | 0.792379 | 0.675092 | 0.818132 | 0.781518 | 0.681265 | 0.554795 |
| flatveto_strong_extra | 0.750 | 0.792025 | 0.674582 | 0.814835 | 0.783326 | 0.683698 | 0.558219 |
| flatveto_strong_extra | 0.740 | 0.791318 | 0.674563 | 0.814560 | 0.781518 | 0.690998 | 0.568493 |
| flatveto_balanced_extra | 0.700 | 0.791082 | 0.674014 | 0.817308 | 0.779485 | 0.683698 | 0.558219 |
| flatveto_balanced_histgb | 0.970 | 0.790374 | 0.673170 | 0.839835 | 0.761636 | 0.661800 | 0.530822 |
| flatveto_balanced_histgb | 0.940 | 0.790138 | 0.673735 | 0.839835 | 0.760506 | 0.669100 | 0.541096 |
| flatveto_balanced_histgb | 0.960 | 0.790138 | 0.672934 | 0.839835 | 0.761184 | 0.661800 | 0.530822 |
| flatveto_strong_extra | 0.730 | 0.790020 | 0.673820 | 0.814011 | 0.779033 | 0.695864 | 0.575342 |
| flatveto_balanced_histgb | 0.950 | 0.789902 | 0.672698 | 0.839835 | 0.760732 | 0.661800 | 0.530822 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\flatbad_veto_cascade_metrics.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\flatbad_veto_cascade_summary.json`
