# N7110 QRS-Low Rule-Mode Artifact

This is an explicit experiment-only rule-mode diagnostic. It is separate from ordinary checkpoint promotion.

## Rule

- Base endpoint: `nl_n7110_gm_trim_bad_geom_directrule_n7100base_g003_m008__69ab5b71cf7d` / `raw`
- Medium endpoint: `nl_n7110_gm_trim_bad_geom_directrule_n7100base_g004_m010__59b96f510b3e` / `medium_guarded_pmed0005`
- Gate: when endpoints disagree good-vs-medium, choose the medium endpoint if `qrs_visibility <= 0.03784960`.
- Threshold source: train+val endpoint disagreements only.

## Node Diagnostic

- Rule-mode acc: `0.952906`
- Macro-F1: `0.957450`
- Good/medium/bad recall: `0.960338` / `0.935302` / `0.970617`
- Delta vs base: acc `+0.003278`, medium recall `+0.008439`, good recall `+0.000000`
- Gate flips: `66`; fixed medium errors: `63`; good lost: `0`
- Decision: `rule_mode_passed_gates`

## Raw Waveforms

![N7110 qrs-low gate waveforms](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/n7110_qrs_visibility_gate_waveforms.png)

## Caveat

This artifact can be used as a transparent rule-engine candidate or as a target for the next single-checkpoint conversion. It is not a standalone neural checkpoint and original BUT remains report-only.
