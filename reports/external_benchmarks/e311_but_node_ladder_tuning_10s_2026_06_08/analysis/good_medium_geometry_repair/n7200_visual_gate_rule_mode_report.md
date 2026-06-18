# N7200 Visual QRS/Detail Rule-Mode Artifact

This is an explicit experiment-only rule-mode diagnostic. It is separate from ordinary checkpoint promotion.

## Rule

- Base endpoint column: `old_best_pred`
- Good rescue: predicted medium + `qrs_visibility >= 0.432896` + `non_qrs_diff_p95 <= 0.0589044` -> good
- Medium rescue: predicted good + `qrs_visibility <= 0.338125` + `non_qrs_diff_p95 >= 0.0699296` -> medium
- Threshold source: N7110 train+val overlap visual errors; original BUT not used.

## Node Diagnostic

- Rule-mode acc: `0.968297`
- Macro-F1: `0.970768`
- Good/medium/bad recall: `0.975694` / `0.959583` / `0.970617`
- Delta vs base: acc `+0.031865`, good `+0.044306`, medium `+0.037500`, bad `+0.000000`
- Gate flips: `611`; good rescue `330`; medium rescue `281`
- Decision: `rule_mode_passed_gates`

## Caveat

This artifact can guide a transparent rule engine or a later logit-level adapter. Direct synthetic-row retraining with the same idea did not promote, so do not treat this as a normal checkpoint.
