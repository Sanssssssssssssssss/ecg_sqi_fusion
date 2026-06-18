# BUT 10s PTB-Trained Seven-SQI Fusion

This experiment freezes the Uformer denoiser/features, trains only PTB classifier heads, and evaluates transfer on formal BUT 10s P1.  Seven SQI features are computed from noisy ECG only.

Calibration for BUT is validation-only.  Test never selects thresholds.

| rank | variant | model | PTB acc | PTB recalls G/M/B | BUT raw acc/bal/macro | BUT raw recalls G/M/B | best val-cal policy | BUT cal acc/bal/macro | BUT cal recalls G/M/B |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 1 | uformer_plus_sqi_gated_delta_0p25 | torch_mlp_head | 0.9868 | 0.985/0.981/0.995 | 0.5612/0.4665/0.4894 | 0.349/0.761/0.290 | acc_primary | 0.6588/0.6297/0.6230 | 0.648/0.676/0.564 |
| 2 | uformer_plus_sqi_gated_delta_0p10 | torch_mlp_head | 0.9864 | 0.989/0.974/0.996 | 0.5809/0.4367/0.4533 | 0.454/0.727/0.129 | acc_primary | 0.6463/0.5823/0.5627 | 0.690/0.631/0.426 |
| 3 | uformer_plus_sqi_concat | torch_mlp_head | 0.9868 | 0.989/0.975/0.996 | 0.5359/0.4108/0.4249 | 0.327/0.742/0.163 | acc_primary | 0.6337/0.6008/0.5616 | 0.668/0.617/0.518 |
| 4 | uformer_no_sqi | torch_mlp_head | 0.9873 | 0.989/0.977/0.996 | 0.5587/0.4271/0.4453 | 0.394/0.732/0.156 | macro_primary | 0.5885/0.5963/0.5124 | 0.556/0.612/0.620 |
| 5 | uformer_plus_sqi_concat_logreg | sklearn_logreg_balanced | 0.9896 | 0.990/0.982/0.996 | 0.4507/0.3300/0.3231 | 0.154/0.726/0.109 | acc_primary | 0.5652/0.5010/0.5055 | 0.553/0.595/0.355 |
| 6 | sqi_only_7feat | torch_mlp_head | 0.6971 | 0.525/0.723/0.843 | 0.2775/0.4439/0.2572 | 0.069/0.394/0.869 | bias_macro | 0.3161/0.3280/0.2521 | 0.540/0.132/0.311 |
| 7 | sqi_only_7feat_logreg | sklearn_logreg_balanced | 0.6567 | 0.663/0.569/0.737 | 0.1509/0.3115/0.1529 | 0.043/0.188/0.703 | bias_macro | 0.2618/0.3621/0.2518 | 0.161/0.312/0.613 |

## Reading

- If concat/gated variants beat `uformer_no_sqi` on BUT without PTB collapse, seven SQI is useful cross-domain quality evidence.
- If `sqi_only_7feat` is strong, BUT quality labels are partly explainable by classical SQI statistics.
- If SQI improves bad but damages medium/good, it remains an analysis branch rather than a mainline head.
