# BUT 10s PTB-Trained Seven-SQI Fusion

This experiment freezes the Uformer denoiser/features, trains only PTB classifier heads, and evaluates transfer on formal BUT 10s P1.  Seven SQI features are computed from noisy ECG only.

Calibration for BUT is validation-only.  Test never selects thresholds.

| rank | variant | model | PTB acc | PTB recalls G/M/B | BUT raw acc/bal/macro | BUT raw recalls G/M/B | best val-cal policy | BUT cal acc/bal/macro | BUT cal recalls G/M/B |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 1 | uformer_plus_sqi_branch_ratio4p00 | torch_mlp_head | 0.9886 | 0.988/0.982/0.996 | 0.6514/0.5444/0.5745 | 0.605/0.722/0.307 | acc_primary | 0.6645/0.6278/0.6187 | 0.704/0.644/0.535 |
| 2 | uformer_plus_sqi_gated_delta_1p00 | torch_mlp_head | 0.9877 | 0.985/0.982/0.996 | 0.6598/0.5089/0.5297 | 0.574/0.775/0.178 | acc_primary | 0.6768/0.6449/0.6160 | 0.724/0.648/0.562 |
| 3 | uformer_plus_sqi_gated_delta_0p50 | torch_mlp_head | 0.9891 | 0.990/0.978/0.999 | 0.5313/0.4048/0.4103 | 0.301/0.756/0.158 | acc_primary | 0.6440/0.6219/0.5799 | 0.715/0.594/0.557 |
| 4 | uformer_plus_sqi_branch_ratio2p00 | torch_mlp_head | 0.9877 | 0.989/0.978/0.996 | 0.6040/0.4502/0.4645 | 0.532/0.709/0.109 | acc_primary | 0.6299/0.5753/0.5681 | 0.651/0.630/0.445 |
| 5 | uformer_plus_sqi_gated_delta_0p25 | torch_mlp_head | 0.9868 | 0.990/0.974/0.996 | 0.4935/0.4321/0.4428 | 0.213/0.738/0.345 | acc_primary | 0.5996/0.6079/0.5677 | 0.534/0.650/0.640 |
| 6 | uformer_plus_sqi_branch_ratio0p50 | torch_mlp_head | 0.9873 | 0.990/0.975/0.996 | 0.4935/0.3911/0.4032 | 0.253/0.718/0.202 | acc_primary | 0.6137/0.5867/0.5622 | 0.596/0.636/0.528 |
| 7 | uformer_no_sqi | torch_mlp_head | 0.9891 | 0.989/0.982/0.996 | 0.5469/0.4094/0.4224 | 0.372/0.730/0.127 | acc_primary | 0.6317/0.5437/0.5580 | 0.631/0.659/0.341 |
| 8 | uformer_plus_sqi_branch_ratio1p00 | torch_mlp_head | 0.9873 | 0.990/0.975/0.996 | 0.5803/0.4226/0.4315 | 0.470/0.717/0.080 | acc_primary | 0.6185/0.5395/0.5127 | 0.666/0.604/0.348 |
| 9 | uformer_plus_sqi_concat_logreg | sklearn_logreg_balanced | 0.9896 | 0.990/0.982/0.996 | 0.4507/0.3300/0.3231 | 0.154/0.726/0.109 | acc_primary | 0.5652/0.5010/0.5055 | 0.553/0.595/0.355 |
| 10 | uformer_plus_sqi_concat | torch_mlp_head | 0.9877 | 0.990/0.977/0.996 | 0.5100/0.3734/0.3792 | 0.312/0.711/0.097 | acc_primary | 0.5917/0.4915/0.5015 | 0.578/0.633/0.263 |
| 11 | uformer_plus_sqi_gated_delta_2p00 | torch_mlp_head | 0.9877 | 0.990/0.977/0.996 | 0.5444/0.3935/0.3884 | 0.366/0.734/0.080 | acc_primary | 0.5677/0.5327/0.4760 | 0.491/0.640/0.467 |
| 12 | sqi_only_7feat | torch_mlp_head | 0.7025 | 0.525/0.744/0.839 | 0.3751/0.5225/0.3247 | 0.109/0.544/0.915 | bias_macro | 0.3434/0.3737/0.2793 | 0.571/0.151/0.399 |
| 13 | sqi_only_7feat_logreg | sklearn_logreg_balanced | 0.6567 | 0.663/0.569/0.737 | 0.1509/0.3115/0.1529 | 0.043/0.188/0.703 | bias_macro | 0.2618/0.3621/0.2518 | 0.161/0.312/0.613 |

## Reading

- If concat/gated variants beat `uformer_no_sqi` on BUT without PTB collapse, seven SQI is useful cross-domain quality evidence.
- If `sqi_only_7feat` is strong, BUT quality labels are partly explainable by classical SQI statistics.
- If SQI improves bad but damages medium/good, it remains an analysis branch rather than a mainline head.
