# E3.11f BUT Protocol Sweep

This sweep is protocol-first: no Uformer denoiser training is performed here.  Calibration uses validation split only.

| rank | protocol | mode | feature/model | acc | bal acc | macro-F1 | recalls good/medium/bad |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | p1_current_10s_center | probe | handcrafted_sqi/linear_svm | 0.8044 | 0.5845 | 0.5808 | 0.899/0.796/0.058 |
| 2 | p2_10s_purity90 | probe | handcrafted_sqi/linear_svm | 0.8044 | 0.5845 | 0.5808 | 0.899/0.796/0.058 |
| 3 | p6_two_5s_crops_ensemble | probe | full_tokens/small_mlp | 0.8041 | 0.7569 | 0.7468 | 0.946/0.704/0.620 |
| 4 | p5_5s_stride2p5_pad10 | probe | full_tokens/small_mlp | 0.7937 | 0.7409 | 0.7464 | 0.936/0.696/0.591 |
| 5 | p1_current_10s_center | probe | handcrafted_sqi/small_mlp | 0.7839 | 0.6956 | 0.6980 | 0.922/0.700/0.465 |
| 6 | p2_10s_purity90 | probe | handcrafted_sqi/small_mlp | 0.7839 | 0.6956 | 0.6980 | 0.922/0.700/0.465 |
| 7 | p1_current_10s_center | probe | full_tokens/small_mlp | 0.7768 | 0.7409 | 0.6811 | 0.920/0.673/0.630 |
| 8 | p2_10s_purity90 | probe | full_tokens/small_mlp | 0.7768 | 0.7409 | 0.6811 | 0.920/0.673/0.630 |
| 9 | p6_two_5s_crops_ensemble | probe | full_tokens/logreg | 0.7737 | 0.7459 | 0.7497 | 0.987/0.611/0.640 |
| 10 | p6_two_5s_crops_ensemble | probe | handcrafted_sqi/linear_svm | 0.7722 | 0.6106 | 0.5940 | 0.930/0.695/0.207 |
| 11 | p6_two_5s_crops_ensemble | probe | full_tokens/linear_svm | 0.7648 | 0.7450 | 0.7347 | 0.981/0.597/0.657 |
| 12 | p1_current_10s_center | probe | bottleneck_only/logreg | 0.7577 | 0.6783 | 0.6403 | 0.949/0.628/0.457 |
| 13 | p2_10s_purity90 | probe | bottleneck_only/logreg | 0.7577 | 0.6783 | 0.6403 | 0.949/0.628/0.457 |
| 14 | p1_current_10s_center | probe | full_tokens/logreg | 0.7478 | 0.6740 | 0.6761 | 0.997/0.570/0.455 |
| 15 | p2_10s_purity90 | probe | full_tokens/logreg | 0.7478 | 0.6740 | 0.6761 | 0.997/0.570/0.455 |
| 16 | p1_current_10s_center | probe | bottleneck_only/small_mlp | 0.7477 | 0.6866 | 0.6455 | 0.950/0.604/0.506 |
| 17 | p2_10s_purity90 | probe | bottleneck_only/small_mlp | 0.7477 | 0.6866 | 0.6455 | 0.950/0.604/0.506 |
| 18 | p6_two_5s_crops_ensemble | probe | handcrafted_sqi/small_mlp | 0.7406 | 0.6885 | 0.6512 | 0.921/0.611/0.533 |
| 19 | p5_5s_stride2p5_pad10 | probe | bottleneck_only/small_mlp | 0.7391 | 0.6714 | 0.6519 | 0.943/0.596/0.475 |
| 20 | p1_current_10s_center | probe | bottleneck_only/linear_svm | 0.7372 | 0.6789 | 0.6196 | 0.896/0.627/0.513 |
| 21 | p2_10s_purity90 | probe | bottleneck_only/linear_svm | 0.7372 | 0.6789 | 0.6196 | 0.896/0.627/0.513 |
| 22 | p3_5s_center_pad10 | probe | handcrafted_sqi/small_mlp | 0.7358 | 0.6910 | 0.6365 | 0.904/0.614/0.555 |
| 23 | p4_5s_purity90_pad10 | probe | handcrafted_sqi/small_mlp | 0.7358 | 0.6910 | 0.6365 | 0.904/0.614/0.555 |
| 24 | p5_5s_stride2p5_pad10 | probe | full_tokens/logreg | 0.7354 | 0.6992 | 0.7125 | 0.993/0.539/0.565 |
| 25 | p3_5s_center_pad10 | probe | full_tokens/small_mlp | 0.7281 | 0.7085 | 0.6805 | 0.946/0.559/0.620 |
| 26 | p4_5s_purity90_pad10 | probe | full_tokens/small_mlp | 0.7281 | 0.7085 | 0.6805 | 0.946/0.559/0.620 |
| 27 | p3_5s_center_pad10 | probe | bottleneck_only/logreg | 0.7264 | 0.6365 | 0.6358 | 0.966/0.561/0.382 |
| 28 | p4_5s_purity90_pad10 | probe | bottleneck_only/logreg | 0.7264 | 0.6365 | 0.6358 | 0.966/0.561/0.382 |
| 29 | p5_5s_stride2p5_pad10 | probe | handcrafted_sqi/linear_svm | 0.7199 | 0.5705 | 0.5504 | 0.958/0.574/0.179 |
| 30 | p3_5s_center_pad10 | probe | bottleneck_only/small_mlp | 0.7190 | 0.6486 | 0.6137 | 0.928/0.572/0.445 |

## Best By Protocol

| protocol | best feature/model | acc | bal acc | macro-F1 | bad recall |
| --- | --- | ---: | ---: | ---: | ---: |
| p1_current_10s_center | handcrafted_sqi/linear_svm | 0.8044 | 0.5845 | 0.5808 | 0.0584 |
| p2_10s_purity90 | handcrafted_sqi/linear_svm | 0.8044 | 0.5845 | 0.5808 | 0.0584 |
| p3_5s_center_pad10 | handcrafted_sqi/small_mlp | 0.7358 | 0.6910 | 0.6365 | 0.5547 |
| p4_5s_purity90_pad10 | handcrafted_sqi/small_mlp | 0.7358 | 0.6910 | 0.6365 | 0.5547 |
| p5_5s_stride2p5_pad10 | full_tokens/small_mlp | 0.7937 | 0.7409 | 0.7464 | 0.5912 |
| p6_two_5s_crops_ensemble | full_tokens/small_mlp | 0.8041 | 0.7569 | 0.7468 | 0.6204 |
