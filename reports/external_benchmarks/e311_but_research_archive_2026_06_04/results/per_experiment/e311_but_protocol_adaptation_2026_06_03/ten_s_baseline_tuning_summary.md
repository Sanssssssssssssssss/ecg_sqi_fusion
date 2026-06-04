# BUT 10s Baseline Tuning

Protocol is fixed to `p1_current_10s_center`.  No 5s crop, ensemble, denoiser training, or synthetic generator changes are used.

## Best By Accuracy

| rank | features | model | calibration | acc | bal acc | macro-F1 | recalls good/medium/bad |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | full_plus_handcrafted | mlp_96 | raw_argmax | 0.8564 | 0.7265 | 0.7616 | 0.960/0.813/0.406 |
| 2 | full_plus_handcrafted | mlp_96 | balanced_primary | 0.8474 | 0.7714 | 0.7559 | 0.873/0.850/0.591 |
| 3 | full_tokens | mlp_128_64 | raw_argmax | 0.8412 | 0.7451 | 0.7733 | 0.893/0.829/0.513 |
| 4 | full_tokens | mlp_128_64 | bias_acc | 0.8392 | 0.7314 | 0.7680 | 0.856/0.859/0.479 |
| 5 | full_tokens | mlp_128_64 | bias_macro | 0.8392 | 0.7314 | 0.7680 | 0.856/0.859/0.479 |
| 6 | full_tokens | mlp_128_64 | bias_bad70_acc | 0.8392 | 0.7314 | 0.7680 | 0.856/0.859/0.479 |
| 7 | handcrafted_sqi | linear_svm_c1 | raw_argmax | 0.8348 | 0.5797 | 0.5696 | 0.789/0.950/0.000 |
| 8 | handcrafted_sqi | linear_svm_c0p5 | raw_argmax | 0.8338 | 0.5789 | 0.5688 | 0.787/0.950/0.000 |
| 9 | handcrafted_sqi | logreg_c0p3 | balanced_primary | 0.8333 | 0.6555 | 0.6572 | 0.869/0.859/0.238 |
| 10 | handcrafted_sqi | logreg_c0p3 | raw_argmax | 0.8318 | 0.5861 | 0.5700 | 0.918/0.838/0.002 |
| 11 | full_plus_handcrafted | logreg_c0p3 | balanced_primary | 0.8288 | 0.7378 | 0.7397 | 0.964/0.748/0.501 |
| 12 | handcrafted_sqi | logreg_c1 | raw_argmax | 0.8274 | 0.6701 | 0.7080 | 0.930/0.793/0.287 |
| 13 | handcrafted_sqi | mlp_96 | raw_argmax | 0.8265 | 0.6673 | 0.7112 | 0.842/0.863/0.297 |
| 14 | full_plus_summary | logreg_c1 | balanced_primary | 0.8254 | 0.7196 | 0.7329 | 0.968/0.744/0.448 |
| 15 | full_plus_handcrafted | mlp_96 | bias_acc | 0.8251 | 0.7218 | 0.7469 | 0.983/0.730/0.453 |
| 16 | full_plus_handcrafted | mlp_96 | bias_macro | 0.8251 | 0.7218 | 0.7469 | 0.983/0.730/0.453 |
| 17 | full_plus_handcrafted | mlp_96 | bias_bad70_acc | 0.8251 | 0.7218 | 0.7469 | 0.983/0.730/0.453 |
| 18 | full_tokens | mlp_128_64 | existing_macro | 0.8245 | 0.8136 | 0.7637 | 0.863/0.797/0.781 |
| 19 | handcrafted_sqi | logreg_c3 | raw_argmax | 0.8238 | 0.6693 | 0.7059 | 0.931/0.785/0.292 |
| 20 | full_tokens | logreg_c1 | balanced_primary | 0.8235 | 0.7207 | 0.7320 | 0.970/0.737/0.455 |

## Best By Balanced Accuracy

| rank | features | model | calibration | acc | bal acc | macro-F1 | recalls good/medium/bad |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | full_tokens | mlp_128_64 | acc_primary | 0.8131 | 0.8186 | 0.7446 | 0.946/0.705/0.805 |
| 2 | full_tokens | mlp_128_64 | bad50_acc | 0.8131 | 0.8186 | 0.7446 | 0.946/0.705/0.805 |
| 3 | full_tokens | mlp_128_64 | bad60_acc | 0.8131 | 0.8186 | 0.7446 | 0.946/0.705/0.805 |
| 4 | full_tokens | mlp_128_64 | bad70_acc | 0.8131 | 0.8186 | 0.7446 | 0.946/0.705/0.805 |
| 5 | full_tokens | mlp_128_64 | macro_primary | 0.8150 | 0.8149 | 0.7470 | 0.862/0.777/0.805 |
| 6 | full_tokens | mlp_128_64 | existing_macro | 0.8245 | 0.8136 | 0.7637 | 0.863/0.797/0.781 |
| 7 | full_tokens | mlp_128_64 | balanced_primary | 0.7664 | 0.8034 | 0.6883 | 0.757/0.762/0.891 |
| 8 | full_plus_summary | mlp_96 | balanced_primary | 0.7756 | 0.7827 | 0.7056 | 0.905/0.669/0.774 |
| 9 | full_plus_summary | extra_trees | balanced_primary | 0.7510 | 0.7804 | 0.7027 | 0.858/0.656/0.827 |
| 10 | full_tokens | extra_trees | balanced_primary | 0.7419 | 0.7790 | 0.6922 | 0.845/0.647/0.844 |
| 11 | full_tokens | extra_trees | existing_macro | 0.7361 | 0.7782 | 0.6866 | 0.895/0.596/0.844 |
| 12 | full_tokens | extra_trees | acc_primary | 0.7361 | 0.7782 | 0.6866 | 0.895/0.596/0.844 |
| 13 | full_tokens | extra_trees | macro_primary | 0.7361 | 0.7782 | 0.6866 | 0.895/0.596/0.844 |
| 14 | full_tokens | extra_trees | bad50_acc | 0.7361 | 0.7782 | 0.6866 | 0.895/0.596/0.844 |
| 15 | full_tokens | extra_trees | bad60_acc | 0.7361 | 0.7782 | 0.6866 | 0.895/0.596/0.844 |
| 16 | full_tokens | extra_trees | bad70_acc | 0.7361 | 0.7782 | 0.6866 | 0.895/0.596/0.844 |
| 17 | full_plus_handcrafted | extra_trees | existing_macro | 0.7385 | 0.7749 | 0.6936 | 0.901/0.597/0.827 |
| 18 | full_plus_handcrafted | extra_trees | acc_primary | 0.6815 | 0.7747 | 0.6092 | 0.866/0.505/0.954 |
| 19 | full_plus_handcrafted | extra_trees | balanced_primary | 0.6815 | 0.7747 | 0.6092 | 0.866/0.505/0.954 |
| 20 | full_plus_handcrafted | extra_trees | macro_primary | 0.6815 | 0.7747 | 0.6092 | 0.866/0.505/0.954 |

## Reading This Baseline

- Accuracy can look high when bad recall is weak because BUT test has far fewer bad windows.
- The recommended 10s baseline should therefore be chosen from the balanced/macro-F1 table, not accuracy alone.
- Any later synthetic adaptation must beat this 10s baseline on acc, macro-F1, balanced acc, and bad recall.
