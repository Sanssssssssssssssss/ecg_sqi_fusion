# Waveform Bad Score Frontier

Report-only diagnostic. Original BUT is not used for training or checkpoint selection.

## Best Overall Thresholds

| candidate | threshold | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_to_bad | medium_to_bad | bad_to_good | bad_to_medium | bad_core_recall | bad_outlier_recall | nonbad_false_bad_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primitive_badexpert_multiscale | 0.42 | 0.815383 | 0.69725 | 0.891209 | 0.801175 | 0.296837 | 0 | 0 | 0 | 92 | 1 | 0.010274 | 0.00334738 |
| primitive_badexpert_multiscale | 0.43 | 0.815383 | 0.69725 | 0.891209 | 0.801175 | 0.296837 | 0 | 0 | 0 | 92 | 1 | 0.010274 | 0.00334738 |
| primitive_badexpert_multiscale | 0.44 | 0.815383 | 0.69725 | 0.891209 | 0.801175 | 0.296837 | 0 | 0 | 0 | 92 | 1 | 0.010274 | 0.00334738 |
| primitive_badexpert_multiscale | 0.45 | 0.815383 | 0.69725 | 0.891209 | 0.801175 | 0.296837 | 0 | 0 | 0 | 92 | 1 | 0.010274 | 0.00334738 |
| primitive_badexpert_multiscale | 0.46 | 0.815383 | 0.69725 | 0.891209 | 0.801175 | 0.296837 | 0 | 0 | 0 | 92 | 1 | 0.010274 | 0.00334738 |
| qrsbank_highamp_mix_patch_guard | 0.28 | 0.815029 | 0.695604 | 0.862912 | 0.823317 | 0.301703 | 0 | 0 | 0 | 108 | 1 | 0.0171233 | 0.00495909 |
| qrsbank_highamp_mix_patch_guard | 0.29 | 0.815029 | 0.695167 | 0.862912 | 0.823543 | 0.29927 | 0 | 0 | 0 | 108 | 1 | 0.0136986 | 0.00471113 |
| qrsbank_highamp_mix_patch_guard | 0.3 | 0.815029 | 0.695383 | 0.862912 | 0.823543 | 0.29927 | 0 | 0 | 0 | 108 | 1 | 0.0136986 | 0.00458716 |
| qrsbank_highamp_mix_patch_guard | 0.37 | 0.815029 | 0.696712 | 0.862912 | 0.823769 | 0.296837 | 0 | 0 | 0 | 109 | 0.991597 | 0.0136986 | 0.00334738 |
| qrsbank_highamp_mix_patch_guard | 0.38 | 0.815029 | 0.696936 | 0.862912 | 0.823769 | 0.296837 | 0 | 0 | 0 | 109 | 0.991597 | 0.0136986 | 0.00322341 |
| qrsbank_badexpert_balanced | 0.15 | 0.803586 | 0.688246 | 0.883516 | 0.784907 | 0.296837 | 0 | 0 | 0 | 58 | 0.991597 | 0.0136986 | 0.00384329 |
| qrsbank_badexpert_balanced | 0.16 | 0.803586 | 0.687586 | 0.883516 | 0.785133 | 0.294404 | 0 | 0 | 0 | 59 | 0.983193 | 0.0136986 | 0.00371932 |
| qrsbank_badexpert_balanced | 0.13 | 0.803468 | 0.687066 | 0.883516 | 0.784681 | 0.296837 | 0 | 0 | 0 | 58 | 0.991597 | 0.0136986 | 0.00446318 |
| qrsbank_badexpert_balanced | 0.14 | 0.803468 | 0.687503 | 0.883516 | 0.784681 | 0.296837 | 0 | 0 | 0 | 58 | 0.991597 | 0.0136986 | 0.00421522 |
| qrsbank_badexpert_balanced | 0.17 | 0.803468 | 0.686842 | 0.883516 | 0.785133 | 0.291971 | 0 | 0 | 0 | 60 | 0.97479 | 0.0136986 | 0.00359534 |
| gated_bad_multiscale_stablemix | 0.46 | 0.80335 | 0.686426 | 0.865934 | 0.798238 | 0.304136 | 0 | 0 | 0 | 91 | 1 | 0.0205479 | 0.00619886 |
| gated_bad_multiscale_stablemix | 0.47 | 0.80335 | 0.686426 | 0.865934 | 0.798238 | 0.304136 | 0 | 0 | 0 | 91 | 1 | 0.0205479 | 0.00619886 |
| gated_bad_multiscale_stablemix | 0.48 | 0.80335 | 0.686426 | 0.865934 | 0.798238 | 0.304136 | 0 | 0 | 0 | 91 | 1 | 0.0205479 | 0.00619886 |
| gated_bad_multiscale_stablemix | 0.49 | 0.80335 | 0.686426 | 0.865934 | 0.798238 | 0.304136 | 0 | 0 | 0 | 91 | 1 | 0.0205479 | 0.00619886 |
| gated_bad_multiscale_stablemix | 0.5 | 0.80335 | 0.686426 | 0.865934 | 0.798238 | 0.304136 | 0 | 0 | 0 | 91 | 1 | 0.0205479 | 0.00619886 |
| qrsbank_badstress_hardneg_balanced | 0.08 | 0.794503 | 0.684294 | 0.880769 | 0.769544 | 0.29927 | 0 | 0 | 0 | 71 | 1 | 0.0136986 | 0.00309943 |
| qrsbank_badstress_hardneg_balanced | 0.09 | 0.794503 | 0.685688 | 0.880769 | 0.769544 | 0.29927 | 0 | 0 | 0 | 71 | 1 | 0.0136986 | 0.00235557 |
| qrsbank_badstress_hardneg_balanced | 0.12 | 0.794503 | 0.685558 | 0.881044 | 0.76977 | 0.294404 | 0 | 0 | 0 | 71 | 1 | 0.00684932 | 0.00148773 |
| qrsbank_badstress_hardneg_balanced | 0.13 | 0.794503 | 0.685132 | 0.881319 | 0.76977 | 0.291971 | 0 | 0 | 0 | 71 | 1 | 0.00342466 | 0.00123977 |
| qrsbank_badstress_hardneg_balanced | 0.14 | 0.794503 | 0.685132 | 0.881319 | 0.76977 | 0.291971 | 0 | 0 | 0 | 71 | 1 | 0.00342466 | 0.00123977 |
| qrsbank_badexpert_badguard | 0.29 | 0.791908 | 0.678026 | 0.908242 | 0.743109 | 0.287105 | 0 | 0 | 0 | 47 | 0.957983 | 0.0136986 | 0.00322341 |
| qrsbank_badexpert_badguard | 0.35 | 0.791908 | 0.677338 | 0.908516 | 0.743335 | 0.282238 | 0 | 0 | 0 | 49 | 0.941176 | 0.0136986 | 0.00260352 |
| qrsbank_badexpert_badguard | 0.36 | 0.791908 | 0.677338 | 0.908516 | 0.743335 | 0.282238 | 0 | 0 | 0 | 49 | 0.941176 | 0.0136986 | 0.00260352 |
| qrsbank_badexpert_badguard | 0.37 | 0.791908 | 0.677338 | 0.908516 | 0.743335 | 0.282238 | 0 | 0 | 0 | 49 | 0.941176 | 0.0136986 | 0.00260352 |
| qrsbank_badexpert_badguard | 0.26 | 0.79179 | 0.676621 | 0.908242 | 0.742883 | 0.287105 | 0 | 0 | 0 | 47 | 0.957983 | 0.0136986 | 0.00396727 |
| qrsbank_stattoken_badguard | 0.43 | 0.771617 | 0.667513 | 0.887637 | 0.720967 | 0.289538 | 0 | 0 | 0 | 26 | 1 | 0 | 0.00185966 |
| qrsbank_stattoken_badguard | 0.44 | 0.771617 | 0.667513 | 0.887637 | 0.720967 | 0.289538 | 0 | 0 | 0 | 26 | 1 | 0 | 0.00185966 |
| qrsbank_stattoken_badguard | 0.45 | 0.771617 | 0.667513 | 0.887637 | 0.720967 | 0.289538 | 0 | 0 | 0 | 26 | 1 | 0 | 0.00185966 |
| qrsbank_stattoken_badguard | 0.46 | 0.771617 | 0.667513 | 0.887637 | 0.720967 | 0.289538 | 0 | 0 | 0 | 26 | 1 | 0 | 0.00185966 |
| qrsbank_stattoken_badguard | 0.47 | 0.771617 | 0.667513 | 0.887637 | 0.720967 | 0.289538 | 0 | 0 | 0 | 26 | 1 | 0 | 0.00185966 |
| qrsbank_highamp_stattoken_badguard | 0.26 | 0.765129 | 0.645789 | 0.88956 | 0.703344 | 0.328467 | 0 | 0 | 0 | 20 | 1 | 0.0547945 | 0.0223159 |
| qrsbank_highamp_stattoken_badguard | 0.27 | 0.765129 | 0.646209 | 0.88956 | 0.703344 | 0.328467 | 0 | 0 | 0 | 20 | 1 | 0.0547945 | 0.021944 |
| qrsbank_highamp_stattoken_badguard | 0.28 | 0.765129 | 0.645929 | 0.889835 | 0.703344 | 0.326034 | 0 | 0 | 0 | 20 | 1 | 0.0513699 | 0.021572 |
| qrsbank_highamp_stattoken_badguard | 0.29 | 0.765129 | 0.646071 | 0.889835 | 0.703344 | 0.326034 | 0 | 0 | 0 | 20 | 1 | 0.0513699 | 0.0214481 |
| qrsbank_highamp_stattoken_badguard | 0.3 | 0.765129 | 0.646213 | 0.889835 | 0.703344 | 0.326034 | 0 | 0 | 0 | 20 | 1 | 0.0513699 | 0.0213241 |
| qrsbank_stattoken_balanced | 0.03 | 0.763714 | 0.654359 | 0.881868 | 0.70967 | 0.29927 | 0 | 0 | 0 | 20 | 1 | 0.0136986 | 0.00830647 |
| qrsbank_stattoken_balanced | 0.05 | 0.763596 | 0.658808 | 0.882967 | 0.710122 | 0.282238 | 0 | 0 | 0 | 23 | 0.97479 | 0 | 0.00210761 |
| qrsbank_stattoken_balanced | 0.04 | 0.763478 | 0.655613 | 0.882692 | 0.709896 | 0.284672 | 0 | 0 | 0 | 23 | 0.97479 | 0.00342466 | 0.0043392 |
| qrsbank_stattoken_balanced | 0.06 | 0.763478 | 0.659205 | 0.882967 | 0.710122 | 0.279805 | 0 | 0 | 0 | 24 | 0.966387 | 0 | 0.00136375 |
| qrsbank_stattoken_balanced | 0.02 | 0.763242 | 0.645935 | 0.879945 | 0.708089 | 0.323601 | 0 | 0 | 0 | 20 | 1 | 0.0479452 | 0.0198364 |
| qrsbank_badstress_hardneg_wide | 0.04 | 0.762534 | 0.654577 | 0.920604 | 0.675328 | 0.301703 | 0 | 0 | 0 | 42 | 0.991597 | 0.0205479 | 0.00805852 |
| qrsbank_badstress_hardneg_wide | 0.06 | 0.762416 | 0.654501 | 0.921703 | 0.675554 | 0.287105 | 0 | 0 | 0 | 46 | 0.957983 | 0.0136986 | 0.00495909 |
| qrsbank_badstress_hardneg_wide | 0.05 | 0.762298 | 0.652792 | 0.921429 | 0.675328 | 0.289538 | 0 | 0 | 0 | 45 | 0.966387 | 0.0136986 | 0.00644681 |
| qrsbank_badstress_hardneg_wide | 0.08 | 0.76218 | 0.653086 | 0.922527 | 0.675554 | 0.274939 | 0 | 0 | 0 | 50 | 0.92437 | 0.010274 | 0.00309943 |
| qrsbank_badstress_hardneg_wide | 0.07 | 0.762062 | 0.65352 | 0.921703 | 0.675554 | 0.279805 | 0 | 0 | 0 | 49 | 0.932773 | 0.0136986 | 0.00384329 |
| qrsbank_highamp_patch_balanced | 0.14 | 0.758877 | 0.653852 | 0.912088 | 0.676457 | 0.289538 | 0 | 0 | 0 | 39 | 0.97479 | 0.010274 | 0.00446318 |
| qrsbank_highamp_patch_balanced | 0.15 | 0.758877 | 0.654289 | 0.912088 | 0.676457 | 0.289538 | 0 | 0 | 0 | 39 | 0.97479 | 0.010274 | 0.00421522 |
| qrsbank_highamp_mix_stattoken_guard | 0.24 | 0.758759 | 0.650392 | 0.907143 | 0.679846 | 0.294404 | 0 | 0 | 0 | 18 | 1 | 0.00684932 | 0.00756261 |
| qrsbank_highamp_patch_balanced | 0.12 | 0.758759 | 0.653355 | 0.911813 | 0.676231 | 0.291971 | 0 | 0 | 0 | 38 | 0.983193 | 0.010274 | 0.00520704 |
| qrsbank_highamp_mix_stattoken_guard | 0.27 | 0.758759 | 0.650343 | 0.907418 | 0.679846 | 0.291971 | 0 | 0 | 0 | 18 | 1 | 0.00342466 | 0.0070667 |
| qrsbank_highamp_mix_stattoken_guard | 0.28 | 0.758759 | 0.650343 | 0.907418 | 0.679846 | 0.291971 | 0 | 0 | 0 | 18 | 1 | 0.00342466 | 0.0070667 |
| qrsbank_highamp_mix_stattoken_guard | 0.29 | 0.758759 | 0.650545 | 0.907418 | 0.679846 | 0.291971 | 0 | 0 | 0 | 18 | 1 | 0.00342466 | 0.00694272 |
| qrsbank_highamp_mix_stattoken_guard | 0.3 | 0.758759 | 0.650747 | 0.907418 | 0.679846 | 0.291971 | 0 | 0 | 0 | 18 | 1 | 0.00342466 | 0.00681875 |
| qrsbank_highamp_patch_balanced | 0.13 | 0.758759 | 0.652691 | 0.912088 | 0.676231 | 0.289538 | 0 | 0 | 0 | 39 | 0.97479 | 0.010274 | 0.00508306 |
| qrsbank_highamp_patch_balanced | 0.16 | 0.758641 | 0.652787 | 0.912088 | 0.676457 | 0.284672 | 0 | 0 | 0 | 41 | 0.957983 | 0.010274 | 0.00396727 |

## Best Bad-Outlier Thresholds

| candidate | threshold | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_to_bad | medium_to_bad | bad_to_good | bad_to_medium | bad_core_recall | bad_outlier_recall | nonbad_false_bad_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primitive_badexpert_multiscale | 0.01 | 0.70072 | 0.599109 | 0.770604 | 0.644374 | 0.688564 | 0 | 0 | 0 | 41 | 1 | 0.561644 | 0.226878 |
| qrsbank_highamp_mix_patch_guard | 0.01 | 0.746491 | 0.632498 | 0.80522 | 0.706055 | 0.6618 | 0 | 0 | 0 | 48 | 1 | 0.523973 | 0.166501 |
| qrsbank_badexpert_badguard | 0.01 | 0.753215 | 0.625678 | 0.860165 | 0.684817 | 0.542579 | 0 | 0 | 0 | 31 | 1 | 0.356164 | 0.142078 |
| qrsbank_highamp_patch_balanced | 0.01 | 0.731745 | 0.604781 | 0.877198 | 0.633077 | 0.506083 | 0 | 0 | 0 | 33 | 1 | 0.304795 | 0.14059 |
| gated_bad_multiscale_stablemix | 0.01 | 0.742834 | 0.612518 | 0.833242 | 0.692047 | 0.489051 | 0 | 0 | 0 | 61 | 1 | 0.280822 | 0.133028 |
| qrsbank_stattoken_badguard | 0.01 | 0.740828 | 0.610991 | 0.860989 | 0.665612 | 0.486618 | 0 | 0 | 0 | 15 | 1 | 0.277397 | 0.128068 |
| primitive_badexpert_multiscale | 0.02 | 0.764303 | 0.629217 | 0.851099 | 0.718934 | 0.484185 | 0 | 0 | 0 | 67 | 1 | 0.273973 | 0.122613 |
| qrsbank_highamp_mix_patch_guard | 0.02 | 0.782824 | 0.647466 | 0.843407 | 0.760958 | 0.481752 | 0 | 0 | 0 | 77 | 1 | 0.270548 | 0.0974461 |
| qrsbank_highamp_stattoken_badguard | 0.01 | 0.74767 | 0.616319 | 0.874176 | 0.669453 | 0.469586 | 0 | 0 | 0 | 15 | 1 | 0.253425 | 0.109968 |
| qrsbank_badstress_hardneg_balanced | 0.01 | 0.788015 | 0.658888 | 0.868956 | 0.75305 | 0.447689 | 0 | 0 | 0 | 62 | 1 | 0.222603 | 0.0612447 |
| qrsbank_stattoken_balanced | 0.01 | 0.761708 | 0.634109 | 0.871154 | 0.702214 | 0.43309 | 0 | 0 | 0 | 20 | 1 | 0.202055 | 0.0671956 |
| primitive_badexpert_multiscale | 0.03 | 0.784004 | 0.642263 | 0.869505 | 0.746724 | 0.428224 | 0 | 0 | 0 | 76 | 1 | 0.195205 | 0.0872799 |
| qrsbank_badexpert_badguard | 0.02 | 0.778459 | 0.639255 | 0.890385 | 0.719385 | 0.423358 | 0 | 0 | 0 | 35 | 1 | 0.188356 | 0.0800893 |
| qrsbank_highamp_mix_patch_guard | 0.03 | 0.796272 | 0.655768 | 0.853297 | 0.78423 | 0.420925 | 0 | 0 | 0 | 89 | 1 | 0.184932 | 0.070419 |
| qrsbank_highamp_mix_stattoken_guard | 0.01 | 0.747434 | 0.616624 | 0.898901 | 0.653186 | 0.420925 | 0 | 0 | 0 | 13 | 1 | 0.184932 | 0.0779817 |
| qrsbank_badexpert_balanced | 0.01 | 0.789194 | 0.649723 | 0.868681 | 0.759602 | 0.403893 | 0 | 0 | 0 | 47 | 1 | 0.160959 | 0.0642202 |
| gated_bad_multiscale_stablemix | 0.02 | 0.779757 | 0.639947 | 0.851374 | 0.756439 | 0.396594 | 0 | 0 | 0 | 75 | 1 | 0.150685 | 0.0680635 |
| qrsbank_highamp_stattoken_badguard | 0.02 | 0.757579 | 0.622802 | 0.881319 | 0.689788 | 0.391727 | 0 | 0 | 0 | 20 | 1 | 0.143836 | 0.0684354 |
| qrsbank_highamp_patch_balanced | 0.02 | 0.752389 | 0.619403 | 0.903022 | 0.662449 | 0.386861 | 0 | 0 | 0 | 36 | 1 | 0.136986 | 0.0647161 |
| primitive_badexpert_multiscale | 0.04 | 0.791318 | 0.646056 | 0.877747 | 0.758021 | 0.384428 | 0 | 0 | 0 | 82 | 1 | 0.133562 | 0.0681875 |
| qrsbank_stattoken_badguard | 0.02 | 0.758877 | 0.62594 | 0.877747 | 0.695888 | 0.384428 | 0 | 0 | 0 | 21 | 1 | 0.133562 | 0.0605009 |
| qrsbank_badexpert_badguard | 0.03 | 0.786481 | 0.649255 | 0.9 | 0.730908 | 0.379562 | 0 | 0 | 0 | 38 | 1 | 0.126712 | 0.0518225 |
| qrsbank_highamp_mix_patch_guard | 0.04 | 0.802053 | 0.659164 | 0.857143 | 0.79643 | 0.374696 | 0 | 0 | 0 | 94 | 1 | 0.119863 | 0.0524424 |
| qrsbank_highamp_stattoken_badguard | 0.03 | 0.760411 | 0.626286 | 0.883516 | 0.695436 | 0.36983 | 0 | 0 | 0 | 20 | 1 | 0.113014 | 0.0560377 |
| gated_bad_multiscale_stablemix | 0.03 | 0.788251 | 0.65058 | 0.856868 | 0.771125 | 0.364964 | 0 | 0 | 0 | 79 | 1 | 0.106164 | 0.0464914 |
| qrsbank_highamp_mix_patch_guard | 0.05 | 0.805474 | 0.665635 | 0.858242 | 0.803208 | 0.36253 | 0 | 0 | 0 | 98 | 1 | 0.10274 | 0.0415324 |
| primitive_badexpert_multiscale | 0.05 | 0.797334 | 0.652141 | 0.881593 | 0.768414 | 0.36253 | 0 | 0 | 0 | 86 | 1 | 0.10274 | 0.054302 |
| qrsbank_stattoken_badguard | 0.03 | 0.76395 | 0.63894 | 0.882418 | 0.703796 | 0.36253 | 0 | 0 | 0 | 23 | 1 | 0.10274 | 0.038061 |
| qrsbank_highamp_stattoken_badguard | 0.04 | 0.760882 | 0.629142 | 0.884615 | 0.696114 | 0.36253 | 0 | 0 | 0 | 20 | 1 | 0.10274 | 0.048847 |
| qrsbank_highamp_stattoken_badguard | 0.05 | 0.761708 | 0.631991 | 0.885165 | 0.697469 | 0.360097 | 0 | 0 | 0 | 20 | 1 | 0.0993151 | 0.0445078 |
| qrsbank_badstress_hardneg_balanced | 0.02 | 0.792379 | 0.66668 | 0.876923 | 0.763895 | 0.350365 | 0 | 0 | 0 | 68 | 1 | 0.0856164 | 0.0261592 |
| qrsbank_badexpert_badguard | 0.04 | 0.788015 | 0.653387 | 0.902473 | 0.734749 | 0.347932 | 0 | 0 | 0 | 39 | 1 | 0.0821918 | 0.0366972 |
| qrsbank_badstress_hardneg_wide | 0.01 | 0.759821 | 0.632624 | 0.916758 | 0.669001 | 0.347932 | 0 | 0 | 0 | 40 | 1 | 0.0821918 | 0.0374411 |
| gated_bad_multiscale_stablemix | 0.04 | 0.789784 | 0.651227 | 0.859066 | 0.77474 | 0.3382 | 0 | 0 | 0 | 82 | 1 | 0.0684932 | 0.037937 |
| qrsbank_highamp_mix_stattoken_guard | 0.02 | 0.756282 | 0.634127 | 0.903846 | 0.673746 | 0.3382 | 0 | 0 | 0 | 17 | 1 | 0.0684932 | 0.0301265 |
| qrsbank_stattoken_badguard | 0.04 | 0.765247 | 0.642177 | 0.883242 | 0.708089 | 0.335766 | 0 | 0 | 0 | 24 | 1 | 0.0650685 | 0.0277709 |
| qrsbank_badexpert_badguard | 0.05 | 0.788486 | 0.655918 | 0.903846 | 0.736105 | 0.3309 | 0 | 0 | 0 | 41 | 0.991597 | 0.0616438 | 0.0290107 |
| qrsbank_highamp_patch_balanced | 0.03 | 0.754276 | 0.62425 | 0.906319 | 0.668324 | 0.333333 | 0 | 0 | 0 | 36 | 1 | 0.0616438 | 0.0386809 |
| gated_bad_multiscale_stablemix | 0.05 | 0.791436 | 0.653418 | 0.860165 | 0.778355 | 0.323601 | 0 | 0 | 0 | 85 | 1 | 0.0479452 | 0.0318621 |
| qrsbank_stattoken_badguard | 0.05 | 0.767253 | 0.648572 | 0.88489 | 0.711704 | 0.323601 | 0 | 0 | 0 | 24 | 1 | 0.0479452 | 0.0199603 |
| qrsbank_stattoken_balanced | 0.02 | 0.763242 | 0.645935 | 0.879945 | 0.708089 | 0.323601 | 0 | 0 | 0 | 20 | 1 | 0.0479452 | 0.0198364 |
| qrsbank_highamp_patch_balanced | 0.04 | 0.755574 | 0.631198 | 0.907967 | 0.670583 | 0.321168 | 0 | 0 | 0 | 36 | 1 | 0.0445205 | 0.0277709 |
| qrsbank_badexpert_balanced | 0.02 | 0.796744 | 0.65798 | 0.878571 | 0.773836 | 0.318735 | 0 | 0 | 0 | 55 | 1 | 0.0410959 | 0.0293826 |
| qrsbank_badstress_hardneg_balanced | 0.03 | 0.793323 | 0.670165 | 0.877747 | 0.767962 | 0.318735 | 0 | 0 | 0 | 71 | 1 | 0.0410959 | 0.0158691 |
| qrsbank_highamp_mix_stattoken_guard | 0.03 | 0.75699 | 0.636483 | 0.90522 | 0.676005 | 0.316302 | 0 | 0 | 0 | 18 | 1 | 0.0376712 | 0.0223159 |
| qrsbank_badstress_hardneg_wide | 0.02 | 0.761236 | 0.642877 | 0.918681 | 0.673294 | 0.313869 | 0 | 0 | 0 | 41 | 1 | 0.0342466 | 0.0185966 |
| qrsbank_badstress_hardneg_balanced | 0.04 | 0.793559 | 0.673541 | 0.878571 | 0.76864 | 0.309002 | 0 | 0 | 0 | 71 | 1 | 0.0273973 | 0.0112819 |
| qrsbank_highamp_mix_stattoken_guard | 0.04 | 0.757107 | 0.637927 | 0.905495 | 0.676683 | 0.309002 | 0 | 0 | 0 | 18 | 1 | 0.0273973 | 0.0192165 |
| qrsbank_badexpert_balanced | 0.03 | 0.799103 | 0.666604 | 0.879945 | 0.778355 | 0.306569 | 0 | 0 | 0 | 56 | 1 | 0.0239726 | 0.0189685 |
| qrsbank_highamp_patch_balanced | 0.05 | 0.756518 | 0.635835 | 0.909341 | 0.672616 | 0.306569 | 0 | 0 | 0 | 36 | 1 | 0.0239726 | 0.0199603 |
| qrsbank_badstress_hardneg_wide | 0.04 | 0.762534 | 0.654577 | 0.920604 | 0.675328 | 0.301703 | 0 | 0 | 0 | 42 | 0.991597 | 0.0205479 | 0.00805852 |
| qrsbank_badstress_hardneg_wide | 0.03 | 0.761944 | 0.648965 | 0.91978 | 0.67465 | 0.304136 | 0 | 0 | 0 | 41 | 1 | 0.0205479 | 0.0120258 |
| qrsbank_highamp_mix_stattoken_guard | 0.07 | 0.757933 | 0.643562 | 0.905769 | 0.678491 | 0.304136 | 0 | 0 | 0 | 18 | 1 | 0.0205479 | 0.0140094 |
| qrsbank_badexpert_balanced | 0.04 | 0.800047 | 0.671176 | 0.88022 | 0.780389 | 0.301703 | 0 | 0 | 0 | 57 | 1 | 0.0171233 | 0.0145053 |
| qrsbank_badexpert_balanced | 0.15 | 0.803586 | 0.688246 | 0.883516 | 0.784907 | 0.296837 | 0 | 0 | 0 | 58 | 0.991597 | 0.0136986 | 0.00384329 |
| qrsbank_badstress_hardneg_balanced | 0.08 | 0.794503 | 0.684294 | 0.880769 | 0.769544 | 0.29927 | 0 | 0 | 0 | 71 | 1 | 0.0136986 | 0.00309943 |
| qrsbank_stattoken_balanced | 0.03 | 0.763714 | 0.654359 | 0.881868 | 0.70967 | 0.29927 | 0 | 0 | 0 | 20 | 1 | 0.0136986 | 0.00830647 |
| qrsbank_badstress_hardneg_wide | 0.06 | 0.762416 | 0.654501 | 0.921703 | 0.675554 | 0.287105 | 0 | 0 | 0 | 46 | 0.957983 | 0.0136986 | 0.00495909 |
| qrsbank_stattoken_balanced | 0.04 | 0.763478 | 0.655613 | 0.882692 | 0.709896 | 0.284672 | 0 | 0 | 0 | 23 | 0.97479 | 0.00342466 | 0.0043392 |
| qrsbank_stattoken_balanced | 0.05 | 0.763596 | 0.658808 | 0.882967 | 0.710122 | 0.282238 | 0 | 0 | 0 | 23 | 0.97479 | 0 | 0.00210761 |

## Bad Score Distributions

| candidate | bucket | n | bad_score_mean | bad_score_p50 | bad_score_p75 | bad_score_p90 | bad_score_p95 | bad_score_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_bad_multiscale_stablemix | test_bad_all | 411 | 0.304507 | 0.00928263 | 0.986815 | 0.995316 | 0.99623 | 0.997027 |
| gated_bad_multiscale_stablemix | test_bad_core | 119 | 0.992126 | 0.993701 | 0.995912 | 0.996575 | 0.996856 | 0.997027 |
| gated_bad_multiscale_stablemix | test_bad_outlier | 292 | 0.0242787 | 0.00518493 | 0.0111938 | 0.0324867 | 0.047534 | 0.979875 |
| gated_bad_multiscale_stablemix | test_nonbad | 8066 | 0.0127675 | 0.00319959 | 0.00606707 | 0.0127027 | 0.0278112 | 0.995303 |
| gated_bad_multiscale_stablemix | test_good | 3640 | 0.00558074 | 0.00167864 | 0.00309666 | 0.00676611 | 0.0123306 | 0.938564 |
| gated_bad_multiscale_stablemix | test_medium | 4426 | 0.018678 | 0.0046371 | 0.00812613 | 0.0198839 | 0.0515803 | 0.995303 |
| gated_bad_multiscale_stablemix | auc_bad_vs_nonbad | 8477 | 0.739718 | 0.624485 | nan | nan | nan | nan |
| primitive_badexpert_multiscale | test_bad_all | 411 | 0.265167 | 0.0179743 | 0.767764 | 0.886236 | 0.911194 | 0.933084 |
| primitive_badexpert_multiscale | test_bad_core | 119 | 0.844244 | 0.866593 | 0.897655 | 0.919471 | 0.924317 | 0.933084 |
| primitive_badexpert_multiscale | test_bad_outlier | 292 | 0.0291737 | 0.0108161 | 0.0232284 | 0.0508553 | 0.0826737 | 0.759467 |
| primitive_badexpert_multiscale | test_nonbad | 8066 | 0.014852 | 0.00463898 | 0.00901427 | 0.0258256 | 0.0535815 | 0.851106 |
| primitive_badexpert_multiscale | test_good | 3640 | 0.00787052 | 0.00463773 | 0.00718294 | 0.0123554 | 0.01921 | 0.432471 |
| primitive_badexpert_multiscale | test_medium | 4426 | 0.0205936 | 0.00466326 | 0.0131088 | 0.0447648 | 0.0849777 | 0.851106 |
| primitive_badexpert_multiscale | auc_bad_vs_nonbad | 8477 | 0.831496 | 0.751817 | nan | nan | nan | nan |
| qrsbank_badexpert_balanced | test_bad_all | 411 | 0.214692 | 0.00450662 | 0.429084 | 0.872073 | 0.929179 | 0.976756 |
| qrsbank_badexpert_balanced | test_bad_core | 119 | 0.713334 | 0.785851 | 0.904917 | 0.952921 | 0.961387 | 0.976756 |
| qrsbank_badexpert_balanced | test_bad_outlier | 292 | 0.011479 | 0.00265438 | 0.00553414 | 0.0125577 | 0.0188632 | 0.595463 |
| qrsbank_badexpert_balanced | test_nonbad | 8066 | 0.00520999 | 0.00145534 | 0.00271992 | 0.0064694 | 0.0124217 | 0.839205 |
| qrsbank_badexpert_balanced | test_good | 3640 | 0.00233846 | 0.00113389 | 0.0020113 | 0.00352218 | 0.00538979 | 0.494315 |
| qrsbank_badexpert_balanced | test_medium | 4426 | 0.00757157 | 0.00178274 | 0.00353921 | 0.00983989 | 0.0196251 | 0.839205 |
| qrsbank_badexpert_balanced | auc_bad_vs_nonbad | 8477 | 0.785214 | 0.687873 | nan | nan | nan | nan |
| qrsbank_badexpert_badguard | test_bad_all | 411 | 0.248703 | 0.0123127 | 0.580653 | 0.945795 | 0.96589 | 0.977742 |
| qrsbank_badexpert_badguard | test_bad_core | 119 | 0.799638 | 0.898605 | 0.957724 | 0.970484 | 0.973381 | 0.977742 |
| qrsbank_badexpert_badguard | test_bad_outlier | 292 | 0.0241786 | 0.00749491 | 0.0156363 | 0.0361671 | 0.0607028 | 0.898424 |
| qrsbank_badexpert_badguard | test_nonbad | 8066 | 0.00957926 | 0.0026863 | 0.00564015 | 0.0153881 | 0.0308012 | 0.90128 |
| qrsbank_badexpert_badguard | test_good | 3640 | 0.00490782 | 0.00264873 | 0.00438645 | 0.00719351 | 0.0102918 | 0.532012 |
| qrsbank_badexpert_badguard | test_medium | 4426 | 0.0134211 | 0.00277812 | 0.0080293 | 0.0259728 | 0.0475837 | 0.90128 |
| qrsbank_badexpert_badguard | auc_bad_vs_nonbad | 8477 | 0.833508 | 0.754968 | nan | nan | nan | nan |
| qrsbank_badstress_hardneg_balanced | test_bad_all | 411 | 0.221395 | 0.00773159 | 0.513011 | 0.855783 | 0.892219 | 0.962546 |
| qrsbank_badstress_hardneg_balanced | test_bad_core | 119 | 0.742623 | 0.807537 | 0.883183 | 0.911851 | 0.92814 | 0.962546 |
| qrsbank_badstress_hardneg_balanced | test_bad_outlier | 292 | 0.00897673 | 0.00438028 | 0.00912942 | 0.0175943 | 0.0254212 | 0.252594 |
| qrsbank_badstress_hardneg_balanced | test_nonbad | 8066 | 0.00371193 | 0.00134744 | 0.00282493 | 0.00634496 | 0.0120328 | 0.656755 |
| qrsbank_badstress_hardneg_balanced | test_good | 3640 | 0.00209354 | 0.0011974 | 0.00215587 | 0.00357943 | 0.00513488 | 0.122995 |
| qrsbank_badstress_hardneg_balanced | test_medium | 4426 | 0.00504292 | 0.00151744 | 0.00373952 | 0.0101185 | 0.017906 | 0.656755 |
| qrsbank_badstress_hardneg_balanced | auc_bad_vs_nonbad | 8477 | 0.854969 | 0.784315 | nan | nan | nan | nan |
| qrsbank_badstress_hardneg_wide | test_bad_all | 411 | 0.175747 | 0.00285564 | 0.162901 | 0.804316 | 0.899064 | 0.941477 |
| qrsbank_badstress_hardneg_wide | test_bad_core | 119 | 0.59159 | 0.721211 | 0.870951 | 0.917344 | 0.924899 | 0.941477 |
| qrsbank_badstress_hardneg_wide | test_bad_outlier | 292 | 0.00627754 | 0.00171927 | 0.00365349 | 0.00877736 | 0.0149002 | 0.41252 |
| qrsbank_badstress_hardneg_wide | test_nonbad | 8066 | 0.00260127 | 0.00069404 | 0.00147475 | 0.00351114 | 0.00721326 | 0.557589 |
| qrsbank_badstress_hardneg_wide | test_good | 3640 | 0.00115601 | 0.000664978 | 0.00104631 | 0.00179944 | 0.00258434 | 0.076865 |
| qrsbank_badstress_hardneg_wide | test_medium | 4426 | 0.00378987 | 0.0007633 | 0.00210161 | 0.00618456 | 0.0126889 | 0.557589 |
| qrsbank_badstress_hardneg_wide | auc_bad_vs_nonbad | 8477 | 0.829795 | 0.749689 | nan | nan | nan | nan |
| qrsbank_stattoken_balanced | test_bad_all | 411 | 0.167886 | 0.00753846 | 0.234825 | 0.730455 | 0.811741 | 0.919079 |
| qrsbank_stattoken_balanced | test_bad_core | 119 | 0.562373 | 0.651491 | 0.761378 | 0.847876 | 0.884322 | 0.919079 |
| qrsbank_stattoken_balanced | test_bad_outlier | 292 | 0.00711903 | 0.00489603 | 0.00796991 | 0.0144773 | 0.0182591 | 0.0412397 |
| qrsbank_stattoken_balanced | test_nonbad | 8066 | 0.00352001 | 0.00185488 | 0.0035696 | 0.0076053 | 0.0120112 | 0.165557 |
| qrsbank_stattoken_balanced | test_good | 3640 | 0.00267258 | 0.00209225 | 0.00297803 | 0.00428714 | 0.00545979 | 0.0400564 |
| qrsbank_stattoken_balanced | test_medium | 4426 | 0.00421694 | 0.00106554 | 0.00499101 | 0.0108838 | 0.0159518 | 0.165557 |
| qrsbank_stattoken_balanced | auc_bad_vs_nonbad | 8477 | 0.864669 | 0.797827 | nan | nan | nan | nan |
| qrsbank_stattoken_badguard | test_bad_all | 411 | 0.288821 | 0.00940603 | 0.956386 | 0.98543 | 0.987843 | 0.990042 |
| qrsbank_stattoken_badguard | test_bad_core | 119 | 0.967063 | 0.983446 | 0.987169 | 0.98833 | 0.989038 | 0.990042 |
| qrsbank_stattoken_badguard | test_bad_outlier | 292 | 0.0124147 | 0.00543105 | 0.0111535 | 0.0298716 | 0.0466138 | 0.167089 |
| qrsbank_stattoken_badguard | test_nonbad | 8066 | 0.00792153 | 0.00231508 | 0.00468997 | 0.0129218 | 0.0238054 | 0.852378 |
| qrsbank_stattoken_badguard | test_good | 3640 | 0.00310746 | 0.00173352 | 0.00315558 | 0.00523502 | 0.00802246 | 0.257401 |
| qrsbank_stattoken_badguard | test_medium | 4426 | 0.0118807 | 0.00288943 | 0.00790431 | 0.0203046 | 0.0377706 | 0.852378 |
| qrsbank_stattoken_badguard | auc_bad_vs_nonbad | 8477 | 0.804594 | 0.714425 | nan | nan | nan | nan |
| qrsbank_highamp_patch_balanced | test_bad_all | 411 | 0.213146 | 0.0104038 | 0.386277 | 0.881754 | 0.925751 | 0.977973 |
| qrsbank_highamp_patch_balanced | test_bad_core | 119 | 0.69439 | 0.803733 | 0.903716 | 0.945088 | 0.95324 | 0.977973 |
| qrsbank_highamp_patch_balanced | test_bad_outlier | 292 | 0.0170232 | 0.00674191 | 0.0130011 | 0.0225904 | 0.0350554 | 0.911705 |
| qrsbank_highamp_patch_balanced | test_nonbad | 8066 | 0.00774237 | 0.00280543 | 0.00536728 | 0.0139681 | 0.0246895 | 0.935059 |
| qrsbank_highamp_patch_balanced | test_good | 3640 | 0.00398572 | 0.00265096 | 0.00403762 | 0.00637423 | 0.00906911 | 0.503697 |
| qrsbank_highamp_patch_balanced | test_medium | 4426 | 0.0108319 | 0.00304621 | 0.00866081 | 0.0216147 | 0.0367605 | 0.935059 |
| qrsbank_highamp_patch_balanced | auc_bad_vs_nonbad | 8477 | 0.836884 | 0.759566 | nan | nan | nan | nan |
| qrsbank_highamp_stattoken_badguard | test_bad_all | 411 | 0.314352 | 0.00760947 | 0.969168 | 0.978478 | 0.981305 | 0.986023 |
| qrsbank_highamp_stattoken_badguard | test_bad_core | 119 | 0.975646 | 0.976908 | 0.979889 | 0.982999 | 0.983907 | 0.986023 |
| qrsbank_highamp_stattoken_badguard | test_bad_outlier | 292 | 0.0448519 | 0.00439387 | 0.0100932 | 0.0444813 | 0.291918 | 0.878629 |
| qrsbank_highamp_stattoken_badguard | test_nonbad | 8066 | 0.0220067 | 0.00203525 | 0.00374519 | 0.0113071 | 0.0380359 | 0.979219 |
| qrsbank_highamp_stattoken_badguard | test_good | 3640 | 0.00587653 | 0.00185253 | 0.00281435 | 0.00391068 | 0.00549551 | 0.929431 |
| qrsbank_highamp_stattoken_badguard | test_medium | 4426 | 0.0352723 | 0.00232158 | 0.00659649 | 0.026678 | 0.131325 | 0.979219 |
| qrsbank_highamp_stattoken_badguard | auc_bad_vs_nonbad | 8477 | 0.829151 | 0.748592 | nan | nan | nan | nan |
| qrsbank_highamp_mix_patch_guard | test_bad_all | 411 | 0.280478 | 0.0186452 | 0.809008 | 0.957008 | 0.969312 | 0.979578 |
| qrsbank_highamp_mix_patch_guard | test_bad_core | 119 | 0.902339 | 0.940892 | 0.9646 | 0.972412 | 0.975022 | 0.979578 |
| qrsbank_highamp_mix_patch_guard | test_bad_outlier | 292 | 0.0270484 | 0.0103738 | 0.0231055 | 0.0501494 | 0.0715237 | 0.631197 |
| qrsbank_highamp_mix_patch_guard | test_nonbad | 8066 | 0.0119868 | 0.00330964 | 0.00663346 | 0.0192899 | 0.0419553 | 0.843719 |
| qrsbank_highamp_mix_patch_guard | test_good | 3640 | 0.00587339 | 0.00326006 | 0.00524637 | 0.00836931 | 0.0126239 | 0.481302 |
| qrsbank_highamp_mix_patch_guard | test_medium | 4426 | 0.0170146 | 0.00338601 | 0.00987233 | 0.035438 | 0.0698909 | 0.843719 |
| qrsbank_highamp_mix_patch_guard | auc_bad_vs_nonbad | 8477 | 0.855617 | 0.785235 | nan | nan | nan | nan |
| qrsbank_highamp_mix_stattoken_guard | test_bad_all | 411 | 0.294477 | 0.00705719 | 0.988204 | 0.992683 | 0.993108 | 0.99389 |
| qrsbank_highamp_mix_stattoken_guard | test_bad_core | 119 | 0.990509 | 0.992055 | 0.992893 | 0.993284 | 0.993608 | 0.99389 |
| qrsbank_highamp_mix_stattoken_guard | test_bad_outlier | 292 | 0.0108205 | 0.00468416 | 0.008017 | 0.0160257 | 0.0266172 | 0.488454 |
| qrsbank_highamp_mix_stattoken_guard | test_nonbad | 8066 | 0.0100505 | 0.00285577 | 0.00523864 | 0.00870387 | 0.0128486 | 0.967858 |
| qrsbank_highamp_mix_stattoken_guard | test_good | 3640 | 0.00397092 | 0.00236936 | 0.00343116 | 0.00526023 | 0.0065489 | 0.658693 |
| qrsbank_highamp_mix_stattoken_guard | test_medium | 4426 | 0.0150505 | 0.00381462 | 0.00701571 | 0.0116979 | 0.0202465 | 0.967858 |
| qrsbank_highamp_mix_stattoken_guard | auc_bad_vs_nonbad | 8477 | 0.790583 | 0.694986 | nan | nan | nan | nan |

- Frontier CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_score_frontier.csv`
- Score distribution CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_score_distributions.csv`
- Prediction CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_score_original_test_predictions.csv`
