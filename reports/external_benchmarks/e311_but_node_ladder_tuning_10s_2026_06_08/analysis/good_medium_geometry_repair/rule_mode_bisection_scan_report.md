# Rule-Mode Bisection Scan

This scan uses existing endpoint predictions only. It does not train, alter checkpoints, or use original BUT for selection.

## N7125_gm_trim_bad

- Artifact: `rule_n7125_gm_trim_bad_qrslow_bisect_959a357114d7_e48e7d59927b`
- Base: `nl_n7125_gm_trim_bad_geom_addedring_n7100base_g002_m010_g_959a357114d7` / `raw`
- Alt: `nl_n7125_gm_trim_bad_geom_addedring_n7100base_g004_m014_g_e48e7d59927b` / `medium_guarded_pmed001`
- Train+val threshold: `qrs_visibility <= 0.51772644`; gate flips `1251`
- Gated acc `0.969783` vs base `0.912567`; macro-F1 `0.972617`
- Good/medium/bad recall `0.985825` / `0.948211` / `0.979432`
- Deltas vs base: acc `+0.057216`, good `-0.009825`, medium `+0.157053`, bad `+0.000000`
- Gate fixed medium `1138`; lost good `0`; eligible `True`
- PCA: `n7125_gm_trim_bad_rule_mode_bisection_pca.png`; waveforms: `n7125_gm_trim_bad_rule_mode_bisection_waveforms.png`

## N7150_gm_trim_bad

- Artifact: `rule_n7150_gm_trim_bad_qrslow_bisect_e319672f8889_f297c2a54bbc`
- Base: `nl_n7150_gm_trim_bad_geom_addedring_n7100base_g006_m020_g_e319672f8889` / `raw`
- Alt: `nl_n7150_gm_trim_bad_geom_mediumlost_n7150base_g000_m022__f297c2a54bbc` / `medium_guarded_pmed0005`
- Train+val threshold: `qrs_visibility <= 0.50527161`; gate flips `695`
- Gated acc `0.954254` vs base `0.931734`; macro-F1 `0.958630`
- Good/medium/bad recall `0.988811` / `0.910350` / `0.970617`
- Deltas vs base: acc `+0.022520`, good `-0.004755`, medium `+0.062657`, bad `+0.000000`
- Gate fixed medium `554`; lost good `0`; eligible `True`
- PCA: `n7150_gm_trim_bad_rule_mode_bisection_pca.png`; waveforms: `n7150_gm_trim_bad_rule_mode_bisection_waveforms.png`

## N7200_gm_trim_bad

- Artifact: `rule_n7200_gm_trim_bad_qrslow_bisect_2839a07720c5_942f7f261b19`
- Base: `nl_n7200_gm_trim_bad_geom_stack_n7000_g008_m044_g115_m184_2839a07720c5` / `medium_guarded_pmed0005`
- Alt: `nl_n7200_gm_trim_bad_geom_tri_atlaspc2_matchold_g002_m006_942f7f261b19` / `medium_guarded_pmed0005`
- Train+val threshold: `qrs_visibility <= 0.53027088`; gate flips `924`
- Gated acc `0.960831` vs base `0.927288`; macro-F1 `0.964402`
- Good/medium/bad recall `0.972778` / `0.943194` / `0.970862`
- Deltas vs base: acc `+0.033543`, good `-0.008194`, medium `+0.094306`, bad `+0.000000`
- Gate fixed medium `753`; lost good `0`; eligible `True`
- PCA: `n7200_gm_trim_bad_rule_mode_bisection_pca.png`; waveforms: `n7200_gm_trim_bad_rule_mode_bisection_waveforms.png`
