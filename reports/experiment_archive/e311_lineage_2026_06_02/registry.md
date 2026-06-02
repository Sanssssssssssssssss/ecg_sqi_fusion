# E3.11f Lineage Registry

| Tag | Status | Method | Key result | Value | Recommendation |
|---|---|---|---|---|---|
| `00_early_e311_grid` | archived context | Legacy E3.11 Transformer/SQI grid | Acc anchors around `0.9519`; preferred balanced run around `0.9505` / bad recall `0.9837` | Established classification guardrails | Historical baseline only |
| `01_residual_denoise` | archived context | Predict `noise_hat` instead of clean ECG | Strong classification, weak early visual denoise | Locked in residual prediction | Retained as design principle |
| `02_snr_sweep` | archived context | Bad/medium SNR gap sweep | Gap 5-7 dB made denoise visually inspectable | Selected med6.25/badgap7 artifact | Current E3.11f dataset |
| `03_morph_denoise_gap5_7` | archived context | Morph-aware U-Net denoise controls | Strong visual denoise and encoder-head classification | U-Net teacher/oracle proof | Not final mainline |
| `04_sqi_classifier` | archived context | Full-joint denoise-before-classifier + explicit SQI | Full Transformer paths around `0.962-0.973` | SQI useful but unstable alone | Keep as ablation |
| `05_loss_conflict` | archived context | PCGrad/cap and guard strategies | CE can corrupt denoiser | Justified detached classifier | Keep as analysis |
| `06_warm_encoder_head` | archived context | Warm encoder-head freeze/detach/scratch audit | Warm latent much stronger than scratch/proxies | Proved denoiser representation matters | Mechanism retained |
| `07_transformer_reentry` | archived context | Transformer classifier adapters on U-Net teacher | Best adapter acc `0.98093`, bad `0.99455` | Reintroduced Transformer logits but not Transformer denoise | Comparator |
| `08_transunet_uformer` | mainline parent | Transformer-based denoiser audit | Uformer acc `0.98955`, bad `0.99728`, denoise `4.303` | Real Transformer denoiser works | Promote Uformer |
| `09_uformer_king_ablation` | mainline decision | Uformer mechanism ablations | Chosen `a_repr`: acc `0.99001`, bad `0.99591`, denoise `4.282` | Shows full tokens/skips/Transformer/detach are necessary | Thesis core |
| `10_mainline_rerun_seed0` | current mainline output | Clean source rerun | Acc `0.98819`, bad `0.99591`, denoise `4.293`, SNR `12.386` | Confirms clean implementation stands up | Current clean mainline |
