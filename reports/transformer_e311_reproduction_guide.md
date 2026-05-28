# E3.11f Strict Mainline Reproduction Guide

This note freezes the current E3.11f visual-SQI experiment so it can be
reproduced on another machine or cluster. It documents the active data version,
strict clean-source filtering, the current best transformer recipe, and the
optional isolated research experiments.

Reference code commit used when this guide was written: `d8dc722`.

## What This Experiment Is

Current mainline:

- Data: `e311f_lite_e310_morph`
- Output root: `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph`
- Clean source: PTB-XL Lead I, 10 s windows at 125 Hz, split by `ecg_id`
- Noise source: NSTDB `em`, `ma`, and `mix`; `bw` is intentionally removed
- Label family: E3.10 smooth morphology rule plus E3.11f lite visual SNR ranges
- Model family: simple raw Lead-I transformer with CLS pooling, positional embedding, D1 warm-start, SNR head, and optional low-weight local-mask/rank auxiliaries

The current best single run is:

```text
e311f_lite_e310_morph_hc2_m0075_rank005_lr625_s1
```

Results:

```text
test acc = 0.9519
recall good / medium / bad = 0.9319 / 0.9428 / 0.9809
confusion matrix =
  [[684, 44,  6],
   [ 27,692, 15],
   [  0, 14,720]]
```

Current reports:

- `reports/transformer_e311_mainline_grid_report.md`
- `reports/transformer_e311_head_combo_grid_report.md`
- `reports/ptbxl_strict_clean_filter_audit.md`

## Required Local Data

The repo expects this layout under the project root:

```text
data/
  ptb-xl/
    ptbxl_database.csv
    records100/ or records500/
  physionet/
    nstdb/
      em.*
      ma.*
      bw.* optional, not used by E3.11f
```

Install the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On CSD3, the Slurm scripts assume:

```text
/home/cx272/final_project/ecg_sqi_fusion
```

If reproducing elsewhere, either edit the `cd` line in the Slurm scripts or run
the Python commands directly from the repository root.

## Strict Clean-Source Filtering

The active source artifact is:

```text
outputs/transformer_source_strict_clean
```

It is built from PTB-XL records after rejecting any record with non-empty
metadata in:

```text
baseline_drift
static_noise
burst_noise
electrodes_problems
```

This is strict across all leads, not only Lead I. A small manual visual exclusion
list is also applied:

```text
config/ptbxl_clean_source_exclude_ecg_ids.txt
```

At the time of this guide, the manual exclusions are:

```text
5428
5484
```

The strict audit verified:

```text
strict + manual records kept = 16787
E3.11f unique clean-source ecg_ids = 5107
metadata noise/electrode hits in E3.11f = 0
hits for 5428/5447/5484 in E3.11f = 0
```

See `reports/ptbxl_strict_clean_filter_audit.md`.

## Generate The E3.11f Data

The easiest path is the CPU data/audit script:

```bash
FORCE_DATA=1 \
ROOT_OUT=outputs/transformer_e311_mainline_strict \
SOURCE_ARTIFACT_DIR=outputs/transformer_source_strict_clean \
GROUP_RETRIES=24 \
MAX_TRAIN_CLEAN=4000 \
MAX_VAL_CLEAN=800 \
MAX_TEST_CLEAN=800 \
bash slurm/run_e311_mainline_data_audit.sh
```

On CSD3 CPU:

```bash
sbatch slurm/run_e311_mainline_data_audit.sh
```

That script generates three diagnostic variants:

```text
e311f_lite_e310_morph      active mainline
e311h_lite_relaxed_morph   pruned, learned poorly
e311i_wide_relaxed_morph   diagnostic only, SQI baseline too high
```

For only the active E3.11f data, the underlying command is:

```bash
.venv/bin/python -u -m src.transformer_pipeline.noise.synthesize_morph_damage_triplet \
  --force \
  --verbose \
  --artifact_dir outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph \
  --source_artifact_dir outputs/transformer_source_strict_clean \
  --label_version e311f_lite_e310_morph \
  --noise_kinds em,ma,mix \
  --group_retries 24 \
  --max_train_clean 4000 \
  --max_val_clean 800 \
  --max_test_clean 800

.venv/bin/python -u -m src.transformer_pipeline.noise.make_rr_noise_level \
  --force \
  --verbose \
  --artifact_dir outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph
```

E3.11f uses these visual SNR ranges:

```text
good   11.5-13.0 dB
medium  8.5-10.0 dB
bad     6.5-8.0 dB
```

Labels still use the smooth morphology rule:

```text
good:
  smooth_morph_score <= 0.10
  qrs_nprd <= 0.10
  beat_corr >= 0.95

medium:
  0.27 <= smooth_morph_score <= 0.40
  qrs_nprd < 0.35
  beat_corr >= 0.80

bad:
  smooth_morph_score >= 0.58
  or (qrs_nprd >= 0.45 and smooth_morph_score >= 0.32)
  or (beat_corr <= 0.70 and smooth_morph_score >= 0.32)

gray:
  otherwise, excluded from the main benchmark
```

Expected E3.11f class counts after generation:

```text
train: good/medium/bad = 3645/3645/3645
val:   good/medium/bad =  728/ 728/ 728
test:  good/medium/bad =  734/ 734/ 734
```

## Visualize The Data

Generate the counterfactual triplet gallery:

```bash
.venv/bin/python -m src.transformer_pipeline.diagnostics.viz_morph_triplet_samples \
  --artifact_dir outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph \
  --out_dir outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/visual_gallery \
  --prefix e311f_strict_refiltered \
  --title "E3.11f strict-filtered" \
  --split test \
  --triplets 10 \
  --examples_per_cell 2
```

Current generated gallery paths:

```text
outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/visual_gallery/e311f_strict_refiltered_counterfactual_triplets_gallery.png
outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/visual_gallery/e311f_strict_refiltered_class_noise_examples_gallery.png
```

These output files are not tracked by git.

## Train The Current Best Model

Dependency: the D1 warm-start checkpoint must exist at:

```text
outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt
```

The exact current best recipe:

```bash
.venv/bin/python -u -m src.transformer_pipeline.run_transformer_all \
  --force \
  --verbose \
  --stage model \
  --artifact_dir outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph \
  --seed 1 \
  --experiment_name e311f_lite_e310_morph_hc2_m0075_rank005_lr625_s1 \
  --epochs 22 \
  --batch_size 32 \
  --lr 6.25e-5 \
  --lr_eta_min 4e-6 \
  --weight_decay 0.03 \
  --dropout 0.10 \
  --cls_pool cls \
  --input_mode raw \
  --snr_head \
  --lambda_snr 0.05 \
  --local_mask_head \
  --lambda_local_mask 0.0075 \
  --lambda_rank 0.005 \
  --rank_margin 0.10 \
  --label_smoothing 0 \
  --class_weight_good 1 \
  --class_weight_medium 1 \
  --class_weight_bad 1 \
  --init_checkpoint outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt \
  --select_best_by val_acc \
  --use_positional_embedding \
  --e_cls 22 \
  --e_denoise 0 \
  --e_level 0 \
  --e_uncert 0 \
  --lambda_den 0 \
  --lambda_lvl 0 \
  --lambda_ord 0 \
  --lambda_noise_type 0
```

Equivalent Slurm command:

```bash
sbatch --array=39 slurm/tune_e311_head_combo_round2.sh
```

The full focused head-combo sweep is:

```bash
sbatch slurm/tune_e311_head_combo_round2.sh
```

That full sweep contains 40 jobs and should be used only when the queue can
handle it. For polite replication, run selected array indices first:

```text
21: mask=0.0075, lr=6.25e-5, seed=1, no rank
39: mask=0.0075, lr=6.25e-5, seed=1, rank=0.005
1:  mask=0.01,   lr=6.25e-5, seed=1, no rank
```

## Summarize Results

After training:

```bash
.venv/bin/python -m src.transformer_pipeline.analyze_training \
  --model_dir outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/models/e311f_lite_e310_morph_hc2_m0075_rank005_lr625_s1

.venv/bin/python -u -m src.transformer_pipeline.diagnostics.summarize_e311_mainline_grid \
  --root_out outputs/transformer_e311_mainline_strict

.venv/bin/python -u -m src.transformer_pipeline.diagnostics.summarize_e311_head_combo_grid \
  --root_out outputs/transformer_e311_mainline_strict
```

Primary result files:

```text
outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/models/e311f_lite_e310_morph_hc2_m0075_rank005_lr625_s1/test_report.json
outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/models/e311f_lite_e310_morph_hc2_m0075_rank005_lr625_s1/eval_best/test_report_best.json
reports/transformer_e311_mainline_grid_report.md
reports/transformer_e311_head_combo_grid_report.md
```

## What We Learned

Keep:

- E3.11f strict-filtered data as the visual mainline.
- CLS pooling, raw input, positional embedding, D1 warm-start.
- SNR auxiliary head with `lambda_snr=0.05`.
- Low-weight local mask supervision. Best single run used `lambda_local_mask=0.0075`.
- Very small rank loss can help in combination with lower mask weight. Best run used `lambda_rank=0.005`.

Drop or treat as diagnostic:

- Relaxed morphology data (`e311h`) stayed around `0.90-0.91`.
- Wide SNR relaxed morphology (`e311i`) has higher SQI baselines and is not the main benchmark.
- Noise-type head did not help.
- Ordinal plus noise was clearly worse.
- Denoise/level auxiliaries were not part of the current best E3.11f recipe.

The local-mask target is the synthetic local mask generated during noise
synthesis. It is used only as a weak auxiliary training target here; it is not a
new input channel at inference time.

## Optional Isolated Research Experiments

An isolated research package exists at:

```text
src/experiment/e311_sqi_research/
```

It does not modify `src/sqi_pipeline` or `src/transformer_pipeline`. It reads the
same E3.11f data and D1 checkpoint, then writes to:

```text
outputs/experiment/e311_sqi_research/
```

List recipes:

```bash
.venv/bin/python -m src.experiment.e311_sqi_research.train --list
```

Dry-run:

```bash
.venv/bin/python -m src.experiment.e311_sqi_research.train \
  --group head_reimpl \
  --task_id 0 \
  --dry_run
```

Submit one small sanity job:

```bash
sbatch --array=0 src/experiment/e311_sqi_research/slurm/run_head_reimpl.sh
```

The research scripts are throttled to `%1` to avoid overloading the GPU queue.
Use them only after reproducing the mainline result above.

## Artifact Sync Checklist

To reproduce without regenerating everything, copy these directories from the
source machine:

```text
outputs/transformer_source_strict_clean/
outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/
outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph/datasets/
```

To reproduce from raw data, copy only `data/` and run the generation commands in
this guide.

Do not rely on internal fields such as `seg_id`, `source_npz_index`, or
`counterfactual_group` as PTB-XL record IDs. For source contamination checks,
use the `ecg_id` column in `synth_10s_125hz_labels.csv`.
