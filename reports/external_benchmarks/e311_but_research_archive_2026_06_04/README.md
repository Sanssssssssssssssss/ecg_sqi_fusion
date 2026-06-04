# E3.11f BUT Synthetic-Rule Research Archive

This folder is the GitHub-readable handoff package for the E3.11f BUT QDB work. It is designed so another chat can understand the full research thread without local checkpoints or raw data.

## Why We Did This

The mainline E3.11f model was built on PTB-derived synthetic SQI labels. On internal PTB synthetic tests the Uformer representation and denoise-before-classifier mechanism performed strongly, but that alone does not prove the signal-quality labels match real expert judgement. BUT QDB is the key external check because it provides expert consensus ECG quality classes:

- BUT class 1 -> good: P/T/QRS visible and reliable enough for detailed analysis.
- BUT class 2 -> medium: QRS usable, but finer intervals/details become unreliable.
- BUT class 3 -> bad: signal unsuitable for further analysis.

The formal protocol in this archive is **BUT 10s P1**: 10-second windows, class 1/2/3 mapped directly to good/medium/bad, validation-only calibration, and test used only for reporting.

## Current Headline Conclusion

The old synthetic PTB rule was too close to a noise-strength/SNR axis. BUT behaves more like a **diagnostic-usability boundary**:

- good is an AND condition: all critical dimensions must be usable.
- medium is not a midpoint; it is a QRS-usable/detail-unreliable cluster.
- bad is an OR condition over fatal usability failures, and some bad examples can still have visible QRS while the whole strip is not analyzable.

The best strict synthetic anchor so far is `h_bad_rescue_05` from the morphology-guided grid: acc 0.8229, balanced 0.8177, macro-F1 0.7454, recalls 0.887/0.773/0.793. Later sample-level OR experiments were mechanism-informative but did not beat that anchor.

## How To Read This Folder

- `timeline.md` explains the order of experiments and why each one happened.
- `rulebook/` contains detailed synthetic data rules and failure-mode interpretations.
- `results/experiment_registry.csv` is the compact index of all experiment families.
- `results/top_variant_metrics.csv` is a metric table for the best/most relevant grid rows.
- `analysis/` contains BUT morphology statistics, medium-cluster analysis, and distance-vs-metric material.
- `figures/` contains representative visual evidence that motivated the rule changes.
- `next_hypotheses.md` summarizes what to try next.

## What Is Intentionally Not Included

No checkpoints, NPZ arrays, raw signals, parquet files, or large `outputs/` trees are committed here. The local machine keeps the full artifacts under `outputs/external_benchmarks/`; this archive keeps the analysis surface needed for GitHub review and another chat.
