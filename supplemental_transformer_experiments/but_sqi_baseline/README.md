# BUT v116 SQI Baseline Comparator

Independent control experiment for running classical SQI SVM and LM-MLP
baselines on the official BUT v116 split.

BUT is single-lead, while the SQI baseline expects the 84-feature 12-lead SQI
table. This experiment copies the BUT lead into pseudo-12-lead columns, then
recomputes QRS detections and SQIs from waveform. The SVM control uses all 84
normalized SQI columns with a linear SVM; the LM-MLP reuses the SQI baseline
model implementation.

Binary label mapping:

- `good -> 1`
- `medium/bad -> -1`

All outputs go to:

```powershell
outputs/transformer/supplemental/but_sqi_baseline/
```

Run:

```powershell
python -m supplemental_transformer_experiments.but_sqi_baseline.run pipeline --run
```

Dry-run:

```powershell
python -m supplemental_transformer_experiments.but_sqi_baseline.run pipeline
```
