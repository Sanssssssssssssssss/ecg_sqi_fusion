# ECG SQI Fusion

ECG signal-quality assessment helps identify recordings that are unsafe or
unreliable for clinical interpretation and downstream modelling. This research
repository compares classical signal-quality-index (SQI) models with waveform
Conformers on public 12-lead Set-A and single-lead Brno University of Technology
(BUT) data, and packages four frozen models for reproducible inference.

## Submission Documents

- [`report/`](report/) contains the complete report and final submission material.
- [`executive summary/`](executive%20summary/) contains the separate executive summary.

Both directories currently contain placeholders; the final documents will be
added before submission.

## Fastest Inference: Docker

From the repository root:

```bash
docker build -f docker/inference/Dockerfile -t ecg-sqi-infer .
docker run --rm -v /host/data:/data ecg-sqi-infer predict \
  --model singlelead-conformer --input /data/input --fs 500 --out /data/output
```

Available models are `12lead-conformer`, `singlelead-conformer`,
`12lead-rbfsvm`, and `singlelead-rbfsvm`. Inputs may be NumPy, CSV, or WFDB
records. See [`docker/inference/README.md`](docker/inference/README.md) for the
four commands, accepted shapes, bundle verification, and WSL path examples.

## Reproduce or Develop

Use Python 3.11 where possible:

```bash
pip install -r requirements.txt
python -m src.sqi_pipeline.run_all --verbose
python -m src.transformer_pipeline.run_all --run --train E31
```

For reproduction instructions, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md),
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md), and
[`docs/code_architecture.md`](docs/code_architecture.md) for commands, data,
outputs, and experiment lineage. Generated artifacts belong under `outputs/`;
the final report and executive summary remain separate.

For tests:

```bash
pip install -e ".[test]"
python -m pytest -q
```

## AI Tool Use

ChatGPT 5.5 was used to generate first drafts of code, format code, and polish
and compress the report language. I reviewed the generated material and accept
full responsibility for all submitted content.
