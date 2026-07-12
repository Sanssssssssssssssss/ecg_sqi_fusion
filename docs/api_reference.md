# Python API

Only the stable inference surface and top-level classical orchestrator are
documented as public API. Experiment support modules remain implementation
details and may change without compatibility guarantees.

Browse the corresponding source modules on GitHub:
[inference core](https://github.com/Sanssssssssssssssss/ecg_sqi_fusion/blob/main/src/ecg_sqi_inference/core.py),
[predictors](https://github.com/Sanssssssssssssssss/ecg_sqi_fusion/blob/main/src/ecg_sqi_inference/models.py), and
[pipeline runner](https://github.com/Sanssssssssssssssss/ecg_sqi_fusion/blob/main/src/sqi_pipeline/runner.py).

## Inference data flow

::: src.ecg_sqi_inference.core.InputRecord

::: src.ecg_sqi_inference.core.SegmentPredictor

::: src.ecg_sqi_inference.core.read_record

::: src.ecg_sqi_inference.core.iter_input_files

::: src.ecg_sqi_inference.core.as_samples_by_lead

::: src.ecg_sqi_inference.core.resample_signal

::: src.ecg_sqi_inference.core.segment_signal

::: src.ecg_sqi_inference.core.predict_records

## Predictor implementations

::: src.ecg_sqi_inference.models.Conformer12Predictor

::: src.ecg_sqi_inference.models.Conformer1Predictor

::: src.ecg_sqi_inference.models.RBFSVMBundlePredictor

::: src.ecg_sqi_inference.models.get_predictor

::: src.ecg_sqi_inference.models.verify_inference_bundles

::: src.ecg_sqi_inference.models.feature_frame

## Classical pipeline orchestration

::: src.sqi_pipeline.config.SQIPipelineConfig

::: src.sqi_pipeline.runner.StepSpec

::: src.sqi_pipeline.runner.steps_for_profile

::: src.sqi_pipeline.runner.run_pipeline

For end-user invocation, prefer the [CLI](cli_reference.md). The Python
orchestration interface is intended for controlled programmatic runs.
