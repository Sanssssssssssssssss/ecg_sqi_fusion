from __future__ import annotations

from src.transformer_pipeline.data_v1_gapfill import train_check


def test_conformer_train_skips_without_cuda(monkeypatch, capsys):
    calls: list[list[str]] = []

    monkeypatch.delenv("ECG_ALLOW_CPU_CONFORMER_TRAIN", raising=False)
    monkeypatch.setattr(train_check, "cuda_available", lambda: False)
    monkeypatch.setattr(train_check, "run_or_print", lambda cmd, *, run: calls.append(cmd))

    train_check.main(model="E31", run=True)

    assert calls == []
    assert "SKIP_GPU_REQUIRED" in capsys.readouterr().out


def test_conformer_train_cpu_override_runs(monkeypatch):
    calls: list[tuple[list[str], bool]] = []

    monkeypatch.setenv("ECG_ALLOW_CPU_CONFORMER_TRAIN", "1")
    monkeypatch.setattr(train_check, "cuda_available", lambda: False)
    monkeypatch.setattr(train_check, "run_or_print", lambda cmd, *, run: calls.append((cmd, run)))

    train_check.main(model="E31", run=True)

    assert len(calls) == 1
    assert calls[0][1] is True
    assert "run_gm_mechanism_repair_suite.py" in calls[0][0][1]
