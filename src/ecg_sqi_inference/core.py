from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


MODEL_FS = 125
WINDOW_SEC = 10
WINDOW_SAMPLES = MODEL_FS * WINDOW_SEC


@dataclass(frozen=True)
class InputRecord:
    """ECG signal loaded from one input file.

    Attributes:
        record_id: File-stem identifier used in output rows.
        signal: One- or two-dimensional ECG sample array.
        input_path: Source file path.
    """

    record_id: str
    signal: np.ndarray
    input_path: Path


class SegmentPredictor(Protocol):
    """Interface implemented by segment-level inference models.

    Attributes:
        name: Stable model identifier written to outputs.
        n_leads: Number of ECG leads required by the model.
    """

    name: str
    n_leads: int

    def predict(self, segments: np.ndarray) -> pd.DataFrame:
        """Classify a batch of fixed-length ECG segments.

        Args:
            segments: Array shaped as batch, samples, and leads.

        Returns:
            One prediction row per input segment.
        """

        ...


def _npz_array(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    for key in ["signal", "ecg", "x", "X", "sig", "sig_125", "signals"]:
        if key in z:
            return np.asarray(z[key])
    for key in z.files:
        arr = np.asarray(z[key])
        if arr.ndim in {1, 2} and np.issubdtype(arr.dtype, np.number):
            return arr
    raise ValueError(f"{path}: no numeric 1D/2D ECG array found")


def _csv_array(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"{path}: CSV has no numeric columns")
    if "ecg" in numeric.columns:
        return numeric[["ecg"]].to_numpy()
    return numeric.to_numpy()


def read_record(path: Path) -> InputRecord:
    """Load one supported ECG file into a normalized record container.

    Args:
        path: NPZ, NPY, CSV, or WFDB header file to read.

    Returns:
        Loaded record with a float32 signal.

    Raises:
        ValueError: If the format or signal shape is unsupported.
    """

    suffix = path.suffix.lower()
    if suffix == ".npz":
        arr = _npz_array(path)
    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".csv":
        arr = _csv_array(path)
    elif suffix == ".hea":
        import wfdb

        arr, _ = wfdb.rdsamp(str(path.with_suffix("")))
    else:
        raise ValueError(f"{path}: unsupported input type")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim not in {1, 2}:
        raise ValueError(f"{path}: expected 1D or 2D ECG array, got shape {arr.shape}")
    return InputRecord(record_id=path.stem, signal=arr, input_path=path)


def iter_input_files(path: Path) -> list[Path]:
    """List supported ECG inputs from a file or directory tree.

    Args:
        path: Input file or directory to scan recursively.

    Returns:
        Supported files in deterministic path order.

    Raises:
        FileNotFoundError: If the input path does not exist.
    """

    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    exts = {".npz", ".npy", ".csv", ".hea"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def as_samples_by_lead(signal: np.ndarray, n_leads: int) -> np.ndarray:
    """Orient an ECG array as samples by the model's required leads.

    Args:
        signal: One- or two-dimensional ECG array.
        n_leads: Required model lead count, currently 1 or 12.

    Returns:
        Float32 array shaped as samples by leads.

    Raises:
        ValueError: If the signal cannot satisfy the requested lead count.
    """

    arr = np.asarray(signal, dtype=np.float32)
    if n_leads == 1:
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr.T
        raise ValueError(f"single-lead model requires 1 lead, got shape {arr.shape}")
    if n_leads == 12:
        if arr.ndim != 2:
            raise ValueError(f"12-lead model requires 2D ECG, got shape {arr.shape}")
        if arr.shape[1] == 12:
            return arr
        if arr.shape[0] == 12:
            return arr.T
        raise ValueError(f"12-lead model requires exactly 12 leads, got shape {arr.shape}")
    raise ValueError(f"unsupported lead count: {n_leads}")


def resample_signal(signal: np.ndarray, fs: float, target_fs: int = MODEL_FS) -> np.ndarray:
    """Resample a samples-by-leads ECG array to the model frequency.

    Args:
        signal: ECG array with time on axis zero.
        fs: Source sampling frequency in hertz.
        target_fs: Destination sampling frequency in hertz.

    Returns:
        Resampled float32 ECG array.

    Raises:
        ValueError: If the source frequency is not positive.
    """

    if fs <= 0:
        raise ValueError("--fs must be positive")
    if abs(float(fs) - float(target_fs)) < 1.0e-9:
        return signal.astype(np.float32, copy=False)
    ratio = Fraction(float(target_fs) / float(fs)).limit_denominator(1000)
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def segment_signal(signal: np.ndarray) -> tuple[np.ndarray, float]:
    """Split ECG samples into complete non-overlapping model windows.

    Args:
        signal: Samples-by-leads ECG at ``MODEL_FS``.

    Returns:
        Segment batch and discarded tail duration in seconds.
    """

    n = int(signal.shape[0] // WINDOW_SAMPLES)
    used = n * WINDOW_SAMPLES
    dropped = float((signal.shape[0] - used) / MODEL_FS)
    if n == 0:
        return np.empty((0, WINDOW_SAMPLES, signal.shape[1]), dtype=np.float32), dropped
    return signal[:used].reshape(n, WINDOW_SAMPLES, signal.shape[1]), dropped


def predict_records(
    *,
    input_path: Path,
    out_dir: Path,
    fs: float,
    predictor: SegmentPredictor,
) -> dict[str, object]:
    """Run one predictor over every supported input and write CSV/JSON outputs.

    Args:
        input_path: ECG file or directory tree to process.
        out_dir: Directory receiving per-record and combined outputs.
        fs: Sampling frequency shared by the input records.
        predictor: Segment classifier with a name and required lead count.

    Returns:
        Run summary when every discovered record succeeds.

    Raises:
        SystemExit: If one or more records fail; outputs and a failure summary
            are still written for reproducibility.
        ValueError: If recursive inputs contain duplicate record identifiers.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_resolved = out_dir.resolve()
    paths = [path for path in iter_input_files(input_path) if out_resolved not in path.resolve().parents]
    duplicate_ids = sorted(record_id for record_id, count in Counter(path.stem for path in paths).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate record_id(s): {', '.join(duplicate_ids)}")
    all_rows: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            rec = read_record(path)
            signal = as_samples_by_lead(rec.signal, predictor.n_leads)
            signal = resample_signal(signal, fs, MODEL_FS)
            segments, dropped = segment_signal(signal)
            if len(segments) == 0:
                raise ValueError("record is shorter than one 10s segment after resampling")
            pred = predictor.predict(segments)
            pred.insert(0, "record_id", rec.record_id)
            pred.insert(1, "segment_index", np.arange(len(pred), dtype=int))
            pred.insert(2, "start_sec", pred["segment_index"].astype(float) * WINDOW_SEC)
            pred.insert(3, "end_sec", pred["start_sec"] + WINDOW_SEC)
            pred["model"] = predictor.name
            pred["input_path"] = str(rec.input_path)
            out_csv = out_dir / f"{rec.record_id}_segments.csv"
            pred.to_csv(out_csv, index=False)
            all_rows.append(pred)
            records.append(
                {
                    "record_id": rec.record_id,
                    "input_path": str(path),
                    "segments": int(len(pred)),
                    "dropped_seconds": dropped,
                    "output": str(out_csv),
                }
            )
        except Exception as exc:
            errors.append({"input_path": str(path), "error": str(exc)})

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
    else:
        all_df = pd.DataFrame(
            columns=[
                "record_id",
                "segment_index",
                "start_sec",
                "end_sec",
                "model",
                "raw_class",
                "display_class",
                "input_path",
            ]
        )
    all_csv = out_dir / "all_segments.csv"
    all_df.to_csv(all_csv, index=False)
    summary: dict[str, object] = {
        "model": predictor.name,
        "input": str(input_path),
        "out": str(out_dir),
        "model_fs": MODEL_FS,
        "window_seconds": WINDOW_SEC,
        "records_ok": len(records),
        "records_failed": len(errors),
        "segments": int(len(all_df)),
        "all_segments": str(all_csv),
        "records": records,
        "errors": errors,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if errors:
        raise SystemExit(json.dumps(summary, indent=2))
    return summary
