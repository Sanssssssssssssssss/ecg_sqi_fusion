from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from wfdb_qrs_kit import detect_both, detect_eplimited, detect_wqrs
from wfdb_qrs_kit.exceptions import DetectorNotFoundError, WFDBQRSKitError
from wfdb_qrs_kit.install import find_executable


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class PaperQRSUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperQRSExecutables:
    wqrs: Path
    eplimited: Path


def _cache_candidates(params: dict[str, Any], work_dir: Path) -> list[Path | None]:
    raw = params.get("wfdb_qrs_cache_dir")
    return [Path(str(raw)) if raw else None, work_dir / "tools", None]


def _resolve_exe(name: str, *, params: dict[str, Any], work_dir: Path) -> Path:
    explicit = params.get(f"{name}_exe")
    errors: list[str] = []
    for cache_dir in _cache_candidates(params, work_dir):
        try:
            return find_executable(name, explicit=explicit, cache_dir=cache_dir)
        except DetectorNotFoundError as exc:
            errors.append(str(exc))
    raise PaperQRSUnavailable(
        f"Missing paper QRS executable '{name}'. Install detectors with "
        "`python -m src.sqi_pipeline.qrs.setup_paper_detectors`, set "
        f"WFDB_QRS_KIT_{name.upper()}_EXE/SQI_{name.upper()}_EXE, or pass {name}_exe. "
        f"Last errors: {' | '.join(errors[-2:])}"
    )


def resolve_paper_qrs_executables(params: dict[str, Any], work_dir: Path) -> PaperQRSExecutables:
    return PaperQRSExecutables(
        wqrs=_resolve_exe("wqrs", params=params, work_dir=work_dir),
        eplimited=_resolve_exe("eplimited", params=params, work_dir=work_dir),
    )


def _as_samples(items: list[Any]) -> list[np.ndarray]:
    return [np.asarray(item.samples, dtype=int) for item in items]


def _run_kit(callable_name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except (DetectorNotFoundError, WFDBQRSKitError, OSError, RuntimeError) as exc:
        raise PaperQRSUnavailable(f"{callable_name} failed: {exc}") from exc


def run_paper_qrs_12lead(
    *,
    record_id: str,
    sig12: np.ndarray,
    fs: int,
    leads: list[str],
    executables: PaperQRSExecutables,
    work_dir: Path,
    eplimited_warmup_sec: float = 8.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run paper-era wqrs and EP Limited/Hamilton detectors via wfdb-qrs-kit.

    Returns:
      rpeaks_1 = wqrs annotations per lead
      rpeaks_2 = eplimited annotations per lead
    """
    if leads != LEADS_12:
        raise ValueError(f"paper QRS expects 12-lead order {LEADS_12}, got {leads}")
    results = _run_kit(
        "detect_both",
        detect_both,
        np.asarray(sig12, dtype=np.float64),
        fs=fs,
        leads=leads,
        axis=0,
        wqrs_executable=executables.wqrs,
        eplimited_executable=executables.eplimited,
        work_dir=work_dir,
        record_id=record_id,
        eplimited_warmup_sec=eplimited_warmup_sec,
    )
    return _as_samples(results["wqrs"]), _as_samples(results["eplimited"])


def run_wqrs_multilead(
    *,
    record_id: str,
    sig: np.ndarray,
    fs: int,
    leads: list[str],
    executable: Path,
    work_dir: Path,
) -> list[np.ndarray]:
    """Run wqrs via wfdb-qrs-kit on an arbitrary multi-lead record."""
    results = _run_kit(
        "detect_wqrs",
        detect_wqrs,
        np.asarray(sig, dtype=np.float64),
        fs=fs,
        leads=leads,
        axis=0,
        executable=executable,
        work_dir=work_dir,
        record_id=record_id,
    )
    return _as_samples(results)


def run_eplimited_multilead(
    *,
    record_id: str,
    sig: np.ndarray,
    fs: int,
    leads: list[str],
    executable: Path,
    work_dir: Path,
    eplimited_warmup_sec: float = 8.0,
) -> list[np.ndarray]:
    """Run EP Limited/Hamilton via wfdb-qrs-kit on an arbitrary multi-lead record."""
    results = _run_kit(
        "detect_eplimited",
        detect_eplimited,
        np.asarray(sig, dtype=np.float64),
        fs=fs,
        leads=leads,
        axis=0,
        executable=executable,
        work_dir=work_dir,
        record_id=record_id,
        warmup_sec=eplimited_warmup_sec,
    )
    return _as_samples(results)


def run_paper_qrs_multilead(
    *,
    record_id: str,
    sig: np.ndarray,
    fs: int,
    leads: list[str],
    executables: PaperQRSExecutables,
    work_dir: Path,
    eplimited_warmup_sec: float = 8.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run paper wqrs and EP Limited/Hamilton detectors on arbitrary lead sets."""
    wqrs = run_wqrs_multilead(
        record_id=record_id,
        sig=sig,
        fs=fs,
        leads=leads,
        executable=executables.wqrs,
        work_dir=work_dir,
    )
    epl = run_eplimited_multilead(
        record_id=record_id,
        sig=sig,
        fs=fs,
        leads=leads,
        executable=executables.eplimited,
        work_dir=work_dir,
        eplimited_warmup_sec=eplimited_warmup_sec,
    )
    return wqrs, epl
