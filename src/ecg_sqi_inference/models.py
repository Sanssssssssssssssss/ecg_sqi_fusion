from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.utils.paths import project_root

from .core import MODEL_FS


DISPLAY_BINARY = {"acceptable": "usable", "unacceptable": "unusable"}
MODEL_NAMES = ("12lead-conformer", "singlelead-conformer", "12lead-rbfsvm", "singlelead-rbfsvm")


@dataclass
class Conformer12Predictor:
    """Load and run the frozen 12-lead Conformer checkpoint.

    Attributes:
        ckpt_dir: Directory containing ``best_model.pt``.
        device: Requested Torch device, ``cpu`` or ``cuda``.
        name: Stable public model identifier.
        n_leads: Required ECG lead count.

    Raises:
        FileNotFoundError: If the checkpoint is absent.
        ValueError: If normalization metadata are absent or incompatible.
    """

    ckpt_dir: Path
    device: str = "cpu"
    name: str = "12lead-conformer"
    n_leads: int = 12

    def __post_init__(self) -> None:
        from src.supplemental_transformer_experiments.sqi12_gapfill.run import (
            LeadWiseSharedConformer,
            _load_torch,
            _train_config_from_checkpoint,
        )

        dev = torch.device("cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu")
        ckpt_path = self.ckpt_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        ckpt = _load_torch(ckpt_path, dev)
        cfg = _train_config_from_checkpoint(dict(ckpt.get("config", {})), device=str(dev))
        norm = ckpt.get("normalization")
        if not isinstance(norm, dict):
            raise ValueError(f"{ckpt_path}: checkpoint has no normalization block")
        factor_dim = len(norm.get("factors", {}).get("columns", [])) or 7
        model = LeadWiseSharedConformer(cfg.width, cfg.layers, cfg.heads, factor_dim).to(dev)
        state = ckpt.get("model_state_dict", ckpt.get("model_state"))
        if state is None:
            raise KeyError(f"{ckpt_path}: no model_state_dict/model_state")
        model.load_state_dict(state, strict=True)
        model.eval()
        self._device = dev
        self._model = model
        self._mean = np.asarray(norm["mean_per_lead"], dtype=np.float32).reshape(1, 1, 12)
        self._std = np.maximum(np.asarray(norm["std_per_lead"], dtype=np.float32).reshape(1, 1, 12), 1e-6)

    def predict(self, segments: np.ndarray) -> pd.DataFrame:
        """Classify 12-lead ECG segments with the Conformer.

        Args:
            segments: Float-compatible array shaped ``(batch, 1250, 12)``.

        Returns:
            Class labels and binary probabilities for each segment.

        Example:
            >>> output = predictor.predict(np.zeros((1, 1250, 12), dtype=np.float32))
            >>> output.shape[0]
            1
        """

        x = ((segments.astype(np.float32) - self._mean) / self._std).transpose(0, 2, 1)
        with torch.no_grad():
            out = self._model(torch.from_numpy(x).to(self._device))
            prob = torch.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy()
        raw = np.where(prob >= 0.5, "acceptable", "unacceptable")
        return pd.DataFrame(
            {
                "raw_class": raw,
                "display_class": [DISPLAY_BINARY[x] for x in raw],
                "prob_unacceptable": 1.0 - prob,
                "prob_acceptable": prob,
            }
        )


@dataclass
class Conformer1Predictor:
    """Load and run the frozen single-lead BUT Conformer.

    Attributes:
        bundle_dir: Directory containing the runtime profile.
        device: Requested Torch device, ``cpu`` or ``cuda``.
        name: Stable public model identifier.
        n_leads: Required ECG lead count.

    Raises:
        FileNotFoundError: If the profile or referenced checkpoint is absent.
        ValueError: If the checkpoint state is incompatible with the model.
    """

    bundle_dir: Path
    device: str = "cpu"
    name: str = "singlelead-conformer"
    n_leads: int = 1

    def __post_init__(self) -> None:
        from src.transformer_pipeline.data_v1_gapfill.support import run_gm_mechanism_repair_suite as gm

        profile_path = self.bundle_dir / "profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        root = project_root()
        ckpt_path = root / str(profile["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        dev = torch.device("cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        cfg = dict(ckpt["candidate_config"])
        gm.ACTIVE_CFG = cfg
        model = gm.GMMechanismConformer(
            in_ch=8,
            factor_dim=len(ckpt["factor_columns"]),
            width=int(cfg["width"]),
            layers=int(cfg["layers"]),
            heads=int(cfg["heads"]),
            dropout=float(cfg.get("dropout", 0.08)),
        ).to(dev)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        obsolete = [name for name in unexpected if name.startswith("query_class_fusion_head.")]
        if missing or len(obsolete) != len(unexpected):
            raise ValueError(f"{ckpt_path}: incompatible model state; missing={missing}, unexpected={unexpected}")
        model.eval()
        stats = profile["channel_stats"]
        self._channel_stats = gm.EVT.DUAL.ChannelStats(
            global_mean=float(stats["global_mean"]),
            global_std=float(stats["global_std"]),
        )
        self._cfg = cfg
        self._device = dev
        self._gm = gm
        self._model = model

    def predict(self, segments: np.ndarray) -> pd.DataFrame:
        """Classify single-lead ECG segments as good, medium, or bad.

        Args:
            segments: Float-compatible array shaped ``(batch, 1250, 1)``.

        Returns:
            Class labels and three-class probabilities for each segment.

        Example:
            >>> output = predictor.predict(np.zeros((1, 1250, 1), dtype=np.float32))
            >>> output.shape[0]
            1
        """

        raw = np.asarray(segments, dtype=np.float32)[:, :, 0]
        channels = self._gm.EVT.DUAL.make_dualview_channels(raw, self._channel_stats)
        self._gm.ACTIVE_CFG = self._cfg
        with torch.no_grad():
            prob = self._model(torch.from_numpy(channels).to(self._device))["probs"].detach().cpu().numpy()
        classes = np.asarray(["good", "medium", "bad"])
        labels = classes[np.argmax(prob, axis=1)]
        return pd.DataFrame(
            {
                "raw_class": labels,
                "display_class": labels,
                "prob_good": prob[:, 0],
                "prob_medium": prob[:, 1],
                "prob_bad": prob[:, 2],
            }
        )


@dataclass
class RBFSVMBundlePredictor:
    """Run a packaged binary or three-class RBF-SVM.

    Attributes:
        bundle_dir: Directory containing ``profile.json`` and ``model.joblib``.
        name: Stable public model identifier.
        n_leads: Required ECG lead count.

    Raises:
        FileNotFoundError: If a bundle file is absent.
        ValueError: If the serialized estimator is incompatible with its profile.
    """

    bundle_dir: Path
    name: str
    n_leads: int

    def __post_init__(self) -> None:
        import joblib

        self._profile = json.loads((self.bundle_dir / "profile.json").read_text(encoding="utf-8"))
        self._feature_columns = list(self._profile["feature_columns"])
        self._classes = list(self._profile["classes"])
        self._model = joblib.load(self.bundle_dir / "model.joblib")["estimator"]

    def predict(self, segments: np.ndarray) -> pd.DataFrame:
        """Classify ECG segments using profile-compatible SQI features.

        Args:
            segments: Float-compatible array shaped ``(batch, 1250, n_leads)``.

        Returns:
            One class label and probability row per segment.

        Raises:
            RuntimeError: If required 12-lead QRS executables are unavailable.

        Example:
            >>> output = predictor.predict(np.zeros((1, 1250, predictor.n_leads)))
            >>> output.shape[0]
            1
        """

        features = feature_frame(segments, self.n_leads, self._profile)
        probability = self._model.predict_proba(features[self._feature_columns].to_numpy(dtype=np.float64))
        if self._classes == ["unacceptable", "acceptable"]:
            acceptable = probability[:, 1]
            poor = 1.0 - acceptable
            raw = np.where(poor >= float(self._profile["poor_threshold"]), "unacceptable", "acceptable")
            return pd.DataFrame(
                {
                    "raw_class": raw,
                    "display_class": [DISPLAY_BINARY[value] for value in raw],
                    "prob_unacceptable": poor,
                    "prob_acceptable": acceptable,
                }
            )
        labels = np.asarray(self._classes)[np.argmax(probability, axis=1)]
        return pd.DataFrame(
            {
                "raw_class": labels,
                "display_class": labels,
                **{f"prob_{name}": probability[:, index] for index, name in enumerate(self._classes)},
            }
        )

def get_predictor(model: str, *, device: str = "cpu") -> Any:
    """Construct a named predictor from the repository's inference assets.

    Args:
        model: Supported public model identifier.
        device: Requested Conformer device, ``cpu`` or ``cuda``.

    Returns:
        Initialized predictor for the requested model.

    Raises:
        ValueError: If the model identifier is unknown.
        FileNotFoundError: If a required checkpoint or bundle file is absent.

    Example:
        >>> predictor = get_predictor("singlelead-rbfsvm")
        >>> (predictor.name, predictor.n_leads)
        ('singlelead-rbfsvm', 1)
    """

    root = project_root()
    if model == "12lead-conformer":
        return Conformer12Predictor(root / "pretrained" / "chapter4" / "seta_e31_leadwise_shared", device=device)
    if model == "singlelead-conformer":
        return Conformer1Predictor(root / "pretrained" / "inference" / "singlelead-conformer", device=device)
    if model == "12lead-rbfsvm":
        return RBFSVMBundlePredictor(root / "pretrained" / "inference" / model, model, 12)
    if model == "singlelead-rbfsvm":
        return RBFSVMBundlePredictor(root / "pretrained" / "inference" / model, model, 1)
    raise ValueError(f"unknown model: {model}")


def verify_inference_bundles() -> dict[str, Any]:
    """Verify every shipped inference artifact against its frozen SHA-256.

    Returns:
        Validation summary containing the verified model names and artifacts.

    Raises:
        FileNotFoundError: If the manifest, profile, or model artifact is absent.
        ValueError: If an artifact hash does not match the manifest.

    Example:
        >>> verify_inference_bundles()["status"]
        'ok'
    """

    root = project_root()
    manifest_path = root / "pretrained" / "inference" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified: dict[str, list[str]] = {}
    for model, spec in manifest["models"].items():
        profile = spec.get("profile")
        if profile and not (root / profile).is_file():
            raise FileNotFoundError(root / profile)
        paths: list[str] = []
        for relative, expected in spec["artifacts"].items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"hash mismatch: {relative}")
            paths.append(relative)
        verified[model] = paths
    return {"status": "ok", "schema": manifest["schema"], "models": verified}


def _norm_values(row: dict[str, float], stats: dict[str, Any]) -> dict[str, float]:
    out = dict(row)
    med = stats.get("median_train", {})
    sd = stats.get("std_train", {})
    for col in stats.get("columns", []):
        if col in out:
            out[col] = (float(out[col]) - float(med[col])) / max(float(sd[col]), float(stats.get("std_eps", 1e-8)))
    return out


def feature_frame(segments: np.ndarray, n_leads: int, profile: dict[str, Any]) -> pd.DataFrame:
    """Compute the profile-compatible SQI feature table for ECG segments.

    Args:
        segments: ECG array shaped batch, samples, and leads.
        n_leads: Feature pipeline lead count, either 1 or 12.
        profile: Bundle profile containing normalization statistics.

    Returns:
        One normalized SQI feature row per segment.

    Raises:
        RuntimeError: If required 12-lead QRS executables are unavailable.

    Example:
        ``feature_frame`` is normally called by ``RBFSVMBundlePredictor`` so
        that the bundle profile, feature order, and normalization stay aligned.
    """

    if n_leads == 1:
        from src.supplemental_transformer_experiments.but_sqi_baseline.run import _compute_one

        rows = []
        for i, x in enumerate(segments[:, :, 0]):
            _, single, _, _ = _compute_one((f"seg{i}", 0, i, x))
            rows.append(_norm_values(single, profile["norm_stats"]))
        return pd.DataFrame(rows)

    from src.sqi_pipeline.diagnostics.paper_extra_experiments import _normalize_record84_row, _record84_from_qrs
    from src.sqi_pipeline.qrs import setup_paper_detectors
    from src.sqi_pipeline.qrs.paper_detectors import resolve_paper_qrs_executables, run_paper_qrs_12lead
    from src.sqi_pipeline.features.make_record84 import LEADS_12

    work = project_root() / "tmp" / "inference_qrs"
    setup_paper_detectors.run(work / "tools", download_sources=False, require_executables=True)
    executables = resolve_paper_qrs_executables({}, work)
    rows = []
    for i, sig12 in enumerate(segments):
        wqrs, epl = run_paper_qrs_12lead(
            record_id=f"infer_{i}",
            sig12=sig12,
            fs=MODEL_FS,
            leads=LEADS_12,
            executables=executables,
            work_dir=work,
        )
        rows.append(_normalize_record84_row(_record84_from_qrs(sig12, wqrs, epl), profile["norm_stats"]))
    return pd.DataFrame(rows)


def export_inference_bundles(out_dir: Path | None = None) -> dict[str, str]:
    """Package available trained baselines as standalone inference bundles.

    Args:
        out_dir: Destination directory, or the repository default when omitted.

    Returns:
        Mapping of exported model names to bundle directories; models whose
        source artifacts are absent are omitted.

    Note:
        This is a maintainer utility. End users should consume the frozen
        bundles already listed in ``pretrained/inference/manifest.json``.
    """

    root = project_root()
    out = out_dir or (root / "pretrained" / "inference")
    out.mkdir(parents=True, exist_ok=True)
    exported: dict[str, str] = {}

    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    frozen = root / "outputs" / "transformer" / "supplemental" / "chapter4_evidence_frozen_final" / "revision_v2"
    seta = frozen / "seta" / "frozen_exact30_models"
    seta_model = seta / "rbf_svm_selected5" / "model.pkl"
    seta_norm = seta / "sqi" / "features" / "norm_stats_seed0.json"
    seta_metrics = seta / "baseline_runs.json"
    if seta_model.exists() and seta_norm.exists() and seta_metrics.exists():
        with seta_model.open("rb") as handle:
            fitted = pickle.load(handle)
        dst = out / "12lead-rbfsvm"
        dst.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": fitted.best_estimator_}, dst / "model.joblib")
        profile = {
            "model": "12lead-rbfsvm",
            "model_kind": "rbf_svm",
            "n_leads": 12,
            "fs": MODEL_FS,
            "classes": ["unacceptable", "acceptable"],
            "feature_columns": [
                f"{lead}__{sqi}"
                for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
                for sqi in ("bSQI", "basSQI", "kSQI", "sSQI", "fSQI")
            ],
            "poor_threshold": json.loads(seta_metrics.read_text(encoding="utf-8"))["rbf_svm"]["threshold"],
            "norm_stats": json.loads(seta_norm.read_text(encoding="utf-8")),
        }
        (dst / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        exported[profile["model"]] = str(dst)

    but_atlas = frozen / "but" / "frozen_anchor_exact30_models" / "split" / "fold0" / "original_region_atlas.csv"
    if but_atlas.exists():
        frame = pd.read_csv(but_atlas, low_memory=False)
        source_columns = [f"sqi_{name}" for name in ("iSQI", "bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI")]
        train = frame["split"].eq("train")
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma=0.14, probability=True, random_state=0),
        ).fit(frame.loc[train, source_columns].to_numpy(float), frame.loc[train, "class_name"].map({"good": 0, "medium": 1, "bad": 2}))
        dst = out / "singlelead-rbfsvm"
        dst.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": estimator}, dst / "model.joblib")
        profile = {
            "model": "singlelead-rbfsvm",
            "model_kind": "rbf_svm",
            "n_leads": 1,
            "fs": MODEL_FS,
            "classes": ["good", "medium", "bad"],
            "feature_columns": [column.replace("sqi_", "single__") for column in source_columns],
            "norm_stats": {"columns": []},
        }
        (dst / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        exported[profile["model"]] = str(dst)

    return exported
