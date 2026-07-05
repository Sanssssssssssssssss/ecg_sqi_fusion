from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpecFromSubplotSpec

from src.supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi
from src.sqi_pipeline.models.lm_mlp import LMConfig, LMMLP
from src.transformer_pipeline.data_v1_gapfill.support import run_event_factorized_sqi_conformer as EVT

from .but_boundary_audit import _load_frame, _split_xy
from .but_models import LM_MLP_J, _ensure_but_sqi
from .but_query_patching import (
    SPLIT_ROOT,
    _align_clean_predictions,
    _ensure_boundary_records,
    _forward,
    _load_model,
    _rescue_pairs,
)
from .common import ROOT, Paths, dry, ensure_dirs, rel, table_to_md, write_json
from .figures import _save


FS = 125
SEVERITIES = [0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]
PERTURBATIONS = ["hf_burst", "emg_noise_floor", "reset_spike"]
SQI_L2_MAX = 2.5
KEY_THRESH = {"iSQI": 0.05, "bSQI": 0.05, "pSQI": 0.10, "basSQI": 0.08}
LABELS = ["good", "medium", "bad"]


@dataclass
class SqiContext:
    bp: but_sqi.Paths
    cols: list[str]
    raw_by_id: pd.DataFrame
    norm_stats: dict[str, Any]
    mlp: "MlpScorer"


class MlpScorer:
    def __init__(self, xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, *, device: str):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.models: list[LMMLP] = []
        cfg = LMConfig()
        xtr_t = torch.tensor(xtr, device=self.device, dtype=self.dtype)
        xva_t = torch.tensor(xva, device=self.device, dtype=self.dtype)
        for cls in range(3):
            model = LMMLP(J=LM_MLP_J, D=xtr.shape[1], device=self.device, dtype=self.dtype, seed=cls)
            model.fit_lm(
                X_train=xtr_t,
                y_train=torch.tensor((ytr == cls).astype(np.float64), device=self.device, dtype=self.dtype),
                cfg=cfg,
                X_val=xva_t,
                y_val=torch.tensor((yva == cls).astype(np.float64), device=self.device, dtype=self.dtype),
                model_select_metric="val_acc",
                patience=15,
                threshold=0.5,
            )
            self.models.append(model)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(np.asarray(x, dtype=np.float64), device=self.device, dtype=self.dtype)
        return np.stack([m.predict_proba(x_t).astype(np.float64) for m in self.models], axis=1)


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    return np.convolve(np.asarray(x, dtype=np.float64), np.ones(win, dtype=np.float64) / win, mode="same")


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    scale = (np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) / 1.349
    scale = scale if scale > 1e-6 else max(float(np.nanstd(x)), 1.0)
    return (x - np.nanmedian(x)) / scale


def _robust_scale(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=np.float64).reshape(-1)
    return max(float((np.nanpercentile(z, 95) - np.nanpercentile(z, 5)) / 4.0), float(np.nanstd(z)), 1e-4)


def _hann_window(n: int) -> np.ndarray:
    return np.hanning(max(4, int(n))).astype(np.float64)


def _window(row_i: int, kind: str, n: int, win: int) -> tuple[int, int]:
    rng = np.random.default_rng(20260703 + row_i * 17 + sum(ord(c) for c in kind))
    lo = FS
    hi = max(lo + 1, n - win - FS)
    start = int(rng.integers(lo, hi))
    return start, min(n, start + win)


def _hf_burst(x: np.ndarray, row_i: int, severity: float) -> tuple[np.ndarray, tuple[int, int]]:
    start, stop = _window(row_i, "hf_burst", len(x), int(0.50 * FS))
    rng = np.random.default_rng(1100 + row_i)
    noise = rng.normal(size=stop - start)
    noise = noise - _rolling_mean(noise, 9)
    out = np.asarray(x, dtype=np.float64).copy()
    out[start:stop] += severity * 0.14 * _robust_scale(x) * noise * _hann_window(stop - start)
    return out.astype(np.float32), (start, stop)


def _emg_noise_floor(x: np.ndarray, row_i: int, severity: float) -> tuple[np.ndarray, tuple[int, int]]:
    start, stop = _window(row_i, "emg_noise_floor", len(x), int(0.95 * FS))
    rng = np.random.default_rng(2100 + row_i)
    noise = rng.normal(size=stop - start)
    noise = noise - _rolling_mean(noise, 15)
    envelope = 0.65 + 0.35 * rng.random(stop - start)
    out = np.asarray(x, dtype=np.float64).copy()
    out[start:stop] += severity * 0.10 * _robust_scale(x) * noise * envelope * _hann_window(stop - start)
    return out.astype(np.float32), (start, stop)


def _reset_spike(x: np.ndarray, row_i: int, severity: float) -> tuple[np.ndarray, tuple[int, int]]:
    start, stop = _window(row_i, "reset_spike", len(x), int(0.80 * FS))
    rng = np.random.default_rng(3100 + row_i)
    out = np.asarray(x, dtype=np.float64).copy()
    scale = _robust_scale(x)
    for center in rng.choice(np.arange(start + 8, stop - 8), size=4, replace=False):
        width = int(rng.integers(2, 6))
        lo = int(center - width)
        hi = int(center + width + 1)
        shape = 1.0 - np.abs(np.arange(lo, hi) - center) / max(width, 1)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        out[lo:hi] += sign * severity * 0.40 * scale * shape
    return out.astype(np.float32), (start, stop)


def _perturb(x: np.ndarray, kind: str, row_i: int, severity: float) -> tuple[np.ndarray, tuple[int, int]]:
    if kind == "hf_burst":
        return _hf_burst(x, row_i, severity)
    if kind == "emg_noise_floor":
        return _emg_noise_floor(x, row_i, severity)
    if kind == "reset_spike":
        return _reset_spike(x, row_i, severity)
    raise ValueError(kind)


def _load_protocol_signals() -> np.ndarray:
    info = json.loads((SPLIT_ROOT / "fold0" / "source_protocol.json").read_text(encoding="utf-8"))
    source = Path(info["source_protocol"])
    x = np.load(source / "signals.npz", allow_pickle=True)["X"].astype(np.float32)
    return x[:, 0, :] if x.ndim == 3 else x


def _raw_wave(signals: np.ndarray, test_ds: Any, pos: int) -> np.ndarray:
    return signals[int(test_ds.frame.iloc[int(pos)]["idx"])].astype(np.float32)


def _fit_sqi_context(paths: Paths, *, force: bool, device: str) -> SqiContext:
    _ensure_but_sqi(paths, force=force, device=device)
    bp = but_sqi.Paths(paths.but / "sqi_baseline")
    df, cols = _load_frame(bp)
    _, xtr, ytr = _split_xy(df, cols, "train")
    _, xva, yva = _split_xy(df, cols, "val")
    raw = pd.read_parquet(bp.record84_parquet)
    raw["record_id"] = raw["record_id"].astype(str)
    stats = json.loads(bp.norm_stats_json.read_text(encoding="utf-8"))
    return SqiContext(bp=bp, cols=cols, raw_by_id=raw.set_index("record_id"), norm_stats=stats, mlp=MlpScorer(xtr, ytr, xva, yva, device=device))


def _compute_sqi(record_id: str, wave: np.ndarray, y: int) -> tuple[dict[str, float], dict[str, Any]]:
    feat, _, qrs = but_sqi._compute_one((record_id, int(y), 0, np.asarray(wave, dtype=np.float32)))
    return feat, qrs


def _normalize(feat: dict[str, Any], ctx: SqiContext) -> np.ndarray:
    vals = {k: feat.get(k, np.nan) for k in ctx.cols}
    for col in ctx.norm_stats.get("columns", []):
        if col in vals:
            med = float(ctx.norm_stats["median_train"][col])
            std = max(float(ctx.norm_stats["std_train"][col]), 1e-8)
            vals[col] = (float(vals[col]) - med) / std
    arr = np.asarray([float(vals[c]) for c in ctx.cols], dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _key_delta(clean: dict[str, Any], cand: dict[str, Any], name: str) -> float:
    return abs(float(cand[f"I__{name}"]) - float(clean[f"I__{name}"]))


def _accepted(clean_feat: dict[str, Any], clean_qrs: dict[str, Any], cand_feat: dict[str, Any], cand_qrs: dict[str, Any], sqi_distance: float) -> tuple[bool, dict[str, float]]:
    deltas = {name: _key_delta(clean_feat, cand_feat, name) for name in KEY_THRESH}
    qrs_same = int(clean_qrs["n_xqrs"]) == int(cand_qrs["n_xqrs"]) and int(clean_qrs["n_gqrs"]) == int(cand_qrs["n_gqrs"])
    ok = qrs_same and sqi_distance <= SQI_L2_MAX and all(deltas[k] <= KEY_THRESH[k] for k in KEY_THRESH)
    deltas["sqi_distance"] = float(sqi_distance)
    deltas["qrs_same"] = float(qrs_same)
    return bool(ok), deltas


@torch.no_grad()
def _score_conformer(model: torch.nn.Module, waves: list[np.ndarray], channel_stats: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    x = EVT.DUAL.make_dualview_channels(np.asarray(waves, dtype=np.float32), channel_stats).astype(np.float32)
    out = _forward(model, torch.from_numpy(x).to(device=device, dtype=torch.float32))
    logits = out["logits"].detach().cpu().numpy().astype(np.float64)
    probs = out["probs"].detach().cpu().numpy().astype(np.float64)
    return logits, probs


def _gm_margin(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    eps = 1e-8
    return np.log(prob[:, 1] + eps) - np.log(prob[:, 0] + eps)


def _validate_test_scope(records: pd.DataFrame) -> pd.DataFrame:
    test = records.loc[records["split"].astype(str).eq("test")].copy()
    counts = test["true_class"].astype(str).value_counts().to_dict()
    generated = pd.to_numeric(test["v116_generated"], errors="coerce").fillna(0).astype(int)
    if len(test) != 1849 or counts != {"good": 1053, "medium": 632, "bad": 164} or int(generated.sum()) != 0:
        raise RuntimeError(f"bad BUT test scope: n={len(test)}, counts={counts}, generated={int(generated.sum())}")
    return test


def _degradation(paths: Paths, records: pd.DataFrame, model: torch.nn.Module, test_ds: Any, signals: np.ndarray, ctx: SqiContext, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_to_pos = {int(r.source_idx): i for i, r in test_ds.frame.reset_index(drop=True).iterrows()}
    good = records.loc[
        records["true_class"].eq("good") & pd.to_numeric(records["source_idx"], errors="coerce").astype(int).isin(source_to_pos)
    ].sort_values("record_id").head(40)
    detail = pd.to_numeric(good["l_detail"], errors="coerce")
    example_record = str(good.iloc[int(np.nanargmin(np.abs(detail.to_numpy(float) - float(detail.median()))))]["record_id"])
    rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for row_i, row in enumerate(good.itertuples(index=False)):
        record_id = str(row.record_id)
        pos = source_to_pos[int(row.source_idx)]
        clean_wave = _raw_wave(signals, test_ds, pos)
        if record_id == example_record:
            for kind in PERTURBATIONS:
                sev_waves = [(0.0, clean_wave, _perturb(clean_wave, kind, row_i, 1.6)[1])]
                sev_waves.extend((sev, *_perturb(clean_wave, kind, row_i, sev)) for sev in [0.8, 1.6])
                _, _, (base_start, base_stop) = sev_waves[-1]
                lo = max(0, int(base_start - 1.00 * FS))
                hi = min(len(clean_wave), int(base_stop + 2.25 * FS))
                for sev, wave, (start, stop) in sev_waves:
                    for j in range(lo, hi):
                        example_rows.append(
                            {
                                "perturbation": kind,
                                "severity": float(sev),
                                "time_s": float((j - lo) / FS),
                                "value": float(wave[j]),
                                "window_start_s": float((start - lo) / FS),
                                "window_stop_s": float((stop - lo) / FS),
                            }
                        )
        clean_feat, clean_qrs = _compute_sqi(f"{record_id}_clean", clean_wave, 1)
        clean_norm = _normalize(clean_feat, ctx)
        clean_logits, clean_probs = _score_conformer(model, [clean_wave], test_ds.channel_stats, device)
        clean_mlp = ctx.mlp.predict(clean_norm[None, :])
        for kind in PERTURBATIONS:
            for severity in SEVERITIES:
                cand_wave, (start, stop) = _perturb(clean_wave, kind, row_i, severity)
                cand_feat, cand_qrs = _compute_sqi(f"{record_id}_{kind}_{severity:.2f}", cand_wave, 1)
                cand_norm = _normalize(cand_feat, ctx)
                sqi_distance = float(np.linalg.norm(cand_norm - clean_norm))
                ok, gate = _accepted(clean_feat, clean_qrs, cand_feat, cand_qrs, sqi_distance)
                cand_logits, cand_probs = _score_conformer(model, [cand_wave], test_ds.channel_stats, device)
                cand_mlp = ctx.mlp.predict(cand_norm[None, :])
                rows.append(
                    {
                        "record_id": record_id,
                        "perturbation": kind,
                        "severity": float(severity),
                        "window_start_s": start / FS,
                        "window_stop_s": stop / FS,
                        "accepted": int(ok),
                        "sqi_distance": gate["sqi_distance"],
                        "delta_iSQI": gate["iSQI"],
                        "delta_bSQI": gate["bSQI"],
                        "delta_pSQI": gate["pSQI"],
                        "delta_basSQI": gate["basSQI"],
                        "qrs_same": int(gate["qrs_same"]),
                        "delta_conformer_medium_logit": float(cand_logits[0, 1] - clean_logits[0, 1]),
                        "delta_conformer_bad_logit": float(cand_logits[0, 2] - clean_logits[0, 2]),
                        "delta_mlp_gm_margin": float(_gm_margin(cand_mlp)[0] - _gm_margin(clean_mlp)[0]),
                        "clean_conformer_pred": LABELS[int(clean_probs[0].argmax())],
                        "cand_conformer_pred": LABELS[int(cand_probs[0].argmax())],
                    }
                )
    raw = pd.DataFrame(rows)
    pd.DataFrame(example_rows).to_csv(paths.source_data / "fig_M8_wave_examples.csv", index=False)
    summary = (
        raw.groupby(["perturbation", "severity"], as_index=False)
        .agg(
            n=("record_id", "count"),
            accepted_n=("accepted", "sum"),
            accepted_rate=("accepted", "mean"),
            median_sqi_distance=("sqi_distance", "median"),
            mean_delta_conformer_medium_logit=("delta_conformer_medium_logit", lambda s: float(raw.loc[s.index][raw.loc[s.index, "accepted"].eq(1)]["delta_conformer_medium_logit"].mean())),
            mean_delta_conformer_bad_logit=("delta_conformer_bad_logit", lambda s: float(raw.loc[s.index][raw.loc[s.index, "accepted"].eq(1)]["delta_conformer_bad_logit"].mean())),
            mean_delta_mlp_gm_margin=("delta_mlp_gm_margin", lambda s: float(raw.loc[s.index][raw.loc[s.index, "accepted"].eq(1)]["delta_mlp_gm_margin"].mean())),
        )
        .fillna(0.0)
    )
    raw.to_csv(paths.tables / "but_sqi_locked_local_degradation_raw.csv", index=False)
    summary.to_csv(paths.tables / "but_sqi_locked_local_degradation_summary.csv", index=False)
    return raw, summary


def _local_score(good: np.ndarray, medium: np.ndarray) -> np.ndarray:
    zg = _robust_z(good)
    zm = _robust_z(medium)
    bg = _rolling_mean(zg, 251)
    bm = _rolling_mean(zm, 251)
    dg = _rolling_mean(np.abs(zg - bg), 31)
    dm = _rolling_mean(np.abs(zm - bm), 31)
    parts = [np.abs(dm - dg), np.abs(bm - bg), _rolling_mean(np.abs(zm - zg), 31)]
    zparts = [_robust_z(p) for p in parts]
    return np.sum(zparts, axis=0)


def _top_window(score: np.ndarray, win: int = FS) -> tuple[int, int]:
    valid = np.convolve(np.asarray(score, dtype=np.float64), np.ones(win) / win, mode="valid")
    valid[: FS // 2] = -np.inf
    valid[-FS // 2 :] = -np.inf
    start = int(np.nanargmax(valid))
    return start, start + win


def _random_window(pair_id: int, n: int, win: int = FS) -> tuple[int, int]:
    rng = np.random.default_rng(20260704 + int(pair_id))
    start = int(rng.integers(FS // 2, max(FS // 2 + 1, n - win - FS // 2)))
    return start, start + win


def _crossfade_transplant(dst: np.ndarray, src: np.ndarray, start: int, stop: int) -> np.ndarray:
    out = np.asarray(dst, dtype=np.float64).copy()
    donor = np.asarray(src, dtype=np.float64)
    fade = min(16, max(2, (stop - start) // 4))
    out[start + fade : stop - fade] = donor[start + fade : stop - fade]
    ramp = np.linspace(0, 1, fade, endpoint=False)
    out[start : start + fade] = (1 - ramp) * out[start : start + fade] + ramp * donor[start : start + fade]
    ramp = np.linspace(1, 0, fade, endpoint=False)
    out[stop - fade : stop] = (1 - ramp) * out[stop - fade : stop] + ramp * donor[stop - fade : stop]
    return out.astype(np.float32)


def _display_clip_fraction(waves: list[np.ndarray], start: int, stop: int) -> float:
    lo = max(0, int(start) - 2 * FS)
    hi = min(1250, int(stop) + 2 * FS)
    vals: list[np.ndarray] = []
    for wave in waves:
        vals.append(_robust_z(np.asarray(wave, dtype=np.float64)[lo:hi]))
    z = np.concatenate(vals) if vals else np.zeros(1)
    return float(np.mean(np.abs(z) > 2.5))


def _transplant_rows(paths: Paths, records: pd.DataFrame, model: torch.nn.Module, test_ds: Any, signals: np.ndarray, ctx: SqiContext, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rescue = _rescue_pairs(records, test_ds)
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for pair_id, rec in rescue.head(60).iterrows():
        nxt = rescue.iloc[(pair_id + 1) % len(rescue)]
        good = _raw_wave(signals, test_ds, int(rec["good_pos"]))
        medium = _raw_wave(signals, test_ds, int(rec["medium_pos"]))
        good_same = _raw_wave(signals, test_ds, int(nxt["good_pos"]))
        medium_same = _raw_wave(signals, test_ds, int(nxt["medium_pos"]))
        top = _top_window(_local_score(good, medium))
        random = _random_window(pair_id, len(good))
        specs = [
            ("medium_into_good_top", "top-local", good, medium, top, "good"),
            ("medium_into_good_random", "random", good, medium, random, "good"),
            ("good_into_good_same_class", "same-class", good, good_same, top, "good"),
            ("good_into_medium_top", "top-local", medium, good, top, "medium"),
            ("good_into_medium_random", "random", medium, good, random, "medium"),
            ("medium_into_medium_same_class", "same-class", medium, medium_same, top, "medium"),
        ]
        clean_feats: dict[str, tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = {}
        for base_name, base_wave, y in [("good", good, 1), ("medium", medium, -1)]:
            feat, qrs = _compute_sqi(f"pair{pair_id}_{base_name}_clean", base_wave, y)
            norm = _normalize(feat, ctx)
            logits, probs = _score_conformer(model, [base_wave], test_ds.channel_stats, device)
            mlp = ctx.mlp.predict(norm[None, :])
            clean_feats[base_name] = (feat, qrs, norm, logits, mlp)
        for direction, control, recipient, donor, (start, stop), base_name in specs:
            cand = _crossfade_transplant(recipient, donor, start, stop)
            y = 1 if base_name == "good" else -1
            cand_feat, cand_qrs = _compute_sqi(f"pair{pair_id}_{direction}", cand, y)
            cand_norm = _normalize(cand_feat, ctx)
            clean_feat, clean_qrs, clean_norm, clean_logits, clean_mlp = clean_feats[base_name]
            sqi_distance = float(np.linalg.norm(cand_norm - clean_norm))
            ok, gate = _accepted(clean_feat, clean_qrs, cand_feat, cand_qrs, sqi_distance)
            cand_logits, cand_probs = _score_conformer(model, [cand], test_ds.channel_stats, device)
            cand_mlp = ctx.mlp.predict(cand_norm[None, :])
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "direction": direction,
                    "control": control,
                    "recipient_class": base_name,
                    "window_start_s": start / FS,
                    "window_stop_s": stop / FS,
                    "accepted": int(ok),
                    "sqi_distance": gate["sqi_distance"],
                    "delta_iSQI": gate["iSQI"],
                    "delta_bSQI": gate["bSQI"],
                    "delta_pSQI": gate["pSQI"],
                    "delta_basSQI": gate["basSQI"],
                    "qrs_same": int(gate["qrs_same"]),
                    "delta_conformer_medium_logit": float(cand_logits[0, 1] - clean_logits[0, 1]),
                    "delta_conformer_bad_logit": float(cand_logits[0, 2] - clean_logits[0, 2]),
                    "delta_mlp_gm_margin": float(_gm_margin(cand_mlp)[0] - _gm_margin(clean_mlp)[0]),
                    "clean_mlp_gm_margin": float(_gm_margin(clean_mlp)[0]),
                    "cand_mlp_gm_margin": float(_gm_margin(cand_mlp)[0]),
                    "cand_conformer_pred": LABELS[int(cand_probs[0].argmax())],
                }
            )
        transplanted = _crossfade_transplant(good, medium, *top)
        examples.append(
            {
                "pair_id": int(pair_id),
                "good": good,
                "medium": medium,
                "transplanted": transplanted,
                "start": top[0],
                "stop": top[1],
                "display_clip_fraction": _display_clip_fraction([good, medium, transplanted], top[0], top[1]),
            }
        )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["direction", "control", "recipient_class"], as_index=False)
        .agg(
            n=("pair_id", "count"),
            accepted_n=("accepted", "sum"),
            accepted_rate=("accepted", "mean"),
            median_sqi_distance=("sqi_distance", "median"),
            mean_delta_conformer_medium_logit=("delta_conformer_medium_logit", lambda s: float(raw.loc[s.index][raw.loc[s.index, "accepted"].eq(1)]["delta_conformer_medium_logit"].mean())),
            mean_delta_conformer_bad_logit=("delta_conformer_bad_logit", lambda s: float(raw.loc[s.index][raw.loc[s.index, "accepted"].eq(1)]["delta_conformer_bad_logit"].mean())),
            mean_delta_mlp_gm_margin=("delta_mlp_gm_margin", lambda s: float(raw.loc[s.index][raw.loc[s.index, "accepted"].eq(1)]["delta_mlp_gm_margin"].mean())),
        )
        .fillna(0.0)
    )
    raw.to_csv(paths.tables / "but_real_pair_segment_transplant_raw.csv", index=False)
    summary.to_csv(paths.tables / "but_real_pair_segment_transplant_summary.csv", index=False)
    ex = pd.DataFrame(examples)
    keep = raw.loc[
        raw["direction"].eq("medium_into_good_top")
        & raw["accepted"].eq(1)
        & raw["delta_conformer_medium_logit"].gt(0.25)
        & raw["sqi_distance"].lt(0.10),
        ["pair_id", "delta_conformer_medium_logit", "sqi_distance"],
    ]
    chosen = ex.merge(keep, on="pair_id", how="inner").sort_values(
        ["delta_conformer_medium_logit", "display_clip_fraction"], ascending=[False, True]
    )
    if len(chosen) < 3:
        chosen = ex.merge(
            raw.loc[raw["direction"].eq("medium_into_good_top"), ["pair_id", "delta_conformer_medium_logit", "sqi_distance"]],
            on="pair_id",
            how="left",
        ).sort_values(["display_clip_fraction", "delta_conformer_medium_logit"], ascending=[True, False])
    return raw, summary, chosen.head(3)


def _copy_to_user_figures(paths: Paths, names: list[str]) -> None:
    target = ROOT / "outputs" / "transformer" / "supplemental" / "chapter4_evidence" / "figures"
    (target / "source_data").mkdir(parents=True, exist_ok=True)
    for name in names:
        for ext in [".png", ".svg", ".pdf", ".tiff"]:
            src = paths.figures / f"{name}{ext}"
            if src.exists():
                shutil.copy2(src, target / src.name)
    for csv in paths.source_data.glob("fig_M[89]_*.csv"):
        shutil.copy2(csv, target / "source_data" / csv.name)


def _fig_m8(paths: Paths, raw: pd.DataFrame, summary: pd.DataFrame) -> Path:
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    raw.to_csv(paths.source_data / "fig_M8_local_degradation_raw.csv", index=False)
    summary.to_csv(paths.source_data / "fig_M8_local_degradation_summary.csv", index=False)
    fig = plt.figure(figsize=(7.4, 3.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.62)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.34)
    bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.42)
    wave_axes = [fig.add_subplot(top[0, i]) for i in range(3)]
    ax = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    colors = {"hf_burst": "#8da0cb", "emg_noise_floor": "#66a61e", "reset_spike": "#c44e52"}
    labels = {"hf_burst": "hf burst", "emg_noise_floor": "EMG noise", "reset_spike": "reset spike"}
    example = pd.read_csv(paths.source_data / "fig_M8_wave_examples.csv")
    trace_colors = {0.0: "#3a3a3a", 0.8: "#bdbdbd", 1.6: "#c44e52"}
    for axis, kind in zip(wave_axes, PERTURBATIONS):
        sub = example.loc[example["perturbation"].eq(kind)].copy()
        clean = sub.loc[np.isclose(sub["severity"], 0.0)].sort_values("time_s")["value"].to_numpy(float)
        center = float(np.nanmedian(clean))
        scale = max(1e-6, float(np.nanpercentile(np.abs(clean - center), 98)) / 3.20)
        for offset, sev in zip([2.7, 0.0, -2.7], [0.0, 0.8, 1.6]):
            g = sub.loc[np.isclose(sub["severity"], sev)].sort_values("time_s")
            y = (g["value"].to_numpy(float) - center) / scale
            y = np.clip(y, -2.55, 2.55)
            axis.plot(g["time_s"], y + offset, lw=1.2 if sev == 1.6 else 0.85, color=trace_colors[sev], label=f"{sev:g}")
        win = sub.loc[np.isclose(sub["severity"], 0.8)].iloc[0]
        axis.axvspan(float(win["window_start_s"]), float(win["window_stop_s"]), color="#e6ab02", alpha=0.18, lw=0)
        axis.set_title(labels[kind], fontsize=7, pad=2)
        axis.set_yticks([2.7, 0.0, -2.7], ["0", "0.8", "1.6"])
        axis.set_xlabel("Time (s)")
        axis.set_xlim(float(sub["time_s"].min()), float(sub["time_s"].max()))
    wave_axes[0].set_ylabel("Severity")
    wave_axes[0].text(-0.22, 1.13, "a", transform=wave_axes[0].transAxes, fontweight="bold", fontsize=9)

    plot_summary = summary.loc[summary["severity"].le(1.6)].copy()
    for kind in PERTURBATIONS:
        sub = plot_summary.loc[plot_summary["perturbation"].eq(kind)].sort_values("severity")
        ax[0].plot(
            sub["severity"],
            sub["median_sqi_distance"],
            marker="o",
            markersize=2.8,
            lw=1.2,
            color=colors[kind],
        )
    ax[0].set_ylim(0, 0.10)
    ax[0].set_xlim(0.15, 1.65)
    ax[0].set_xlabel("Perturbation severity")
    ax[0].text(0.98, 0.92, f"gate {SQI_L2_MAX:g}", transform=ax[0].transAxes, ha="right", va="top", fontsize=6, color="#777777")
    ax[0].set_ylabel("84-SQI distance")
    for metric, axis, ylabel in [
        ("mean_delta_mlp_gm_margin", ax[1], "LM-MLP delta GM margin"),
        ("mean_delta_conformer_medium_logit", ax[2], "Conformer delta medium logit"),
    ]:
        for kind in PERTURBATIONS:
            sub = plot_summary.loc[plot_summary["perturbation"].eq(kind)].sort_values("severity")
            axis.plot(sub["severity"], sub[metric], marker="o", markersize=2.8, lw=1.3, color=colors[kind], label=labels[kind])
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xlim(0.15, 1.65)
        axis.set_xlabel("Perturbation severity")
        axis.set_ylabel(ylabel)
    ax[1].set_ylim(-0.01, 0.1)
    ax[1].legend(fontsize=6)
    for label, axis in zip(["b", "c", "d"], ax):
        axis.text(-0.14, 1.08, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14)
    out = paths.figures / "fig_M8_but_sqi_locked_local_degradation"
    _save(fig, out)
    return out.with_suffix(".png")


def _norm_wave(x: np.ndarray) -> np.ndarray:
    z = _robust_z(x)
    return np.clip(z, -1.25, 1.25)


def _display_segment(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    seg = np.asarray(x, dtype=np.float64)[lo:hi]
    center = float(np.nanmedian(seg))
    p01, p99 = np.nanpercentile(seg, [1, 99])
    scale = max(1e-6, float(p99 - p01) / 1.6)
    z = (seg - center) / scale
    return np.clip(z, -1.35, 1.35)


def _fig_m9(paths: Paths, raw: pd.DataFrame, summary: pd.DataFrame, examples: pd.DataFrame) -> Path:
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    raw.to_csv(paths.source_data / "fig_M9_segment_transplant_raw.csv", index=False)
    summary.to_csv(paths.source_data / "fig_M9_segment_transplant_summary.csv", index=False)
    fig = plt.figure(figsize=(7.4, 5.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
    ax_wave = [fig.add_subplot(gs[0, i]) for i in range(3)]
    t = np.arange(1250) / FS
    wave_rows: list[dict[str, Any]] = []
    for axis, row in zip(ax_wave, examples.itertuples(index=False)):
        lo = max(0, int(row.start) - 2 * FS)
        hi = min(1250, int(row.stop) + 2 * FS)
        for name, color, yoff in [("good", "#4c78a8", 1.65), ("medium", "#c44e52", 0.0), ("transplanted", "#66a61e", -1.65)]:
            wave = _display_segment(getattr(row, name), lo, hi)
            axis.plot(t[lo:hi], wave + yoff, color=color, lw=0.7)
            wave_rows.extend({"pair_id": int(row.pair_id), "trace": name, "time_s": float(tt), "value": float(v)} for tt, v in zip(t[lo:hi], wave))
        axis.axvspan(float(row.start) / FS, float(row.stop) / FS, color="#e6ab02", alpha=0.18, lw=0)
        axis.set_xlim(t[lo], t[hi - 1])
        axis.set_ylim(-2.9, 2.9)
        axis.set_yticks([1.65, 0.0, -1.65], ["good", "medium", "transplant"])
        axis.set_xlabel("Time (s)")
    pd.DataFrame(wave_rows).to_csv(paths.source_data / "fig_M9_example_waves.csv", index=False)
    ax_wave[0].set_ylabel("Amplitude (a.u.)")
    ax_wave[0].text(-0.14, 1.08, "a", transform=ax_wave[0].transAxes, fontweight="bold", fontsize=9)

    ax_b = fig.add_subplot(gs[1, 0:2])
    accepted = raw.loc[raw["accepted"].eq(1)].copy()
    plot = accepted.groupby(["recipient_class", "control"], as_index=False)["delta_conformer_medium_logit"].mean()
    controls = ["top-local", "random", "same-class"]
    xs = np.arange(len(controls))
    width = 0.34
    for cls, offset, color in [("good", -width / 2, "#4c78a8"), ("medium", width / 2, "#c44e52")]:
        vals = [float(plot.loc[plot["recipient_class"].eq(cls) & plot["control"].eq(c), "delta_conformer_medium_logit"].mean()) for c in controls]
        bars = ax_b.bar(xs + offset, vals, width, label=cls, color=color)
        ax_b.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.set_xticks(xs, controls)
    ax_b.set_ylabel("Conformer delta medium logit")
    ax_b.legend(fontsize=6)
    ax_b.text(-0.08, 1.08, "b", transform=ax_b.transAxes, fontweight="bold", fontsize=9)

    ax_c = fig.add_subplot(gs[1, 2])
    m = accepted.groupby("control", as_index=False).agg(sqi_distance=("sqi_distance", "median"), mlp_delta=("delta_mlp_gm_margin", "mean"))
    x = np.arange(len(m))
    ax_c.bar(x - 0.16, m["sqi_distance"], 0.32, color="#bdbdbd", label="SQI dist")
    ax_c.bar(x + 0.16, m["mlp_delta"], 0.32, color="#66a61e", label="MLP delta")
    ax_c.axhline(0, color="black", lw=0.8)
    ax_c.set_xticks(x, m["control"], rotation=25, ha="right")
    ax_c.set_ylabel("Control metric")
    ax_c.legend(fontsize=6)
    ax_c.text(-0.18, 1.08, "c", transform=ax_c.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout()
    out = paths.figures / "fig_M9_but_real_pair_segment_transplant"
    _save(fig, out)
    return out.with_suffix(".png")


def _write_report(paths: Paths, deg_raw: pd.DataFrame, deg_summary: pd.DataFrame, tr_raw: pd.DataFrame, tr_summary: pd.DataFrame, fig8: Path, fig9: Path) -> None:
    acc_deg = deg_raw.loc[deg_raw["accepted"].eq(1)]
    acc_tr = tr_raw.loc[tr_raw["accepted"].eq(1)]
    hf = deg_summary.loc[deg_summary["perturbation"].eq("hf_burst")].sort_values("severity")
    hf_hi = hf.iloc[-1] if len(hf) else pd.Series(dtype=float)
    emg = deg_summary.loc[deg_summary["perturbation"].eq("emg_noise_floor")].sort_values("severity")
    emg_hi = emg.iloc[-1] if len(emg) else pd.Series(dtype=float)
    reset = deg_summary.loc[deg_summary["perturbation"].eq("reset_spike")].sort_values("severity")
    reset_hi = reset.iloc[-1] if len(reset) else pd.Series(dtype=float)
    top_good = acc_tr.loc[acc_tr["direction"].eq("medium_into_good_top"), "delta_conformer_medium_logit"].mean()
    rand_good = acc_tr.loc[acc_tr["direction"].eq("medium_into_good_random"), "delta_conformer_medium_logit"].mean()
    same_good = acc_tr.loc[acc_tr["direction"].eq("good_into_good_same_class"), "delta_conformer_medium_logit"].mean()
    reverse_top = acc_tr.loc[acc_tr["direction"].eq("good_into_medium_top"), "delta_conformer_medium_logit"].mean()
    verdict = (
        "strong for SQI-locked high-frequency/EMG local noise sensitivity; raw segment transplant is supportive but not definitive"
        if len(hf) and len(emg) and float(hf_hi.get("mean_delta_conformer_medium_logit", 0.0)) > 5 * abs(float(hf_hi.get("mean_delta_mlp_gm_margin", 0.0)))
        else "partial evidence; inspect controls before using as a strong claim"
    )
    lines = [
        "# BUT Local Evidence Sensitivity Counterfactuals",
        "",
        f"- Verdict: **{verdict}**.",
        f"- SQI-locked degradation accepted rows: `{int(deg_raw['accepted'].sum())}/{len(deg_raw)}`.",
        f"- Segment transplant accepted rows: `{int(tr_raw['accepted'].sum())}/{len(tr_raw)}`.",
        f"- High-frequency burst at severity {float(hf_hi.get('severity', 0.0)):.1f}: Conformer `delta_medium={float(hf_hi.get('mean_delta_conformer_medium_logit', 0.0)):.4f}`, LM-MLP `delta_GM={float(hf_hi.get('mean_delta_mlp_gm_margin', 0.0)):.4f}`, Conformer `delta_bad={float(hf_hi.get('mean_delta_conformer_bad_logit', 0.0)):.4f}`.",
        f"- EMG noise floor at severity {float(emg_hi.get('severity', 0.0)):.1f}: Conformer `delta_medium={float(emg_hi.get('mean_delta_conformer_medium_logit', 0.0)):.4f}`, LM-MLP `delta_GM={float(emg_hi.get('mean_delta_mlp_gm_margin', 0.0)):.4f}`, Conformer `delta_bad={float(emg_hi.get('mean_delta_conformer_bad_logit', 0.0)):.4f}`.",
        f"- Reset spike at severity {float(reset_hi.get('severity', 0.0)):.1f}: accepted `{int(reset_hi.get('accepted_n', 0))}/{int(reset_hi.get('n', 0))}`, Conformer `delta_medium={float(reset_hi.get('mean_delta_conformer_medium_logit', 0.0)):.4f}`, LM-MLP `delta_GM={float(reset_hi.get('mean_delta_mlp_gm_margin', 0.0)):.4f}`, Conformer `delta_bad={float(reset_hi.get('mean_delta_conformer_bad_logit', 0.0)):.4f}`.",
        f"- Raw transplant medium-to-good: top-local `{float(top_good):.4f}`, random `{float(rand_good):.4f}`, same-class `{float(same_good):.4f}`; reverse good-to-medium top-local `{float(reverse_top):.4f}`.",
        f"- Figures: `{rel(fig8)}`, `{rel(fig9)}`.",
        "",
        "## SQI-locked local degradation",
        "",
        table_to_md(deg_summary),
        "",
        "## Real-pair segment transplant",
        "",
        table_to_md(tr_summary),
        "",
    ]
    (paths.reports / "but_local_counterfactuals_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        paths.reports / "but_local_counterfactuals_summary.json",
        {
            "verdict": verdict,
            "degradation_accepted": int(deg_raw["accepted"].sum()),
            "degradation_total": int(len(deg_raw)),
            "transplant_accepted": int(tr_raw["accepted"].sum()),
            "transplant_total": int(len(tr_raw)),
            "fig_M8": rel(fig8),
            "fig_M9": rel(fig9),
        },
    )


def run(paths: Paths, *, execute: bool, force: bool, device: str = "cuda") -> dict[str, Any]:
    if not execute:
        dry("but-local-counterfactuals", paths)
        return {"step": "but-local-counterfactuals", "skipped": True}
    ensure_dirs(paths)
    records = _validate_test_scope(_ensure_boundary_records(paths, force=False, device=device))
    model, _, test_ds, torch_device = _load_model(device)
    _align_clean_predictions(model, test_ds, torch_device)
    signals = _load_protocol_signals()
    sample = np.stack([_raw_wave(signals, test_ds, i) for i in range(min(8, len(test_ds)))])
    rebuilt = EVT.DUAL.make_dualview_channels(sample, test_ds.channel_stats)
    err = float(np.max(np.abs(rebuilt - test_ds.x[: len(sample)])))
    if err > 1.0e-5:
        raise RuntimeError(f"counterfactual view builder does not match test dataset: max_abs_err={err:.3g}")
    ctx = _fit_sqi_context(paths, force=force, device=device)
    deg_raw, deg_summary = _degradation(paths, records, model, test_ds, signals, ctx, torch_device)
    tr_raw, tr_summary, examples = _transplant_rows(paths, records, model, test_ds, signals, ctx, torch_device)
    fig8 = _fig_m8(paths, deg_raw, deg_summary)
    fig9 = _fig_m9(paths, tr_raw, tr_summary, examples)
    _copy_to_user_figures(paths, ["fig_M8_but_sqi_locked_local_degradation", "fig_M9_but_real_pair_segment_transplant"])
    index_path = paths.reports / "supplemental_figure_index.json"
    figure_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    figure_index["fig_M8_but_sqi_locked_local_degradation"] = str(fig8.resolve())
    figure_index["fig_M9_but_real_pair_segment_transplant"] = str(fig9.resolve())
    write_json(index_path, figure_index)
    _write_report(paths, deg_raw, deg_summary, tr_raw, tr_summary, fig8, fig9)
    out = {
        "degradation_raw": rel(paths.tables / "but_sqi_locked_local_degradation_raw.csv"),
        "degradation_summary": rel(paths.tables / "but_sqi_locked_local_degradation_summary.csv"),
        "transplant_raw": rel(paths.tables / "but_real_pair_segment_transplant_raw.csv"),
        "transplant_summary": rel(paths.tables / "but_real_pair_segment_transplant_summary.csv"),
        "fig_M8": rel(fig8),
        "fig_M9": rel(fig9),
        "report": rel(paths.reports / "but_local_counterfactuals_summary.md"),
    }
    print(json.dumps(out, indent=2))
    return {"step": "but-local-counterfactuals", "skipped": False, "outputs": out}
