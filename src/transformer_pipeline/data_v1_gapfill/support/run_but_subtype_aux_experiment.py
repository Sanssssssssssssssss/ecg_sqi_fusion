"""BUT subtype-aware waveform-only experiment.

This external-only runner tests whether a normal waveform Transformer learns
BUT bad better when the training target is not a single mixed "bad" bucket.

Inference input remains waveform-derived channels only.  Subtype labels are
training supervision and diagnostics only; the formal prediction is still the
3-class good/medium/bad output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from support_paths import ANALYSIS_DIR, OUT_ROOT, REPORT_DIR, REPORT_ROOT, ROOT, RUN_TAG  # noqa: E402
EVENT_SCRIPT = SCRIPT_DIR / "run_event_factorized_sqi_conformer.py"

BUT_KEEP_PROTOCOL = ANALYSIS_DIR / "clean_but_protocols" / "margin_ge_5s_keep_outlier"
ACTIVE_PROTOCOL = BUT_KEEP_PROTOCOL
OUT_DIR = ANALYSIS_DIR / "but_subtype_aux_experiment"
RUN_DIR = OUT_ROOT / "runs" / "but_subtype_aux_experiment"
REPORT_OUT_DIR = REPORT_DIR / "but_subtype_aux_experiment"
REPORT_PATH = REPORT_OUT_DIR / "but_subtype_aux_experiment_report.md"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVENT = load_module(EVENT_SCRIPT, "event_factorized_for_but_subtype_aux")
CLASS_TO_INT = EVENT.CLASS_TO_INT
INT_TO_CLASS = EVENT.INT_TO_CLASS

SUBTYPES = [
    "good_stable",
    "good_overlap_or_mild_artifact",
    "medium_stable",
    "medium_overlap_or_detail",
    "bad_dense_right_island",
    "bad_detector_template_disagree",
    "bad_baseline_wander_lowfreq",
    "bad_contact_reset_flatline",
    "bad_low_qrs_visibility",
    "bad_highfreq_detail_noise",
    "bad_other_boundary",
]
SUBTYPE_TO_INT = {name: i for i, name in enumerate(SUBTYPES)}
BAD_SUBTYPES = {name for name in SUBTYPES if name.startswith("bad_")}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in view.columns:
            val = row[col]
            vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def add_missing_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    defaults = {
        "original_region": "",
        "ambiguous_type": "",
        "baseline_step": 0.0,
        "band_0p3_1": 0.0,
        "lf_ratio": 0.0,
        "sqi_basSQI": 1.0,
        "flatline_ratio": 0.0,
        "contact_loss_win_ratio": 0.0,
        "local_rms_cv": 0.0,
        "low_amp_ratio": 0.0,
        "qrs_visibility": 1.0,
        "qrs_band_ratio": 1.0,
        "template_corr": 1.0,
        "detector_agreement": 1.0,
        "non_qrs_diff_p95": 0.0,
        "diff_abs_p95": 0.0,
        "band_30_45": 0.0,
        "detail_instability": 0.0,
    }
    for col, val in defaults.items():
        if col not in work.columns:
            work[col] = val
    return work


def subtype_labels(frame: pd.DataFrame) -> np.ndarray:
    """Build interpretable subtype targets from waveform-computable SQI features.

    Priority is used only to make a single auxiliary label.  Multi-label audit is
    saved separately to retain the BW/contact/low-QRS overlap distinction.
    """

    f = add_missing_columns(frame)
    cls = f["class_name"].astype(str).to_numpy()
    region = f["original_region"].fillna("").astype(str).str.lower()
    amb = f["ambiguous_type"].fillna("").astype(str).str.lower()

    bw = (
        (pd.to_numeric(f["baseline_step"], errors="coerce").fillna(0.0) >= 0.30)
        | (pd.to_numeric(f["band_0p3_1"], errors="coerce").fillna(0.0) >= 0.02)
        | (pd.to_numeric(f["lf_ratio"], errors="coerce").fillna(0.0) >= 0.02)
        | (pd.to_numeric(f["sqi_basSQI"], errors="coerce").fillna(1.0) <= 0.98)
    )
    contact = (
        (pd.to_numeric(f["flatline_ratio"], errors="coerce").fillna(0.0) >= 0.05)
        | (pd.to_numeric(f["contact_loss_win_ratio"], errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(f["local_rms_cv"], errors="coerce").fillna(0.0) >= 0.20)
        | (pd.to_numeric(f["low_amp_ratio"], errors="coerce").fillna(0.0) >= 0.18)
    )
    low_qrs = (
        (pd.to_numeric(f["qrs_visibility"], errors="coerce").fillna(1.0) <= 0.12)
        | (pd.to_numeric(f["qrs_band_ratio"], errors="coerce").fillna(1.0) <= 0.40)
    )
    detail = (
        (pd.to_numeric(f["non_qrs_diff_p95"], errors="coerce").fillna(0.0) >= 0.42)
        | (pd.to_numeric(f["diff_abs_p95"], errors="coerce").fillna(0.0) >= 0.45)
        | (pd.to_numeric(f["band_30_45"], errors="coerce").fillna(0.0) >= 0.15)
        | (pd.to_numeric(f["detail_instability"], errors="coerce").fillna(0.0) >= 0.50)
    )
    disagree = (
        (pd.to_numeric(f["detector_agreement"], errors="coerce").fillna(1.0) <= 0.35)
        | (pd.to_numeric(f["template_corr"], errors="coerce").fillna(1.0) <= 0.11)
    )
    gm_overlap = region.str.contains("good_medium|outlier_low|boundary", regex=True) | amb.str.contains("overlap|boundary|outlier", regex=True)
    mild_artifact = bw | contact | low_qrs | detail | disagree

    out = np.empty(len(f), dtype=object)
    out[cls == "good"] = "good_stable"
    out[(cls == "good") & (gm_overlap.to_numpy() | mild_artifact.to_numpy())] = "good_overlap_or_mild_artifact"
    out[cls == "medium"] = "medium_stable"
    out[(cls == "medium") & (gm_overlap.to_numpy() | detail.to_numpy() | low_qrs.to_numpy() | bw.to_numpy())] = "medium_overlap_or_detail"

    bad = cls == "bad"
    out[bad] = "bad_other_boundary"
    # Priority: contact/BW/low-QRS/detail are the mechanisms we want to rescue
    # from the broad outlier bucket before assigning dense-noise or detector-only.
    for name, mask in [
        ("bad_contact_reset_flatline", contact.to_numpy()),
        ("bad_baseline_wander_lowfreq", bw.to_numpy()),
        ("bad_low_qrs_visibility", low_qrs.to_numpy()),
        ("bad_highfreq_detail_noise", detail.to_numpy()),
        ("bad_detector_template_disagree", disagree.to_numpy()),
    ]:
        m = bad & mask & (out == "bad_other_boundary")
        out[m] = name
    dense = bad & region.str.contains("right_bad_island", regex=False).to_numpy() & (out == "bad_other_boundary")
    out[dense] = "bad_dense_right_island"
    # Synthetic subtype-balanced protocols carry the generator's intended bad
    # mechanism.  Prefer that explicit target for bad rows so training follows
    # the block generator instead of re-bucketing generated rows with old BUT
    # thresholds.  Real BUT protocols leave this column empty and still use the
    # waveform-computable audit above.
    for col in ("target_bad_subtype", "computed_subtype"):
        if col in f.columns:
            target = f[col].fillna("").astype(str).to_numpy()
            valid = bad & np.isin(target, list(BAD_SUBTYPES))
            out[valid] = target[valid]
    return np.asarray([SUBTYPE_TO_INT[str(x)] for x in out], dtype=np.int64)


def subtype_multilabel_audit(frame: pd.DataFrame) -> pd.DataFrame:
    f = add_missing_columns(frame)
    bad = f["class_name"].astype(str).eq("bad")
    rows = []
    checks = {
        "baseline_wander_lowfreq": (
            (pd.to_numeric(f["baseline_step"], errors="coerce").fillna(0.0) >= 0.30)
            | (pd.to_numeric(f["band_0p3_1"], errors="coerce").fillna(0.0) >= 0.02)
            | (pd.to_numeric(f["lf_ratio"], errors="coerce").fillna(0.0) >= 0.02)
            | (pd.to_numeric(f["sqi_basSQI"], errors="coerce").fillna(1.0) <= 0.98)
        ),
        "contact_reset_flatline": (
            (pd.to_numeric(f["flatline_ratio"], errors="coerce").fillna(0.0) >= 0.05)
            | (pd.to_numeric(f["contact_loss_win_ratio"], errors="coerce").fillna(0.0) > 0.0)
            | (pd.to_numeric(f["local_rms_cv"], errors="coerce").fillna(0.0) >= 0.20)
            | (pd.to_numeric(f["low_amp_ratio"], errors="coerce").fillna(0.0) >= 0.18)
        ),
        "low_qrs_visibility": (
            (pd.to_numeric(f["qrs_visibility"], errors="coerce").fillna(1.0) <= 0.12)
            | (pd.to_numeric(f["qrs_band_ratio"], errors="coerce").fillna(1.0) <= 0.40)
        ),
        "highfreq_detail_noise": (
            (pd.to_numeric(f["non_qrs_diff_p95"], errors="coerce").fillna(0.0) >= 0.42)
            | (pd.to_numeric(f["diff_abs_p95"], errors="coerce").fillna(0.0) >= 0.45)
            | (pd.to_numeric(f["band_30_45"], errors="coerce").fillna(0.0) >= 0.15)
            | (pd.to_numeric(f["detail_instability"], errors="coerce").fillna(0.0) >= 0.50)
        ),
        "detector_template_disagree": (
            (pd.to_numeric(f["detector_agreement"], errors="coerce").fillna(1.0) <= 0.35)
            | (pd.to_numeric(f["template_corr"], errors="coerce").fillna(1.0) <= 0.11)
        ),
    }
    for name, mask in checks.items():
        sub = f[bad & mask]
        rows.append(
            {
                "mechanism": name,
                "bad_rows": int(len(sub)),
                "records": ";".join(f"{k}:{v}" for k, v in sub["record_id"].value_counts().head(8).items()) if "record_id" in sub else "",
                "regions": ";".join(f"{k}:{v}" for k, v in sub["original_region"].value_counts().items()) if "original_region" in sub else "",
            }
        )
    return pd.DataFrame(rows)


class SubtypeDataset(EVENT.CleanProtocolDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.subtype = subtype_labels(self.frame)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        item = super().__getitem__(i)
        item["subtype"] = torch.tensor(int(self.subtype[i]), dtype=torch.long)
        return item


class SubtypeModel(nn.Module):
    def __init__(self, base: nn.Module, width: int, subtype_count: int) -> None:
        super().__init__()
        self.base = base
        self.subtype_head = nn.Sequential(
            nn.LayerNorm(width * len(EVENT.EventFactorizedSQIConformer.query_names)),
            nn.Linear(width * len(EVENT.EventFactorizedSQIConformer.query_names), 128),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(128, subtype_count),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base(x)
        q = out["query_tokens"].reshape(x.shape[0], -1)
        out["subtype_logits"] = self.subtype_head(q)
        return out


def make_dataset(split: str, ref_ds: SubtypeDataset | None = None, max_rows: int = 0, seed: int = 0) -> SubtypeDataset:
    return SubtypeDataset(
        ACTIVE_PROTOCOL,
        split,
        channel_stats=None if ref_ds is None else ref_ds.channel_stats,
        feature_norm=None if ref_ds is None else ref_ds.feature_norm,
        factor_transform=None if ref_ds is None else ref_ds.factor_transform,
        artifact_thresholds=None if ref_ds is None else ref_ds.artifact_thresholds,
        max_rows=max_rows,
        seed=seed,
    )


def make_model(cfg: dict[str, Any], in_ch: int, device: torch.device) -> nn.Module:
    base = EVENT.EventFactorizedSQIConformer(
        in_ch=int(in_ch),
        factor_dim=len(EVENT.FACTOR_COLUMNS),
        width=int(cfg["width"]),
        layers=int(cfg["layers"]),
        heads=int(cfg["heads"]),
        use_queries=bool(cfg.get("use_queries", True)),
        use_highres_fusion=bool(cfg.get("use_highres_fusion", True)),
        use_local_supervision=bool(cfg.get("use_local_supervision", True)),
        use_artifact_aux=bool(cfg.get("use_artifact_aux", True)),
        use_hierarchical_head=True,
        branch_upper_block=bool(cfg.get("branch_upper_block", False)),
    )
    model = SubtypeModel(base, width=int(cfg["width"]), subtype_count=len(SUBTYPES))
    return model.to(device)


def subtype_sampler(ds: SubtypeDataset, seed: int) -> WeightedRandomSampler:
    y = pd.Series(ds.y).astype(int)
    st = pd.Series(ds.subtype).astype(int)
    rec = pd.Series(ds.record_id).astype(str)
    counts = pd.DataFrame({"record": rec, "y": y, "subtype": st}).groupby(["record", "y", "subtype"]).size()
    weights = []
    for r, cls, sub in zip(rec, y, st):
        weights.append(1.0 / float(max(1, counts.loc[(r, cls, sub)])))
    arr = np.asarray(weights, dtype=np.float64)
    arr = arr / arr.mean()
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(arr, dtype=torch.double), num_samples=len(arr), replacement=True, generator=gen)


def loader(ds: SubtypeDataset, batch_size: int, shuffle: bool = False, sampler: Any | None = None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def subtype_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    target = batch["subtype"].to(device=device)
    # Mild class weighting prevents the tiny BW/contact/low-QRS groups from
    # being ignored while avoiding an extreme rare-class explosion.
    counts = torch.bincount(target, minlength=len(SUBTYPES)).float().to(device)
    weights = torch.sqrt(torch.clamp(counts.sum() / torch.clamp(counts, min=1.0), min=1.0, max=25.0))
    weights = weights / weights.mean().clamp(min=1e-6)
    return F.cross_entropy(out["subtype_logits"], target, weight=weights)


def full_loss(model: nn.Module, batch: dict[str, torch.Tensor], cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    total, parts, out = EVENT.model_loss(model, batch, cfg, device)
    st = subtype_loss(out, batch, device)
    total = total + float(cfg.get("subtype_weight", 0.0)) * st
    parts = dict(parts)
    parts["loss_subtype"] = float(st.detach().cpu())
    return total, parts


@torch.no_grad()
def eval_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    ys, probs, subtype_true, subtype_pred, records = [], [], [], [], []
    for batch in data_loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        out = model(x)
        ys.append(batch["y"].numpy())
        probs.append(out["probs"].detach().cpu().numpy())
        subtype_true.append(batch["subtype"].numpy())
        subtype_pred.append(torch.argmax(out["subtype_logits"], dim=1).detach().cpu().numpy())
        records.append(np.asarray(batch["record_id"], dtype=object))
    y = np.concatenate(ys)
    p = np.concatenate(probs)
    st = np.concatenate(subtype_true)
    sp = np.concatenate(subtype_pred)
    rec = np.concatenate(records).astype(str)
    rep = EVENT.basic_metric(y, p, supported_only=False)
    pred = np.argmax(p, axis=1)
    rep["subtype_acc"] = float(np.mean(st == sp))
    bad_mask = y == CLASS_TO_INT["bad"]
    rep["bad_subtype_acc"] = float(np.mean(st[bad_mask] == sp[bad_mask])) if np.any(bad_mask) else np.nan
    rep["good_to_medium"] = int(np.sum((y == 0) & (pred == 1)))
    rep["medium_to_good"] = int(np.sum((y == 1) & (pred == 0)))
    rep["bad_to_nonbad"] = int(np.sum((y == 2) & (pred != 2)))
    rep["nonbad_to_bad"] = int(np.sum((y != 2) & (pred == 2)))
    rows = []
    for sid, name in enumerate(SUBTYPES):
        mask = st == sid
        if not np.any(mask):
            continue
        rows.append(
            {
                "subtype": name,
                "n": int(np.sum(mask)),
                "subtype_recall": float(np.mean(sp[mask] == sid)),
                "main_acc": float(np.mean(pred[mask] == y[mask])),
                "main_bad_recall_if_bad": float(np.mean(pred[mask] == 2)) if name in BAD_SUBTYPES else np.nan,
                "records": ";".join(f"{k}:{v}" for k, v in pd.Series(rec[mask]).value_counts().head(6).items()),
            }
        )
    return rep, pd.DataFrame(rows)


def candidate_configs(seed: int) -> dict[str, dict[str, Any]]:
    grid = EVENT.candidate_grid("phase1", int(seed))
    e1 = dict(grid["E1_query_only"])
    e4 = dict(grid["E4_query_highres_local_art"])
    return {
        "E1_baseline_keep": {**e1, "subtype_weight": 0.0, "subtype_sampler": False},
        "E1_subtype_aux_keep": {**e1, "subtype_weight": 0.35, "subtype_sampler": True},
        "E4_local_art_subtype_keep": {**e4, "subtype_weight": 0.35, "subtype_sampler": True},
    }


def train_one(candidate: str, cfg: dict[str, Any], args: argparse.Namespace, seed_index: int) -> dict[str, Any]:
    seed = int(args.seed) + 1000 * int(seed_index)
    cfg = dict(cfg)
    cfg["seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = make_dataset("train", max_rows=int(args.max_train_rows), seed=seed)
    val_ds = make_dataset("val", ref_ds=train_ds, max_rows=int(args.max_val_rows), seed=seed + 1)
    test_ds = make_dataset("test", ref_ds=train_ds, max_rows=int(args.max_test_rows), seed=seed + 2)
    sampler = subtype_sampler(train_ds, seed) if bool(cfg.get("subtype_sampler", False)) else EVENT.record_balanced_sampler(train_ds, seed)
    train_loader = loader(train_ds, int(args.batch_size), sampler=sampler)
    val_loader = loader(val_ds, int(args.batch_size) * 2)
    test_loader = loader(test_ds, int(args.batch_size) * 2)

    model = make_model(cfg, int(train_ds.x.shape[1]), device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    best_state = None
    best_score = -1e9
    logs = []
    run_path = RUN_DIR / f"{candidate}_seed{seed_index}"
    run_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        parts_rows = []
        for batch in train_loader:
            total, parts = full_loss(model, batch, cfg, device)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(total.detach().cpu()))
            parts_rows.append(parts)
        val_rep, _ = eval_model(model, val_loader, device)
        score = float(val_rep["macro_f1"]) + 0.25 * min(float(val_rep["good_recall"]), float(val_rep["medium_recall"]), float(val_rep["bad_recall"]))
        part_mean = pd.DataFrame(parts_rows).mean(numeric_only=True).to_dict() if parts_rows else {}
        row = {
            "candidate": candidate,
            "seed_index": seed_index,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"train_{k}": v for k, v in part_mean.items()},
            **{f"val_{k}": v for k, v in val_rep.items() if k != "confusion_3x3"},
        }
        logs.append(row)
        print(
            f"{candidate} seed={seed_index} epoch={epoch}/{args.epochs} "
            f"loss={np.mean(losses):.4f} val_acc={val_rep['acc']:.4f} "
            f"rec={val_rep['good_recall']:.3f}/{val_rep['medium_recall']:.3f}/{val_rep['bad_recall']:.3f} "
            f"badSubtype={val_rep['bad_subtype_acc']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    metrics = []
    subtype_rows = []
    for bucket, ds, dl in [("val", val_ds, val_loader), ("test", test_ds, test_loader)]:
        rep, sub = eval_model(model, dl, device)
        rep_row = {"candidate": candidate, "seed_index": seed_index, "bucket": bucket, "n_rows": int(len(ds)), **rep}
        if "confusion_3x3" in rep_row:
            rep_row["confusion_3x3"] = json.dumps(rep_row["confusion_3x3"])
        metrics.append(rep_row)
        sub["candidate"] = candidate
        sub["seed_index"] = seed_index
        sub["bucket"] = bucket
        subtype_rows.extend(sub.to_dict("records"))
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_config": cfg,
            "subtypes": SUBTYPES,
            "protocol": str(ACTIVE_PROTOCOL),
            "input_contract": "waveform-derived channels only; subtype labels are training/diagnostic only",
            "no_warm_start": True,
        },
        run_path / "ckpt_best.pt",
    )
    pd.DataFrame(logs).to_csv(run_path / "train_log.csv", index=False)
    return {"metrics": metrics, "subtype_rows": subtype_rows, "run_path": str(run_path)}


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "acc",
        "macro_f1",
        "good_recall",
        "medium_recall",
        "bad_recall",
        "subtype_acc",
        "bad_subtype_acc",
        "good_to_medium",
        "medium_to_good",
        "bad_to_nonbad",
        "nonbad_to_bad",
    ]
    rows = []
    for (candidate, bucket), sub in metrics.groupby(["candidate", "bucket"], dropna=False):
        row = {"candidate": candidate, "bucket": bucket, "runs": int(len(sub))}
        for col in numeric:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[col] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, metrics_path: Path, subtype_path: Path, audit_path: Path) -> None:
    counts = pd.read_csv(ACTIVE_PROTOCOL / "original_region_atlas.csv")
    split_counts = counts.groupby(["split", "class_name"], dropna=False).size().reset_index(name="n")
    subtype_counts = counts.assign(subtype=[SUBTYPES[i] for i in subtype_labels(counts)]).groupby(["split", "class_name", "subtype"], dropna=False).size().reset_index(name="n")
    lines = [
        "# BUT Subtype-Auxiliary Waveform Experiment",
        "",
        f"- Generated: {now()}",
        f"- Protocol: `{ACTIVE_PROTOCOL}`",
        "- Inference input: waveform-derived channels only.",
        "- Formal output remains 3-class good/medium/bad.",
        "- Subtype labels are auxiliary supervision and diagnostics only.",
        "",
        "## Split Counts",
        "",
        md_table(split_counts),
        "",
        "## Subtype Counts",
        "",
        md_table(subtype_counts, max_rows=120),
        "",
        "## Results",
        "",
        md_table(summary.sort_values(["bucket", "acc"], ascending=[True, False])),
        "",
        "## Output Files",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Subtype metrics: `{subtype_path}`",
        f"- Bad mechanism audit: `{audit_path}`",
        f"- Checkpoints/logs: `{RUN_DIR}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    global ACTIVE_PROTOCOL, OUT_DIR, RUN_DIR, REPORT_OUT_DIR, REPORT_PATH
    ACTIVE_PROTOCOL = Path(args.protocol)
    if args.output_name:
        out_name = str(args.output_name)
    elif ACTIVE_PROTOCOL != BUT_KEEP_PROTOCOL:
        out_name = f"but_subtype_aux_experiment_{ACTIVE_PROTOCOL.name}"
    else:
        out_name = "but_subtype_aux_experiment"
    OUT_DIR = ANALYSIS_DIR / out_name
    RUN_DIR = OUT_ROOT / "runs" / out_name
    REPORT_OUT_DIR = REPORT_DIR / out_name
    REPORT_PATH = REPORT_OUT_DIR / f"{out_name}_report.md"
    ensure_dirs()
    audit = subtype_multilabel_audit(pd.read_csv(ACTIVE_PROTOCOL / "original_region_atlas.csv"))
    audit_path = OUT_DIR / "but_bad_subtype_multilabel_audit.csv"
    audit.to_csv(audit_path, index=False)
    metric_rows = []
    subtype_rows = []
    candidates = [c.strip() for c in str(args.candidates).split(",") if c.strip()]
    for seed_index in range(int(args.seeds)):
        cfgs = candidate_configs(int(args.seed) + 1000 * seed_index)
        for candidate in candidates:
            res = train_one(candidate, cfgs[candidate], args, seed_index)
            metric_rows.extend(res["metrics"])
            subtype_rows.extend(res["subtype_rows"])
            metrics = pd.DataFrame(metric_rows)
            subtype_df = pd.DataFrame(subtype_rows)
            metrics_path = OUT_DIR / "but_subtype_aux_metrics.csv"
            subtype_path = OUT_DIR / "but_subtype_aux_subtype_metrics.csv"
            metrics.to_csv(metrics_path, index=False)
            subtype_df.to_csv(subtype_path, index=False)
            summary = summarize(metrics)
            summary.to_csv(OUT_DIR / "but_subtype_aux_summary.csv", index=False)
            write_report(summary, metrics_path, subtype_path, audit_path)
    print(REPORT_PATH, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--protocol", type=str, default=str(BUT_KEEP_PROTOCOL))
    parser.add_argument("--output-name", type=str, default="")
    parser.add_argument(
        "--candidates",
        type=str,
        default="E1_baseline_keep,E1_subtype_aux_keep,E4_local_art_subtype_keep",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

