from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from src.utils.paths import project_root
from src.sqi_pipeline.noise.dower import LEADS_12, dower_3_to_12
from src.sqi_pipeline.noise.make_balanced_noisy_cases import (
    CASE_SEC,
    FS_ECG,
    FS_NOISE_IN,
    N_SAMPLES,
    load_clean_500_from_wfdb,
    read_nstdb,
    scale_noise_to_snr,
    write_case_500_npz,
)
from src.sqi_pipeline.noise.pca import make_3_orthogonal_from_2_paper


logger = logging.getLogger(__name__)

NOISE_TYPES = ("em", "ma")
SPLIT_RATIO = (0.70, 0.15, 0.15)
SNR_DB = -6.0


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def _outputs_exist(out_split: Path, audit_csv: Path, cases_dir: Path, qc_png: Path) -> bool:
    def ok(p: Path) -> bool:
        return p.exists() and p.is_file() and p.stat().st_size > 0

    if not (ok(out_split) and ok(audit_csv) and ok(qc_png) and cases_dir.exists()):
        return False
    try:
        df = pd.read_csv(out_split, usecols=["record_id"])
    except Exception:
        return False
    expected = {str(x) for x in df["record_id"].astype(str)}
    if not expected:
        return False
    have = {p.stem for p in cases_dir.glob("*.npz")}
    return expected.issubset(have)


def _plot_label_counts(df: pd.DataFrame, out_png: Path) -> None:
    table = df.groupby(["split", "y"]).size().unstack(fill_value=0).sort_index()
    ax = table.plot(kind="bar")
    ax.set_title("Paper-aligned Set-a label counts")
    ax.set_xlabel("split")
    ax.set_ylabel("count")
    ax.legend(title="y")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _cache_clean_cases(record_ids: list[str], set_a_dir: Path, cases_dir: Path, *, force: bool) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    n_skip = 0
    for i, rid in enumerate(record_ids, start=1):
        out = cases_dir / f"{rid}.npz"
        if out.exists() and not force:
            n_skip += 1
            continue
        sig = load_clean_500_from_wfdb(rid, set_a_dir)
        write_case_500_npz(out, sig, {"record_id": rid, "kind": "clean", "fs": FS_ECG})
        n_new += 1
        if i <= 5 or i % 200 == 0:
            logger.info("[clean] cached %s", rid)
    logger.info("[clean] cached new=%d skipped=%d total=%d", n_new, n_skip, len(record_ids))


class UniqueNoiseSegmentSampler:
    def __init__(
        self,
        nstdb_signals: dict[str, np.ndarray],
        *,
        rng: np.random.Generator,
        stride_s: float,
    ) -> None:
        self.nstdb_signals = nstdb_signals
        self.rng = rng
        self.seg_len_in = int(round(CASE_SEC * FS_NOISE_IN))
        stride = max(1, int(round(float(stride_s) * FS_NOISE_IN)))
        self.starts: dict[str, list[int]] = {}
        for noise_type, sig in nstdb_signals.items():
            hi = int(sig.shape[0] - self.seg_len_in)
            if hi <= 0:
                raise ValueError(f"NSTDB {noise_type} too short for {CASE_SEC}s windows")
            starts = np.arange(0, hi + 1, stride, dtype=int)
            self.rng.shuffle(starts)
            self.starts[noise_type] = starts.tolist()

    def draw_500(self, noise_type: str) -> tuple[np.ndarray, int]:
        starts = self.starts[noise_type]
        if not starts:
            raise RuntimeError(f"No unique NSTDB segment starts left for noise_type={noise_type}")
        start = int(starts.pop())
        sig360 = self.nstdb_signals[noise_type]
        seg360 = sig360[start : start + self.seg_len_in, :]
        seg500 = resample_poly(seg360, up=25, down=18, axis=0).astype(np.float64)
        if seg500.shape[0] < N_SAMPLES:
            seg500 = np.pad(seg500, ((0, N_SAMPLES - seg500.shape[0]), (0, 0)), mode="edge")
        return seg500[:N_SAMPLES, :], start


def _make_noisy_case_paper(
    clean500_12: np.ndarray,
    noise2_500: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    noise3 = make_3_orthogonal_from_2_paper(noise2_500, rng=rng)
    noise12 = dower_3_to_12(noise3)
    scaled = scale_noise_to_snr(clean500_12, noise12, snr_db)
    noisy = clean500_12 + scaled
    return noisy, {
        "pca_mode": "paper_pca_deterministic_third_axis",
        "dower_mode": "inverse_dower_12x3",
        "snr_db": float(snr_db),
        "pv_mean_per_lead_before_scale": [float(np.mean(noise12[:, i] ** 2)) for i in range(12)],
        "pv_mean_per_lead_after_scale": [float(np.mean(scaled[:, i] ** 2)) for i in range(12)],
    }


def _stratified_group_split(group_df: pd.DataFrame, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    train_r, val_r, _ = SPLIT_RATIO
    out: dict[str, str] = {}
    for group_type in sorted(group_df["group_type"].unique()):
        ids = group_df.loc[group_df["group_type"] == group_type, "source_record_id"].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(train_r * n))
        n_val = int(round(val_r * n))
        for gid in ids[:n_train]:
            out[str(gid)] = "train"
        for gid in ids[n_train:n_train + n_val]:
            out[str(gid)] = "val"
        for gid in ids[n_train + n_val:]:
            out[str(gid)] = "test"
    return out


def _assign_group_types(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gid, g in df.groupby("source_record_id", sort=True):
        n_good = int((g["y"] == 1).sum())
        n_bad = int((g["y"] == -1).sum())
        n_aug = int((g["is_augmented"] == 1).sum())
        if n_aug > 0:
            group_type = "acceptable_with_noise"
        elif n_good > 0:
            group_type = "acceptable_clean_only"
        else:
            group_type = "unacceptable_original"
        rows.append({
            "source_record_id": str(gid),
            "n_good": n_good,
            "n_bad": n_bad,
            "n_augmented": n_aug,
            "group_type": group_type,
        })
    return pd.DataFrame(rows)


def _validate_split(df: pd.DataFrame) -> None:
    n_good = int((df["y"] == 1).sum())
    n_bad = int((df["y"] == -1).sum())
    if n_good != n_bad:
        raise AssertionError(f"paper balanced set is not globally balanced: good={n_good} bad={n_bad}")

    owner = df[["source_record_id", "split"]].drop_duplicates()
    leaked = owner.groupby("source_record_id")["split"].nunique()
    leaked = leaked[leaked > 1]
    if len(leaked) > 0:
        raise AssertionError(f"source_record_id leakage across splits: {leaked.head(10).index.tolist()}")


def run(params: dict[str, Any]) -> dict[str, Any]:
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()
    seed = int(params.get("seed", 0))
    snr_db = float(params.get("snr_db", SNR_DB))
    noise_stride_s = float(params.get("noise_start_stride_s", 1.0))

    manifest_csv = _resolve_path(root, params["manifest_csv"])
    out_split = _resolve_path(root, params["out_split_csv"])
    audit_csv = _resolve_path(root, params.get("audit_csv", out_split.with_suffix(".audit.csv")))
    qc_png = _resolve_path(root, params.get("qc_png", out_split.with_suffix(".label_counts.png")))
    set_a_dir = _resolve_path(root, params["set_a_dir"])
    nstdb_dir = _resolve_path(root, params["nstdb_dir"])
    cases_dir = _resolve_path(root, params["cases_500_dir"])

    if (not force) and _outputs_exist(out_split, audit_csv, cases_dir, qc_png):
        logger.info("paper_balanced_seta: outputs exist -> skip (set force=True to rerun)")
        return {"step": "paper_balanced_seta", "skipped": True, "outputs": [str(out_split), str(audit_csv), str(qc_png)]}

    logger.info("manifest_csv: %s", manifest_csv)
    logger.info("set_a_dir: %s", set_a_dir)
    logger.info("nstdb_dir: %s", nstdb_dir)
    logger.info("cases_500_dir: %s", cases_dir)
    logger.info("seed=%d snr_db=%.1f noise_stride_s=%.3f", seed, snr_db, noise_stride_s)

    df_manifest = pd.read_csv(manifest_csv)
    df_labeled = df_manifest[df_manifest["quality_record"].isin(["acceptable", "unacceptable"])].copy()
    df_labeled["record_id"] = df_labeled["record_id"].astype(str)
    df_labeled["y"] = df_labeled["quality_record"].map({"acceptable": 1, "unacceptable": -1}).astype(int)
    df_labeled = df_labeled.sort_values("record_id").reset_index(drop=True)

    n_good = int((df_labeled["y"] == 1).sum())
    n_bad = int((df_labeled["y"] == -1).sum())
    need = n_good - n_bad
    if need < 0:
        raise ValueError(f"Set-a has more unacceptable than acceptable records: good={n_good} bad={n_bad}")
    if need % len(NOISE_TYPES) != 0:
        logger.warning("need=%d is not divisible by %d; final source group may have fewer noise variants", need, len(NOISE_TYPES))
    n_sources = int(np.ceil(need / len(NOISE_TYPES))) if need else 0

    rng = np.random.default_rng(seed)
    acceptable_ids = df_labeled.loc[df_labeled["y"] == 1, "record_id"].astype(str).to_numpy()
    chosen_sources = rng.choice(acceptable_ids, size=n_sources, replace=False).astype(str).tolist() if n_sources else []
    logger.info("label counts before balance: acceptable=%d unacceptable=%d need_noisy_bad=%d", n_good, n_bad, need)
    logger.info("chosen acceptable sources for noise: %d", len(chosen_sources))

    _cache_clean_cases(df_labeled["record_id"].astype(str).tolist(), set_a_dir, cases_dir, force=force)

    nstdb_signals = {noise_type: read_nstdb(noise_type, nstdb_dir) for noise_type in NOISE_TYPES}
    sampler = UniqueNoiseSegmentSampler(nstdb_signals, rng=rng, stride_s=noise_stride_s)

    rows: list[dict[str, Any]] = []
    for _, r in df_labeled.iterrows():
        rid = str(r["record_id"])
        rows.append({
            "record_id": rid,
            "y": int(r["y"]),
            "split": "",
            "seed": seed,
            "quality_record": str(r["quality_record"]),
            "is_augmented": 0,
            "source_record_id": rid,
            "noise_type": "",
            "snr_db": np.nan,
        })

    audit_rows: list[dict[str, Any]] = []
    made = 0
    for source_id in chosen_sources:
        clean = load_clean_500_from_wfdb(source_id, set_a_dir)
        for noise_type in NOISE_TYPES:
            if made >= need:
                break
            noise2_500, start360 = sampler.draw_500(noise_type)
            noisy, meta_extra = _make_noisy_case_paper(clean, noise2_500, rng, snr_db)
            new_id = f"{source_id}__paper_{noise_type}__snr{int(snr_db)}__seed{seed}__{made:05d}"
            out_npz = cases_dir / f"{new_id}.npz"
            meta = {
                "new_record_id": new_id,
                "source_record_id": source_id,
                "kind": "paper_aligned_noisy",
                "noise_type": noise_type,
                "noise_start_360": int(start360),
                "noise_start_stride_s": float(noise_stride_s),
                "fs_ecg": FS_ECG,
                "fs_noise_in": FS_NOISE_IN,
                **meta_extra,
            }
            write_case_500_npz(out_npz, noisy, meta)
            rows.append({
                "record_id": new_id,
                "y": -1,
                "split": "",
                "seed": seed,
                "quality_record": "paper_noisy_unacceptable",
                "is_augmented": 1,
                "source_record_id": source_id,
                "noise_type": noise_type,
                "snr_db": float(snr_db),
            })
            audit_rows.append({
                "record_id": new_id,
                "source_record_id": source_id,
                "noise_type": noise_type,
                "snr_db": float(snr_db),
                "noise_record": noise_type,
                "noise_start_360": int(start360),
                "noise_start_stride_s": float(noise_stride_s),
                "pca_mode": "paper_pca_deterministic_third_axis",
                "dower_mode": "inverse_dower_12x3",
                "out_npz_500": str(out_npz),
            })
            made += 1
        if made >= need:
            break

    df_bal = pd.DataFrame(rows)
    group_df = _assign_group_types(df_bal)
    split_map = _stratified_group_split(group_df, seed)
    df_bal["split"] = df_bal["source_record_id"].map(split_map).astype(str)
    if (df_bal["split"] == "nan").any() or df_bal["split"].isna().any():
        raise AssertionError("Some rows did not receive a split")

    _validate_split(df_bal)

    out_split.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_csv(out_split, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
    _plot_label_counts(df_bal, qc_png)

    logger.info("[saved] paper balanced split -> %s rows=%d", out_split, len(df_bal))
    logger.info("[saved] paper noise audit -> %s rows=%d", audit_csv, len(audit_rows))
    logger.info("[saved] label-count plot -> %s", qc_png)
    logger.info("global balanced counts: %s", df_bal["y"].value_counts().to_dict())
    logger.info("split counts:\n%s", df_bal.groupby(["split", "y"]).size().unstack(fill_value=0).to_string())
    logger.info("group type counts: %s", group_df["group_type"].value_counts().to_dict())

    return {"step": "paper_balanced_seta", "skipped": False, "outputs": [str(out_split), str(audit_csv), str(qc_png)]}


def main() -> None:
    root = project_root()
    run({
        "manifest_csv": root / "outputs/sqi_paper_aligned/manifests/manifest_challenge2011_seta.csv",
        "out_split_csv": root / "outputs/sqi_paper_aligned/splits/split_seta_seed0_paper_balanced.csv",
        "set_a_dir": root / "data/physionet/challenge-2011/set-a",
        "nstdb_dir": root / "data/physionet/nstdb",
        "cases_500_dir": root / "outputs/sqi_paper_aligned/cases_500",
        "force": True,
    })


if __name__ == "__main__":
    main()
