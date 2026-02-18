# src/data/make_manifest_raw.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import pandas as pd
import wfdb

from src.utils.paths import project_root


# -----------------------------
# Config
# -----------------------------
LABEL_FILES = {
    "records": "RECORDS",
    "acceptable": "RECORDS-acceptable",
    "unacceptable": "RECORDS-unacceptable",
}

logger = logging.getLogger(__name__)

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

@dataclass(frozen=True)
class ChallengePaths:
    set_a_dir: Path
    records_file: Path
    acceptable_file: Path
    unacceptable_file: Path


def _read_id_list(path: Path) -> list[str]:
    """
    Read a PhysioNet-style ID list file.
    Each line can be:
      - "1002603"
      - "set-a/1002603"
      - "1002603.hea"
    Returns a list in file order (duplicates removed while preserving order).
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing list file: {path}")

    seen = set()
    out: list[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        s = s.replace("\\", "/").split("/")[-1]
        rid = Path(s).stem  # drop extension if any
        if rid and rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def _scan_hea_ids(dir_: Path) -> list[str]:
    """Scan *.hea under dir_ and return sorted record ids (stem)."""
    return sorted(p.stem for p in dir_.glob("*.hea"))


def _resolve_challenge_paths(challenge_root: Path) -> ChallengePaths:
    set_a_dir = challenge_root / "set-a"
    if not set_a_dir.exists():
        raise FileNotFoundError(f"Missing directory: {set_a_dir}")

    def p(name: str) -> Path:
        return set_a_dir / name

    return ChallengePaths(
        set_a_dir=set_a_dir,
        records_file=p(LABEL_FILES["records"]),
        acceptable_file=p(LABEL_FILES["acceptable"]),
        unacceptable_file=p(LABEL_FILES["unacceptable"]),
    )

# --- add near top (config) ---
STRICT_LABELS = False  # True -> unknown 直接报错；False -> 允许 unknown 并输出单独清单


def _load_seta_labels(paths: ChallengePaths) -> tuple[dict[str, str], list[str]]:
    """
    Load record-level labels for set-a.
    Returns:
      labels: {record_id: "acceptable" | "unacceptable" | "unknown"}
      unknown_ids: list of record ids that are in RECORDS but not in label lists
    """
    recs = _read_id_list(paths.records_file)
    acc = set(_read_id_list(paths.acceptable_file))
    unacc = set(_read_id_list(paths.unacceptable_file))

    overlap = acc & unacc
    if overlap:
        sample = sorted(list(overlap))[:10]
        raise ValueError(f"Label overlap between acceptable/unacceptable: {sample} (and more)")

    labels: dict[str, str] = {}
    unknown_ids: list[str] = []

    for rid in recs:
        if rid in acc:
            labels[rid] = "acceptable"
        elif rid in unacc:
            labels[rid] = "unacceptable"
        else:
            labels[rid] = "unknown"
            unknown_ids.append(rid)

    if unknown_ids and STRICT_LABELS:
        raise ValueError(
            f"Found {len(unknown_ids)} records in RECORDS but not in label lists. "
            f"Examples: {unknown_ids[:10]}. Check label files."
        )

    return labels, unknown_ids


def manifest_challenge_seta(challenge_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build manifests for Challenge 2011 set-a only.
    Returns:
      df_all: all records in set-a/RECORDS (including unknown if any)
      df_unknown: subset where quality_record == "unknown"
    """
    paths = _resolve_challenge_paths(challenge_root)
    labels, unknown_ids = _load_seta_labels(paths)

    hea_ids = set(_scan_hea_ids(paths.set_a_dir))
    rec_ids = set(labels.keys())

    missing_on_disk = sorted(list(rec_ids - hea_ids))
    extra_on_disk = sorted(list(hea_ids - rec_ids))

    if missing_on_disk:
        raise ValueError(f"IDs in RECORDS but missing .hea on disk (examples): {missing_on_disk[:10]}")
    if extra_on_disk:
        # 不致命：可能目录里有别的文件；但为了干净我还是建议 raise
        raise ValueError(f".hea exists but not listed in RECORDS (examples): {extra_on_disk[:10]}")

    rows = []
    for rid in sorted(rec_ids):
        rec_path = str(paths.set_a_dir / rid)
        h = wfdb.rdheader(rec_path)

        q = labels[rid]
        y = 1 if q == "acceptable" else 0 if q == "unacceptable" else -1

        rows.append(
            dict(
                record_id=rid,
                split="set-a",
                quality_record=q,     # acceptable / unacceptable / unknown
                y=y,                  # 1 / 0 / -1
                fs=float(h.fs),
                sig_len=int(h.sig_len),
                n_sig=int(h.n_sig),
                duration_s=float(h.sig_len / h.fs),
                sig_name=",".join(h.sig_name) if getattr(h, "sig_name", None) else "",
                base_path=str(paths.set_a_dir),
            )
        )

    df_all = pd.DataFrame(rows)
    df_unknown = df_all[df_all["quality_record"] == "unknown"].copy()

    if unknown_ids:
        logger.warning("set-a labels do not cover all RECORDS: unknown=%d", len(unknown_ids))
        logger.warning("unknown examples: %s", unknown_ids[:20])

    return df_all, df_unknown

def manifest_nstdb(nstdb_root: Path) -> pd.DataFrame:
    """
    NSTDB manifest for em/ma with a split_index (halfway point) used to avoid leakage.
    """
    rows = []
    for rec in ["em", "ma"]:
        rec_path = str(nstdb_root / rec)
        h = wfdb.rdheader(rec_path)
        rows.append(
            dict(
                record_id=rec,
                fs=float(h.fs),
                sig_len=int(h.sig_len),
                n_sig=int(h.n_sig),
                duration_s=float(h.sig_len / h.fs),
                sig_name=",".join(h.sig_name) if getattr(h, "sig_name", None) else "",
                base_path=str(nstdb_root),
                split_index=int(h.sig_len // 2),
            )
        )
    return pd.DataFrame(rows)

def _outputs_exist(out_dir: Path) -> bool:
    out_c = out_dir / "manifest_challenge2011_seta.csv"
    out_u = out_dir / "manifest_challenge2011_seta_unknown.csv"
    out_n = out_dir / "manifest_nstdb_raw.csv"

    def ok(p: Path) -> bool:
        return p.exists() and p.is_file() and p.stat().st_size > 0

    return ok(out_c) and ok(out_u) and ok(out_n)

def run(params: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline-callable entrypoint.

    params (optional):
      - verbose: bool
      - force: bool
      - artifacts_dir: str (default "artifacts")
      - challenge_root: str (default project_root()/data/physionet/challenge-2011)
      - nstdb_root: str (default project_root()/data/physionet/nstdb)

    Returns: {"step": "...", "skipped": bool, "outputs": [..]}
    """
    verbose = bool(params.get("verbose", False))
    force = bool(params.get("force", False))
    _setup_logging(verbose)

    root = project_root()

    # default roots (no CLI pain)
    challenge_root = params.get("challenge_root")
    if not challenge_root:
        challenge_root = root / "data" / "physionet" / "challenge-2011"
    else:
        challenge_root = Path(str(challenge_root))

    nstdb_root = params.get("nstdb_root")
    if not nstdb_root:
        nstdb_root = root / "data" / "physionet" / "nstdb"
    else:
        nstdb_root = Path(str(nstdb_root))

    artifacts_dir = params.get("artifacts_dir")
    if not artifacts_dir:
        artifacts_dir = root / "artifacts"
    else:
        artifacts_dir = Path(str(artifacts_dir))

    out_dir = Path(artifacts_dir) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_c = out_dir / "manifest_challenge2011_seta.csv"
    out_u = out_dir / "manifest_challenge2011_seta_unknown.csv"
    out_n = out_dir / "manifest_nstdb_raw.csv"

    if (not force) and _outputs_exist(out_dir):
        logger.info("manifest_raw: outputs exist -> skip (set force=True to rerun)")
        return {"step": "manifest_raw", "skipped": True, "outputs": [str(out_c), str(out_u), str(out_n)]}

    logger.info("manifest_raw: challenge_root=%s", challenge_root)
    logger.info("manifest_raw: nstdb_root=%s", nstdb_root)
    logger.info("manifest_raw: out_dir=%s", out_dir)

    df_c, df_u = manifest_challenge_seta(Path(challenge_root))
    df_n = manifest_nstdb(Path(nstdb_root))

    df_c.to_csv(out_c, index=False)
    df_u.to_csv(out_u, index=False)
    df_n.to_csv(out_n, index=False)

    # concise logs
    logger.info("Wrote: %s rows=%d", out_c, len(df_c))
    logger.info("label_counts: %s", df_c["quality_record"].value_counts(dropna=False).to_dict())
    logger.info("Wrote: %s rows=%d", out_u, len(df_u))
    logger.info("Wrote: %s rows=%d", out_n, len(df_n))

    return {"step": "manifest_raw", "skipped": False, "outputs": [str(out_c), str(out_u), str(out_n)]}


def main() -> None:
    root = project_root()
    params = {
        "challenge_root": str(root / "data" / "physionet" / "challenge-2011"),
        "nstdb_root": str(root / "data" / "physionet" / "nstdb"),
        "artifacts_dir": str(root / "artifacts"),
        "verbose": False,
        "force": False,
    }
    run(params)


if __name__ == "__main__":
    main()
