# src/tool/read_mitbih_atr.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import wfdb
import json
from typing import Any

from src.utils.paths import project_root

# =========================
# EDIT PARAMS HERE
# =========================
RECORD_ID = "100"         # e.g. "100"
ANN_EXT = "atr"           # usually "atr"
PRINT_HEAD = 30

WRITE_CSV = True
OUT_CSV_REL = Path("artifacts/mitbih/100_atr.csv")

# dataset location relative to repo root
MITBIH_DIR_REL = Path("data/physionet/mit-bih")  # contains 100.dat/100.hea/100.atr
# =========================


def read_atr(record_base: Path, ann_ext: str = "atr") -> pd.DataFrame:
    """Read WFDB annotation file (e.g., 100.atr) and return DataFrame."""
    hdr = wfdb.rdheader(str(record_base))
    fs = float(hdr.fs)

    ann = wfdb.rdann(str(record_base), extension=ann_ext)

    df = pd.DataFrame(
        {
            "sample": ann.sample.astype(int),
            "time_s": ann.sample.astype(float) / fs,
            "symbol": list(ann.symbol),
            "aux_note": list(getattr(ann, "aux_note", [""] * len(ann.sample))),
        }
    )
    return df

def _safe_repr(x: Any, max_len: int = 300) -> str:
    """Short safe repr for debug printing."""
    try:
        s = repr(x)
    except Exception:
        s = f"<unreprable {type(x)}>"
    if len(s) > max_len:
        s = s[:max_len] + " ... <truncated>"
    return s


def dump_ann(ann: wfdb.Annotation, print_arrays_head: int = 20) -> None:
    """
    Print ALL public attributes on the WFDB Annotation object.
    For array-like attributes, print length + first few items.
    """
    print("\n====================")
    print("[read_mitbih_atr] RAW ann object dump")
    print("====================")

    keys = sorted([k for k in dir(ann) if not k.startswith("_")])
    for k in keys:
        try:
            v = getattr(ann, k)
        except Exception as e:
            print(f"- {k}: <error: {e}>")
            continue

        # skip callables (methods)
        if callable(v):
            continue

        # pretty print common array-likes
        try:
            n = len(v)  # type: ignore
            # strings also have len; treat as scalar if it's a str
            if isinstance(v, str):
                print(f"- {k}: {_safe_repr(v, max_len=600)}")
                continue

            head = None
            try:
                head = list(v[:print_arrays_head])  # type: ignore
            except Exception:
                try:
                    head = list(v)[:print_arrays_head]  # type: ignore
                except Exception:
                    head = None

            if head is not None:
                print(f"- {k}: len={n} head={_safe_repr(head, max_len=600)}")
            else:
                print(f"- {k}: len={n} value={_safe_repr(v)}")
        except Exception:
            # not sized
            print(f"- {k}: {_safe_repr(v, max_len=600)}")


def main() -> None:
    root = project_root()

    mitbih_dir = root / MITBIH_DIR_REL
    record_base = mitbih_dir / RECORD_ID
    out_csv = root / OUT_CSV_REL

    print(f"[read_mitbih_atr] project_root: {root}")
    print(f"[read_mitbih_atr] mitbih_dir:   {mitbih_dir}")
    print(f"[read_mitbih_atr] record_base:  {record_base}")
    print(f"[read_mitbih_atr] ann_ext:      {ANN_EXT}")

    # quick existence checks (clearer error than wfdb stacktrace)
    hea = record_base.with_suffix(".hea")
    dat = record_base.with_suffix(".dat")
    ann_path = record_base.with_suffix(f".{ANN_EXT}")
    for p in [hea, dat, ann_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    hdr = wfdb.rdheader(str(record_base))
    duration_s = float(hdr.sig_len) / float(hdr.fs)
    print(
        f"[read_mitbih_atr] fs={hdr.fs}Hz | sig_len={hdr.sig_len} | "
        f"duration={duration_s:.2f}s ({duration_s/60:.2f} min)"
    )

    # read dataframe view
    df = read_atr(record_base, ann_ext=ANN_EXT)

    print(f"[read_mitbih_atr] n_annotations={len(df)}")
    print("[read_mitbih_atr] symbol counts (top 20):")
    print(df["symbol"].value_counts().head(20).to_string())

    print(f"\n[read_mitbih_atr] first {PRINT_HEAD} rows:")
    print(df.head(PRINT_HEAD).to_string(index=False))

    # write csv
    if WRITE_CSV:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n[read_mitbih_atr] wrote csv -> {out_csv}")

    # ===== FULL PRINTS (ALL ROWS + ALL ANNOTATION FIELDS) =====
    print("\n====================")
    print("[read_mitbih_atr] FULL PRINTS")
    print("====================")

    # pandas display: print ALL rows/cols and do not truncate long strings
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", None)

    print("\n[read_mitbih_atr] FULL DataFrame (all rows):")
    print(df.to_string(index=False))

    # re-read ann to dump full object attributes (cheap and reliable)
    ann = wfdb.rdann(str(record_base), extension=ANN_EXT)

    # dump every public attribute in ann (helps discover noise/rhythm markers)
    dump_ann(ann, print_arrays_head=50)

    # optional: write a JSON snapshot of ann attributes (very handy for diff/debug)
    # you can delete this block if you don't want files.
    ann_dump_path = root / Path("artifacts/mitbih") / f"{RECORD_ID}_{ANN_EXT}_ann_dump.json"
    ann_dump_path.parent.mkdir(parents=True, exist_ok=True)

    ann_dict = {}
    for k in sorted([k for k in dir(ann) if not k.startswith("_")]):
        try:
            v = getattr(ann, k)
        except Exception:
            continue
        if callable(v):
            continue

        # convert to JSON-friendly
        try:
            if isinstance(v, (list, tuple)):
                ann_dict[k] = list(v)
            elif hasattr(v, "tolist"):
                ann_dict[k] = v.tolist()
            else:
                ann_dict[k] = v
        except Exception:
            ann_dict[k] = _safe_repr(v, max_len=2000)

    with open(ann_dump_path, "w", encoding="utf-8") as f:
        json.dump(ann_dict, f, ensure_ascii=False, indent=2)

    print(f"\n[read_mitbih_atr] wrote ann dump -> {ann_dump_path}")


if __name__ == "__main__":
    main()
