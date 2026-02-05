# src/data/download_dbs.py
from pathlib import Path
import wfdb
from src.utils.paths import project_root


def main() -> None:
    root = project_root()
    out = root / "data" / "physionet"
    out.mkdir(parents=True, exist_ok=True)

    #
    wfdb.dl_database("challenge-2011", dl_dir=str(out), keep_subdirs=True)

    # NSTDB
    wfdb.dl_database("nstdb", dl_dir=str(out), keep_subdirs=True)

    print("Downloaded to:", out.resolve())


if __name__ == "__main__":
    main()
