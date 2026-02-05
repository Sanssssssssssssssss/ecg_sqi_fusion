# src/data/download_nstdb.py
from pathlib import Path
import wfdb
from src.utils.paths import project_root

def main():
    root = project_root()
    base = root / "data" / "physionet"
    nstdb_dir = base / "nstdb"
    nstdb_dir.mkdir(parents=True, exist_ok=True)

    # download file to data/physionet/nstdb/
    # em/ma only
    wfdb.dl_database(
        "nstdb",
        dl_dir=str(nstdb_dir),
        records=["em", "ma"],
        keep_subdirs=False,
    )

    print("NSTDB downloaded to:", nstdb_dir.resolve())
    print("Has em.hea?", (nstdb_dir / "em.hea").exists())
    print("Has ma.hea?", (nstdb_dir / "ma.hea").exists())

if __name__ == "__main__":
    main()
