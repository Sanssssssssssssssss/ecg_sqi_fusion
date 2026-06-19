from pathlib import Path
import json
from src.transformer_pipeline.external_benchmarks.run import preprocess_butqdb
root = Path(r"E:\GPTProject2\ecg")
res = preprocess_butqdb(
    data_root=root / "data" / "external",
    out_root=root / "outputs" / "external_benchmarks" / "e311_realdata_2026_06_02",
    seed=0,
    force=True,
)
print(json.dumps(res, indent=2, ensure_ascii=False))
