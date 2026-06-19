from pathlib import Path
import json, sys
root = Path(r"E:\GPTProject2\ecg")
sys.path.insert(0, str(root))
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_REALDATA_ROOT,
    PROTOCOLS,
    ensure_but_processed,
    load_current_but,
    make_protocol_dataset,
    write_json,
)
processed = ensure_but_processed(DEFAULT_OUT_ROOT, DEFAULT_REALDATA_ROOT)
X, meta = load_current_but(processed)
spec = [s for s in PROTOCOLS if s.name == "p1_current_10s_center"][0]
audit = make_protocol_dataset(spec, X, meta, DEFAULT_OUT_ROOT / "protocols", force=True)
write_json(DEFAULT_REPORT_ROOT / "protocol_audits" / f"{spec.name}.json", audit)
print(json.dumps(audit, indent=2, ensure_ascii=False))
