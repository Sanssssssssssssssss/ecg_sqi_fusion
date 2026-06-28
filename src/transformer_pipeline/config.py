from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import project_root


@dataclass(frozen=True)
class TransformerPipelineConfig:
    root: Path
    artifacts_dir: Path
    seed: int = 20260876
    force: bool = False
    verbose: bool = False

    @classmethod
    def build(
        cls,
        *,
        artifacts_dir: str | Path = "outputs/transformer/v116_e31",
        seed: int = 20260876,
        force: bool = False,
        verbose: bool = False,
    ) -> "TransformerPipelineConfig":
        root = project_root()
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = root / artifacts_path
        return cls(root=root, artifacts_dir=artifacts_path, seed=seed, force=force, verbose=verbose)

    @property
    def butqdb_root(self) -> Path:
        return self.root / "data" / "external" / "butqdb_1_0_0"

    @property
    def ptbxl_root(self) -> Path:
        return self.root / "data" / "ptb-xl"

    @property
    def analysis_dir(self) -> Path:
        return self.artifacts_dir / "analysis" / "good_medium_geometry_repair"

    @property
    def report_dir(self) -> Path:
        return self.artifacts_dir / "reports" / "analysis" / "good_medium_geometry_repair"

    @property
    def figures_dir(self) -> Path:
        return self.artifacts_dir / "figures"
