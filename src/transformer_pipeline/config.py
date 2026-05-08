from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.paths import project_root


@dataclass(frozen=True)
class TransformerPipelineConfig:
    root: Path
    artifact_dir: Path
    seed: int = 0
    verbose: bool = False
    dry_run: bool = False

    @classmethod
    def build(
        cls,
        *,
        artifact_dir: str | Path = "artifact1",
        seed: int = 0,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> "TransformerPipelineConfig":
        root = project_root()
        path = Path(artifact_dir)
        if not path.is_absolute():
            path = root / path
        return cls(root=root, artifact_dir=path, seed=seed, verbose=verbose, dry_run=dry_run)

    @property
    def ptbxl_root(self) -> Path:
        return self.root / "data" / "ptb-xl"

    @property
    def nstdb_root(self) -> Path:
        return self.root / "data" / "physionet" / "nstdb"

    @property
    def model_dir(self) -> Path:
        return self.artifact_dir / "models" / f"mtl_transformer_seed{self.seed}_step6"

    def base_params(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "verbose": self.verbose,
            "dry_run": self.dry_run,
            "artifact_dir": str(self.artifact_dir),
            "ptbxl_root": str(self.ptbxl_root),
            "nstdb_root": str(self.nstdb_root),
            "model_dir": str(self.model_dir),
        }
