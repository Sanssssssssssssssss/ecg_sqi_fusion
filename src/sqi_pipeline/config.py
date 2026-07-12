from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.paths import project_root


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass(frozen=True)
class SQIPipelineConfig:
    """Resolved configuration for one classical SQI pipeline run.

    Attributes:
        root: Repository root used for all relative paths.
        artifacts_dir: Directory receiving generated pipeline outputs.
        profile: Ordered stage profile, ``baseline`` or ``paper_aligned``.
        seed: Frozen random seed used by split and model stages.
        verbose: Whether stages should emit verbose diagnostics.
        force: Whether reusable stage outputs should be regenerated.
    """

    root: Path
    artifacts_dir: Path
    profile: str = "baseline"
    seed: int = 0
    verbose: bool = False
    force: bool = False

    @classmethod
    def build(
        cls,
        *,
        artifacts_dir: str | Path = "outputs/sqi",
        profile: str = "baseline",
        seed: int = 0,
        verbose: bool = False,
        force: bool = False,
    ) -> "SQIPipelineConfig":
        """Build a repository-relative pipeline configuration.

        Args:
            artifacts_dir: Absolute output path or path relative to the repository.
            profile: Stage profile, ``baseline`` or ``paper_aligned``.
            seed: Random seed recorded in generated artifacts.
            verbose: Enable verbose stage logging.
            force: Force stages to regenerate reusable outputs.

        Returns:
            Fully resolved immutable configuration.

        Raises:
            ValueError: If ``profile`` is unknown.

        Example:
            >>> cfg = SQIPipelineConfig.build(profile="paper_aligned", seed=0)
            >>> cfg.artifacts_dir.is_absolute()
            True
        """

        if profile not in {"baseline", "paper_aligned"}:
            raise ValueError(f"unknown SQI pipeline profile: {profile}")
        root = project_root()
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = root / artifacts_path
        return cls(
            root=root,
            artifacts_dir=artifacts_path,
            profile=profile,
            seed=seed,
            verbose=verbose,
            force=force,
        )

    @property
    def challenge_root(self) -> Path:
        return self.root / "data" / "physionet" / "challenge-2011"

    @property
    def set_a_dir(self) -> Path:
        return self.challenge_root / "set-a"

    @property
    def nstdb_root(self) -> Path:
        return self.root / "data" / "physionet" / "nstdb"

    def base_params(self) -> dict[str, Any]:
        """Return parameters shared by every stage in the selected profile.

        Returns:
            JSON-compatible scalar settings and repository-resolved paths.
        """

        return {
            "profile": self.profile,
            "seed": self.seed,
            "verbose": self.verbose,
            "force": self.force,
            "artifacts_dir": str(self.artifacts_dir),
            "challenge_root": str(self.challenge_root),
            "nstdb_root": str(self.nstdb_root),
            "leads": LEADS_12,
            "fs": {"raw": 500, "noise": 360, "work": 125},
            "half_policy": {"train": "first", "val": "second", "test": "second"},
            "beat_match_tol_ms": 150,
            "snr_db": -6.0,
        }
