import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sqi_pipeline.noise.make_paper_aligned_balanced_cases import SplitNoiseSegmentSampler


def test_split_noise_sampler_uses_midpoint_halves():
    n = 20000
    signals = {"em": np.zeros((n, 2), dtype=float)}
    sampler = SplitNoiseSegmentSampler(signals, rng=np.random.default_rng(0), stride_s=1.0)

    _, train_start, train_region, _ = sampler.draw_500("em", "train")
    _, val_start, val_region, _ = sampler.draw_500("em", "val")
    _, test_start, test_region, _ = sampler.draw_500("em", "test")

    assert train_region == "train"
    assert val_region == "val"
    assert test_region == "test"
    assert train_start < n // 2
    assert val_start >= n // 2
    assert test_start >= n // 2
    assert val_start < test_start


if __name__ == "__main__":
    test_split_noise_sampler_uses_midpoint_halves()
