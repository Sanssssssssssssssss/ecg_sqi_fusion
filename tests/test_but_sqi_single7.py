import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.supplemental_transformer_experiments.but_sqi_baseline import run as but_sqi


def test_but_sqi_single7_feature_contract():
    names = but_sqi._feature_names([but_sqi.SINGLE_LEAD])

    assert len(names) == 7
    assert names == [f"{but_sqi.SINGLE_LEAD}__{sqi}" for sqi in but_sqi.SQI_ORDER]


if __name__ == "__main__":
    test_but_sqi_single7_feature_contract()
