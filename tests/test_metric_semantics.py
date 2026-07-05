import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sqi_pipeline.models import svm_tables
from src.supplemental_sqi_experiments import common as sqi_common


def test_paper_se_sp_use_unacceptable_then_acceptable_recall():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9, 0.3])

    svm = svm_tables._metrics(y, score, threshold=0.5)
    supp = sqi_common.binary_metrics(y, score, threshold=0.5)

    assert svm["Se"] == supp["Se"] == 2 / 3
    assert svm["Sp"] == supp["Sp"] == 1 / 2
    assert svm["acceptable_recall"] == supp["acceptable_recall"] == 1 / 2
    assert svm["unacceptable_recall"] == supp["unacceptable_recall"] == 2 / 3


if __name__ == "__main__":
    test_paper_se_sp_use_unacceptable_then_acceptable_recall()
