import os
import pytest
from regAutomata import run_regAutomata, predictRegAutomata


def test_run_and_predict(tmp_path):
    # create simple CSV
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("x,y\n1,2\n2,4\n3,6\n")

    artifacts = run_regAutomata(
        dataset_csv=str(csv_file),
        qS="x",
        qF="y",
        regression_type="linear",
        save_artifacts_path=str(tmp_path / "artifacts.pkl"),
    )

    assert "models" in artifacts
    seq, final = predictRegAutomata(str(tmp_path / "artifacts_best.pkl"), x0=2.0, qS="x")
    assert isinstance(seq, list)
    assert isinstance(final, float)
