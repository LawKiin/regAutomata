import pytest
from pathlib import Path
from regAutomata import run_regAutomata, predictRegAutomata


def test_run_and_predict(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("x,y\n1,2\n2,4\n3,6\n")

    artifacts = run_regAutomata(
        dataset_csv=str(csv_file),
        qS="x",
        qF="y",
        regression_type="linear",
        output_dir=str(tmp_path / "results"),
    )

    assert "models" in artifacts
    assert "run_dir" in artifacts

    run_dir = Path(artifacts["run_dir"])
    assert (run_dir / "artifacts.pkl").exists()
    assert (run_dir / "artifacts_best.pkl").exists()
    assert (run_dir / "graph.html").exists()
    assert (run_dir / "graph_best.html").exists()
    assert not (run_dir / "_tmp").exists()  # cleaned up

    seq, final = predictRegAutomata(
        str(run_dir / "artifacts_best.pkl"),
        x0=2.0,
        qS="x",
    )
    assert isinstance(seq, list)
    assert isinstance(final, float)
