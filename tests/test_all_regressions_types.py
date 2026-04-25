"""
Regression type sweep — tests all 8 regression types on the iris dataset.
Run:  python test_all_regressions_types.py
"""
from pathlib import Path
from regAutomata import run_regAutomata, predictRegAutomata, SUPPORTED_REGRESSION

for rtype in SUPPORTED_REGRESSION:
    print(f"\n{'─'*40}")
    print(f"Testing: {rtype}")
    try:
        artifacts = run_regAutomata(
            dataset_csv="data/iris/iris_dataset.csv",
            qS="sepal length (cm)",
            qF="petal width (cm)",
            regression_type=rtype,
            save_png=False,
        )
        run_dir = Path(artifacts["run_dir"])
        seq, final = predictRegAutomata(
            str(run_dir / "artifacts_best.pkl"),
            x0=5.0,
            qS="sepal length (cm)",
        )
        print(f"  OK  best={artifacts['paths'][artifacts['best_path_index']]}  pred={final:.4f}")
    except Exception as e:
        print(f"  FAIL  {e}")
