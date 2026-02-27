# test_all_regression_types.py
from regAutomata import run_regAutomata, predictRegAutomata, SUPPORTED_REGRESSION

REGRESSIONS = SUPPORTED_REGRESSION  # all 8

for rtype in REGRESSIONS:
    print(f"\n{'─'*40}")
    print(f"Testing: {rtype}")
    try:
        artifacts = run_regAutomata(
            dataset_csv="data/iris/iris_dataset.csv",
            qS="sepal length (cm)",
            qF="petal width (cm)",
            regression_type=rtype,
            save_artifacts_path=f"artifacts_{rtype}.pkl",
            save_png=False,
        )
        seq, final = predictRegAutomata(
            f"artifacts_{rtype}_best.pkl",
            x0=5.0,
            qS="sepal length (cm)",
        )
        print(f"  ✓  final prediction: {final:.4f}")
    except Exception as e:
        print(f"  ✗  {e}")
        