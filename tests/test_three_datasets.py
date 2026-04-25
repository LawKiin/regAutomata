"""
Three-dataset test — polynomial regression on iris, wine, diabetes.
Run:  python test_three_datasets.py
"""
from pathlib import Path
from regAutomata import run_regAutomata, predictRegAutomata

DATASETS = [
    ("data/iris/iris_dataset.csv",         "sepal length (cm)", "petal width (cm)", 5.1),
    ("data/wine/wine_dataset.csv",         "alcohol",           "proline",          13.0),
    ("data/diabetes/diabetes_dataset.csv", "bmi",               "target",           0.05),
]

for csv, qS, qF, x0 in DATASETS:
    print(f"\n{'='*50}")
    print(f"Dataset: {csv}  |  {qS} -> {qF}")
    try:
        art = run_regAutomata(
            csv, qS, qF,
            regression_type="polynomial",
            visualization_type="best",
            save_png=False,
            use_prefilter=True,
        )
        run_dir = Path(art["run_dir"])
        seq, final = predictRegAutomata(
            str(run_dir / "artifacts_best.pkl"),
            x0=x0,
            qS=qS,
        )
        print(f"  Best path : {art['paths'][art['best_path_index']]}")
        print(f"  Prediction: {final:.4f}")
        print(f"  Output    : {run_dir}")
    except Exception as e:
        print(f"  ERROR: {e}")
