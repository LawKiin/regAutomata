"""
Smoke test — quickest sanity check: one dataset, linear regression, predict.
Run:  python test_smoke.py
"""
from pathlib import Path
from regAutomata import run_regAutomata, predictRegAutomata

artifacts = run_regAutomata(
    dataset_csv="data/iris/iris_dataset.csv",
    qS="sepal length (cm)",
    qF="petal width (cm)",
    regression_type="linear",
    visualization_type="full",
    open_html=False,
    save_png=False,     # set True if Playwright is installed
    use_prefilter=True,
)

run_dir = Path(artifacts["run_dir"])
seq, final = predictRegAutomata(
    str(run_dir / "artifacts_best.pkl"),
    x0=5.0,
    qS="sepal length (cm)",
)

print("Path sequence:")
for a1, a2, val in seq:
    print(f"  {a1} -> {a2} : {val:.4f}")
print(f"Final predicted value: {final:.4f}")
print(f"Output folder: {run_dir}")
