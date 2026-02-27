# test_smoke.py
from regAutomata import run_regAutomata, predictRegAutomata

artifacts = run_regAutomata(
    dataset_csv="data/iris/iris_dataset.csv",
    qS="sepal length (cm)",
    qF="petal width (cm)",
    regression_type="linear",
    visualization_type="full",
    output_html="test_linear.html",
    save_artifacts_path="test_linear.pkl",
    open_html=False,
    save_png=True,         # no Playwright needed
    use_prefilter=True,
)

seq, final = predictRegAutomata(
    "test_linear_best.pkl",
    x0=5.0,
    qS="sepal length (cm)",
)
print("Path sequence:")
for a1, a2, val in seq:
    print(f"  {a1} → {a2} : {val:.4f}")
print(f"Final predicted value: {final:.4f}")
