"""
regAutomata — comprehensive usage examples.
Demonstrates every public parameter across all regression types and datasets.
Run:  python test_all_regAutomata.py
"""
import pandas as pd
from pathlib import Path
from regAutomata import (
    run_regAutomata,
    predictRegAutomata,
    load_artifacts,
    get_correlation_matrix,
    SUPPORTED_REGRESSION,
    SUPPORTED_CORRELATION,
)


# ── 1. Minimal call ───────────────────────────────────────────────────────────

artifacts = run_regAutomata(
    dataset_csv="data/iris/iris_dataset.csv",
    qS="sepal length (cm)",
    qF="petal width (cm)",
)


# ── 2. Full parameter call ────────────────────────────────────────────────────

artifacts = run_regAutomata(
    # Data
    dataset_csv="data/iris/iris_dataset.csv",

    # Automaton definition
    qS="sepal length (cm)",         # initial state  (must be a column name)
    qF="petal width (cm)",          # final state    (must be a column name)

    # Regression: linear | polynomial | cubic | loess | ridge | lasso | svr | gbr
    regression_type="linear",

    # Correlation: pearson | spearman | kendall  (None = auto per regression type)
    correlation_method="pearson",

    # Graph scope: "full" = all explored paths  |  "best" = highest-corr path only
    visualization_type="full",

    # Output root (default: "regAutomata_results")
    output_dir="regAutomata_results",
    open_html=False,

    # PNG export (requires: pip install playwright && playwright install chromium)
    save_png=False,
    png_kwargs={
        "pad_h": 120,           # horizontal padding — keeps node labels unclipped
        "pad_v": 40,            # vertical padding
        "device_scale": 2,      # 2x pixel density (retina quality)
        "wait_after_load": 1.0,
    },

    # Pre-filter: removes low-correlation pairs before path generation.
    # Threshold: |corr| >= (max + mean) / 2  (adaptive, no fixed cutoff).
    # Strongly recommended for datasets with more than ~6 columns.
    use_prefilter=True,
)

run_dir = Path(artifacts["run_dir"])
print("Best path:", artifacts["paths"][artifacts["best_path_index"]])
print("Run folder:", run_dir)


# ── 3. Prediction from saved artifact file ────────────────────────────────────
# Always use artifacts_best.pkl for prediction — it contains only the
# highest-correlation path and its fitted models.

seq, final_value = predictRegAutomata(
    artifacts_or_path=str(run_dir / "artifacts_best.pkl"),
    x0=5.1,
    qS="sepal length (cm)",
    save_html=None,       # e.g. str(run_dir / "prediction.html")
    save_png_path=None,   # e.g. str(run_dir / "prediction.png")
)

print("\nPrediction sequence:")
for a1, a2, val in seq:
    print(f"  {a1}  ->  {a2}  :  {val:.4f}")
print(f"Final predicted value at '{artifacts['qF']}': {final_value:.4f}")


# ── 4. Prediction from already-loaded dict ────────────────────────────────────

loaded = load_artifacts(str(run_dir / "artifacts_best.pkl"))
seq2, final2 = predictRegAutomata(loaded, x0=6.3, qS="sepal length (cm)")
print(f"\nDirect dict prediction  x0=6.3  ->  {final2:.4f}")


# ── 5. Correlation matrix utility ─────────────────────────────────────────────

df = pd.read_csv("data/iris/iris_dataset.csv")
for method in SUPPORTED_CORRELATION:
    cm = get_correlation_matrix(df, method=method)
    print(f"\nCorrelation matrix ({method}):")
    print(cm.round(3).to_string())


# ── 6. Loop over all regression types ─────────────────────────────────────────

results = {}
for rtype in SUPPORTED_REGRESSION:
    art = run_regAutomata(
        dataset_csv="data/iris/iris_dataset.csv",
        qS="sepal length (cm)",
        qF="petal width (cm)",
        regression_type=rtype,
        save_png=False,
        use_prefilter=True,
    )
    rd = Path(art["run_dir"])
    seq, final = predictRegAutomata(str(rd / "artifacts_best.pkl"), x0=5.1, qS="sepal length (cm)")
    best_path = art["paths"][art["best_path_index"]]
    results[rtype] = {"path": best_path, "prediction": round(final, 4)}
    print(f"[{rtype:12s}]  path={best_path}  pred={final:.4f}")

print("\nSummary:")
for rtype, info in results.items():
    print(f"  {rtype:12s}  ->  {info['prediction']}")


# ── 7. Three datasets — thesis verification ───────────────────────────────────

DATASETS = [
    ("data/iris/iris_dataset.csv",         "sepal length (cm)", "petal width (cm)", 5.1),
    ("data/wine/wine_dataset.csv",         "alcohol",           "proline",          13.0),
    ("data/diabetes/diabetes_dataset.csv", "bmi",               "target",           0.05),
]

for csv, qS, qF, x0 in DATASETS:
    print(f"\n{'─'*50}")
    print(f"Dataset : {csv}  |  {qS}  ->  {qF}")

    art = run_regAutomata(
        dataset_csv=csv,
        qS=qS,
        qF=qF,
        regression_type="polynomial",
        visualization_type="best",
        save_png=False,
        use_prefilter=True,
    )
    rd = Path(art["run_dir"])
    seq, final = predictRegAutomata(str(rd / "artifacts_best.pkl"), x0=x0, qS=qS)
    print(f"Best path : {art['paths'][art['best_path_index']]}")
    print(f"Prediction: {final:.4f}")
    print(f"Output    : {rd}")
