"""
regAutomata — example usage
Demonstrates every public parameter across all regression types and datasets.
"""

from regAutomata import (
    run_regAutomata,
    predictRegAutomata,
    load_artifacts,
    get_correlation_matrix,
    SUPPORTED_REGRESSION,
    SUPPORTED_CORRELATION,
)


# ── 1. Quick single run ────────────────────────────────────────────────────────
# Minimal call — only required parameters, everything else uses defaults.

artifacts = run_regAutomata(
    dataset_csv="data/iris/iris_dataset.csv",
    qS="sepal length (cm)",
    qF="petal width (cm)",
)


# ── 2. Full parameter run ──────────────────────────────────────────────────────
# Every parameter explicitly set so you can see what each one does.

artifacts = run_regAutomata(
    # --- Data ---
    dataset_csv="data/iris/iris_dataset.csv",   # Path to any CSV file

    # --- Automaton definition ---
    qS="sepal length (cm)",                     # Initial state  (must be a column name)
    qF="petal width (cm)",                      # Final state    (must be a column name)

    # --- Regression ---
    # One of: "linear" | "polynomial" | "cubic" | "loess" | "ridge" | "lasso" | "svr" | "gbr"
    regression_type="linear",

    # --- Correlation ---
    # One of: "pearson" | "spearman" | "kendall"
    # If None, a sensible default is chosen automatically per regression type.
    correlation_method="pearson",

    # --- Graph scope ---
    # "full" — show all explored paths in the HTML
    # "best" — show only the highest-correlation path
    visualization_type="full",

    # --- HTML output ---
    output_html="regAutomata_linear_full.html",  # Path for the interactive graph
    open_html=False,                              # Set True to open browser automatically

    # --- Artifacts ---
    # Saves two .pkl files:
    #   regAutomata_linear.pkl        — full graph artifacts
    #   regAutomata_linear_best.pkl   — best-path-only artifacts (used for prediction)
    save_artifacts_path="regAutomata_linear.pkl",

    # --- PNG export (requires: pip install playwright && playwright install chromium) ---
    save_png=False,                              # Set True to capture HTML → PNG
    png_kwargs={
        "out_png": "regAutomata_linear_full.png",
        "extra_left_px": 60,
        "extra_right_px": 60,
        "extra_top_px": 60,
        "extra_bottom_px": 60,
        "device_scale": 2,                       # 2 = 2× pixel density (retina quality)
        "wait_after_load": 1.0,                  # Seconds to wait after vis.js renders
    },

    # --- Pre-filter ---
    # True  — removes low-correlation pairs before path generation.
    #         Threshold: |corr| >= (max + mean) / 2  (adaptive, no fixed cutoff)
    #         Strongly recommended for datasets with more than ~6 columns.
    # False — explore all possible paths (factorial growth, slow on wide datasets)
    use_prefilter=True,
)

print("Best path:", artifacts["paths"][artifacts["best_path_index"]])
print("Regression:", artifacts["regression_type"])
print("Correlation method:", artifacts["correlation_method"])


# ── 3. Prediction from saved artifacts ────────────────────────────────────────
# Always load the *_best.pkl file for prediction — it contains only the
# highest-correlation path and its fitted models.

seq, final_value = predictRegAutomata(
    artifacts_or_path="regAutomata_linear_best.pkl",  # Path or already-loaded dict
    x0=5.1,                                           # Starting value at qS
    qS="sepal length (cm)",                           # Must match the artifact's qS

    # Optional: save a new HTML/PNG with predicted p-values annotated on each node
    save_html=None,           # e.g. "prediction_result.html"
    save_png_path=None,       # e.g. "prediction_result.png"  (requires Playwright)
)

print("\nPrediction sequence:")
for a1, a2, val in seq:
    print(f"  {a1}  →  {a2}  :  {val:.4f}")
print(f"Final predicted value at '{artifacts['qF']}': {final_value:.4f}")


# ── 4. Prediction from an already-loaded artifact dict ────────────────────────
# You can also pass the dict returned by run_regAutomata or load_artifacts
# directly — no file I/O needed.

loaded = load_artifacts("regAutomata_linear_best.pkl")
seq2, final2 = predictRegAutomata(loaded, x0=6.3, qS="sepal length (cm)")
print(f"\nDirect dict prediction  x0=6.3  →  {final2:.4f}")


# ── 5. Correlation matrix utility ─────────────────────────────────────────────
import pandas as pd

df = pd.read_csv("data/iris/iris_dataset.csv")
for method in SUPPORTED_CORRELATION:
    cm = get_correlation_matrix(df, method=method)
    print(f"\nCorrelation matrix ({method}):")
    print(cm.round(3).to_string())


# ── 6. Loop over all regression types ─────────────────────────────────────────
# Useful for comparing model quality across regression algorithms on the same data.

results = {}
for rtype in SUPPORTED_REGRESSION:
    art = run_regAutomata(
        dataset_csv="data/iris/iris_dataset.csv",
        qS="sepal length (cm)",
        qF="petal width (cm)",
        regression_type=rtype,
        save_artifacts_path=f"artifacts_{rtype}.pkl",
        save_png=False,
        use_prefilter=True,
    )
    seq, final = predictRegAutomata(
        f"artifacts_{rtype}_best.pkl",
        x0=5.1,
        qS="sepal length (cm)",
    )
    best_path = art["paths"][art["best_path_index"]]
    results[rtype] = {"path": best_path, "prediction": round(final, 4)}
    print(f"[{rtype:12s}]  path={best_path}  pred={final:.4f}")

print("\nSummary:")
for rtype, info in results.items():
    print(f"  {rtype:12s}  →  {info['prediction']}")


# ── 7. Three datasets — thesis verification ───────────────────────────────────

DATASETS = [
    {
        "dataset_csv": "data/iris/iris_dataset.csv",
        "qS": "sepal length (cm)",
        "qF": "petal width (cm)",
        "x0": 5.1,
    },
    {
        "dataset_csv": "data/wine/wine_dataset.csv",
        "qS": "alcohol",
        "qF": "proline",
        "x0": 13.0,
    },
    {
        "dataset_csv": "data/diabetes/diabetes_dataset.csv",
        "qS": "bmi",
        "qF": "target",
        "x0": 0.05,
    },
]

for ds in DATASETS:
    print(f"\n{'─'*50}")
    print(f"Dataset : {ds['dataset_csv']}")
    print(f"Path    : {ds['qS']}  →  {ds['qF']}")

    art = run_regAutomata(
        dataset_csv=ds["dataset_csv"],
        qS=ds["qS"],
        qF=ds["qF"],
        regression_type="polynomial",
        visualization_type="best",
        output_html=f"graph_{ds['qS'].split()[0]}.html",
        save_artifacts_path=f"artifacts_{ds['qS'].split()[0]}.pkl",
        save_png=False,
        use_prefilter=True,
    )

    seq, final = predictRegAutomata(
        f"artifacts_{ds['qS'].split()[0]}_best.pkl",
        x0=ds["x0"],
        qS=ds["qS"],
    )

    print(f"Best path : {art['paths'][art['best_path_index']]}")
    print(f"Prediction: {final:.4f}")
