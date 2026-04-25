"""
Singular dataset test — run one dataset with one regression type.
Edit the CONFIG block and run:  python test_singular_dataset.py
"""
import time
import pandas as pd
from pathlib import Path
from regAutomata import run_regAutomata, predictRegAutomata

# ── CONFIG ────────────────────────────────────────────────────────────────────

CSV        = "data/iris/iris_dataset.csv"
QS         = "sepal length (cm)"
QF         = "petal width (cm)"
REGRESSION = "linear"   # linear | polynomial | cubic | loess | ridge | lasso | svr | gbr
OPEN_HTML  = False       # set True to open browser when done

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print(f"Dataset   : {CSV}")
    print(f"Path      : {QS}  ->  {QF}")
    print(f"Regression: {REGRESSION}\n")

    artifacts = run_regAutomata(
        dataset_csv=CSV,
        qS=QS,
        qF=QF,
        regression_type=REGRESSION,
        visualization_type="full",
        open_html=OPEN_HTML,
        save_png=True,    # requires: pip install playwright && playwright install chromium
        use_prefilter=True,
    )

    run_dir = Path(artifacts["run_dir"])
    x0 = float(pd.read_csv(CSV)[QS].mean())
    seq, final = predictRegAutomata(
        str(run_dir / "artifacts_best.pkl"),
        x0=x0,
        qS=QS,
    )

    print(f"\nBest path : {artifacts['paths'][artifacts['best_path_index']]}")
    print(f"Input x0  : {x0:.4f}  ({QS})")
    for a1, a2, v in seq:
        print(f"  {a1}  ->  {a2}  :  {v:.4f}")
    print(f"Final pred: {final:.4f}  ({QF})")
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Output folder: {run_dir}")
