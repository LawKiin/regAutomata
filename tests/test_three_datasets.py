# test_three_datasets.py
from regAutomata import run_regAutomata, predictRegAutomata

DATASETS = [
    ("data/iris/iris_dataset.csv",     "sepal length (cm)", "petal width (cm)"),
    ("data/wine/wine_dataset.csv",     "alcohol",           "proline"),
    ("data/diabetes/diabetes_dataset.csv", "bmi",           "target"),  # adjust if needed
]

for csv, qS, qF in DATASETS:
    print(f"\n{'═'*50}")
    print(f"Dataset: {csv}  |  {qS} → {qF}")
    try:
        art = run_regAutomata(csv, qS, qF, regression_type="polynomial", save_png=False)
        seq, final = predictRegAutomata(art, x0=0.05, qS=qS)
        print(f"  Best path: {art['paths'][art['best_path_index']]}")
        print(f"  Final prediction: {final:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
