from regAutomata import *


if __name__ == "__main__":
    artifacts = run_regAutomata(
        dataset_csv="data/iris/iris_dataset.csv",
        qS="sepal length (cm)",
        qF="petal width (cm)",
        regression_type="linear",
        visualization_type="full",
        output_html="regAutomata_linear_full.html",
        open_html=False,
        save_artifacts_path="art_linear_full.pkl",
        save_png=True,
        png_kwargs={"out_png": "regAutomata_linear_full.png", "extra_left_px": 50, "extra_right_px": 50, "extra_bottom_px": 50, "device_scale": 3},
    )
    seq, final = predictRegAutomata(
        "art_linear_full_best.pkl",
        x0=5.0,
        qS="sepal length (cm)",
        save_html="prediction_best.html",
        save_png_path="prediction_best.png",
    )
    print("Predikčná sekvencia:", seq)
    print("Finálna hodnota:", final)
