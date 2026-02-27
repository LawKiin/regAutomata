"""
Author: Bc. Jozef Lauko
University: University of Matej Bel in Banská Bystrica
Faculty: Faculty of Natural Sciences
Department: KIN FPV
Email: jozef.lauko2@student.umb.sk
Thesis: Regression automata based on correlation sequences

Supported regression types:
    linear | polynomial | cubic | loess | ridge | lasso | svr | gbr
"""

import base64
import copy
import itertools
import os
import pickle
import time
import warnings
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from pyvis.network import Network
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_HTML = "regAutomata.html"
OUTPUT_DIR = Path("regAutomata_output")
OUTPUT_DIR.mkdir(exist_ok=True)

REGRESSION_CORR_DEFAULTS: Dict[str, str] = {
    "linear":     "pearson",
    "polynomial": "spearman",
    "cubic":      "spearman",
    "loess":      "spearman",
    "ridge":      "pearson",
    "lasso":      "pearson",
    "svr":        "spearman",
    "gbr":        "spearman",
}

SUPPORTED_REGRESSION = list(REGRESSION_CORR_DEFAULTS.keys())
SUPPORTED_CORRELATION = ("pearson", "spearman", "kendall")

# ── Utilities ─────────────────────────────────────────────────────────────────

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _ensure_parent_dir(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def calculate_correlation(x, y, method: str) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if method == "pearson":
        return float(pearsonr(x, y)[0])
    if method == "spearman":
        return float(spearmanr(x, y)[0])
    if method == "kendall":
        return float(kendalltau(x, y)[0])
    raise ValueError(f"Invalid correlation method: {method!r}. Choose from {SUPPORTED_CORRELATION}")


def encode_image_from_file(filename: str) -> str:
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Pre-filter ────────────────────────────────────────────────────────────────

def prefilter_pairs(df: pd.DataFrame, method: str = "pearson") -> Set[Tuple[str, str]]:
    """
    Keep only directed pairs (a, b) where |corr(a,b)| >= (max + mean) / 2.
    Reduces the number of paths on wide datasets without a fixed cutoff.
    """
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(num_cols) < 2:
        return {(a, b) for a in num_cols for b in num_cols if a != b}

    abs_corr = df[num_cols].corr(method=method).abs()
    vals = [abs_corr.loc[c1, c2] for c1 in num_cols for c2 in num_cols if c1 != c2]

    if not vals:
        return set()

    threshold = (float(np.nanmax(vals)) + float(np.nanmean(vals))) / 2.0
    return {
        (c1, c2)
        for c1 in num_cols
        for c2 in num_cols
        if c1 != c2 and abs_corr.loc[c1, c2] >= threshold
    }


def get_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).corr(method=method)


# ── Regression: fit & predict ─────────────────────────────────────────────────

def fit_model(x: np.ndarray, y: np.ndarray, regression_type: str) -> Tuple[str, object]:
    """
    Fit a regression model and return (mtype, params).

    mtype "poly_obj"  -> numpy.polynomial.Polynomial (Chebyshev basis, immune to
                         Vandermonde ill-conditioning regardless of data scale).
    mtype "loess"     -> sorted (N, 2) array from statsmodels lowess.
    mtype "sklearn"   -> (fitted_estimator, scaler_x, scaler_y)
    """
    if regression_type not in SUPPORTED_REGRESSION:
        raise ValueError(f"Invalid regression_type: {regression_type!r}. Choose from {SUPPORTED_REGRESSION}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if regression_type in ("linear", "polynomial", "cubic"):
        from numpy.polynomial import polynomial as P
        degree = {"linear": 1, "polynomial": 2, "cubic": 3}[regression_type]
        return ("poly_obj", P.Polynomial.fit(x, y, degree))

    if regression_type == "loess":
        from statsmodels.nonparametric.smoothers_lowess import lowess
        result = lowess(y, x, frac=min(0.5, max(0.1, 10 / len(x))))
        return ("loess", result[result[:, 0].argsort()])

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(x.reshape(-1, 1))
    Y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    if regression_type == "ridge":
        model = Ridge(alpha=1.0)
    elif regression_type == "lasso":
        model = Lasso(alpha=0.05, max_iter=10_000)
    elif regression_type == "svr":
        model = SVR(kernel="rbf", C=10.0, epsilon=0.05)
    elif regression_type == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
    else:
        raise ValueError(f"Unhandled regression type: {regression_type!r}")

    model.fit(X, Y)
    return ("sklearn", (model, scaler_x, scaler_y))


def predict_model(model_info: Tuple[str, object], x_vals: np.ndarray) -> np.ndarray:
    mtype, params = model_info
    x_vals = np.asarray(x_vals, dtype=float)

    if mtype == "poly_obj":
        return params(x_vals)

    if mtype == "loess":
        return np.interp(x_vals, params[:, 0], params[:, 1])

    if mtype == "sklearn":
        model, scaler_x, scaler_y = params
        X = scaler_x.transform(x_vals.reshape(-1, 1))
        return scaler_y.inverse_transform(model.predict(X).reshape(-1, 1)).ravel()

    raise ValueError(f"Unsupported model type: {mtype!r}")


def manual_eval_model(model_info: Tuple[str, object], x: float) -> float:
    return float(predict_model(model_info, np.array([float(x)]))[0])


# ── Regression plot ───────────────────────────────────────────────────────────

def create_regression_plot(
    x,
    y,
    xlabel: str,
    ylabel: str,
    regression_type: str = "linear",
    filename: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[str, object]]:
    """Fit model, plot scatter + regression curve, optionally save PNG."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    model_info = fit_model(x_arr, y_arr, regression_type)
    pred = predict_model(model_info, x_arr)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.scatter(x_arr, y_arr, s=60, alpha=0.65, zorder=3, label="Data")
    x_line = np.linspace(x_arr.min(), x_arr.max(), 300)
    ax.plot(x_line, predict_model(model_info, x_line), linewidth=2.5, color="tab:red", label=regression_type.title())
    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=13)
    plt.tight_layout()

    if filename:
        _ensure_parent_dir(filename)
        plt.savefig(filename, format="png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    return pred, model_info


# ── Path generation ───────────────────────────────────────────────────────────

def generate_paths(
    attributes: List[str],
    qS: str,
    qF: str,
    allowed_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> List[List[str]]:
    """
    Generate all permutation paths from qS to qF through optional intermediates.
    Filtered by allowed_pairs when provided. Always guarantees a direct [qS, qF] path.
    """
    intermediate = [a for a in attributes if a not in (qS, qF)]
    paths: List[List[str]] = []

    for r in range(0, len(intermediate) + 1):
        for perm in itertools.permutations(intermediate, r):
            path = [qS] + list(perm) + [qF]
            if allowed_pairs is not None:
                if not all((path[i], path[i + 1]) in allowed_pairs for i in range(len(path) - 1)):
                    continue
            paths.append(path)

    return paths if paths else [[qS, qF]]


# ── Graph building ────────────────────────────────────────────────────────────

def build_graph(
    df: pd.DataFrame,
    qS: str,
    qF: str,
    correlation_method: str,
    regression_type: str,
    visualization_type: str = "full",
    allowed_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[nx.DiGraph, Dict[str, Tuple[str, float]], Dict[Tuple[str, str, int], Dict], List[List[str]], Optional[int]]:
    """
    Build the regression automaton graph.

    Node IDs are globally unique per path and step: "{attr}_p{path_idx}_s{step}"
    for intermediate nodes and "{attr}_p{path_idx}_final" for final nodes.
    Regression plots are cached per (a1, a2) pair and reused across paths.
    """
    G = nx.DiGraph()
    image_data_dict: Dict[str, Tuple[str, float]] = {}
    models: Dict[Tuple[str, str, int], Dict] = {}
    pair_cache: Dict[Tuple[str, str], Tuple[np.ndarray, object, float, str]] = {}

    paths = generate_paths(list(df.columns), qS, qF, allowed_pairs)
    best_path_idx: Optional[int] = None
    best_avg_corr: float = -1.0

    G.add_node(qS, label=qS, shape="dot", size=50)

    for path_idx, path in enumerate(paths):
        prev_node_id = qS
        path_corrs: List[float] = []

        for step_i in range(1, len(path)):
            a1, a2 = path[step_i - 1], path[step_i]
            is_final = (step_i == len(path) - 1)
            node2_id = f"{a2}_p{path_idx}_final" if is_final else f"{a2}_p{path_idx}_s{step_i}"

            if node2_id not in G:
                G.add_node(node2_id, label=a2, shape="square")

                if (a1, a2) not in pair_cache:
                    filename = str(OUTPUT_DIR / f"{a1}_to_{a2}_{regression_type}.png")
                    pred, model_info = create_regression_plot(df[a1], df[a2], a1, a2, regression_type, filename)
                    rmse = round(calculate_rmse(df[a2].to_numpy(), pred), 4)
                    pair_cache[(a1, a2)] = (pred, model_info, rmse, encode_image_from_file(filename))

                _, model_info, rmse, img_b64 = pair_cache[(a1, a2)]
                image_data_dict[node2_id] = (img_b64, rmse)
                models[(a1, a2, path_idx)] = {"model_info": model_info, "rmse": rmse}

            corr = round(calculate_correlation(df[a1], df[a2], correlation_method), 4)
            path_corrs.append(abs(corr))
            if not G.has_edge(prev_node_id, node2_id):
                G.add_edge(prev_node_id, node2_id, correlation=f"{corr}")
            prev_node_id = node2_id

        avg_corr = round(float(np.mean(path_corrs)), 4) if path_corrs else 0.0
        G.nodes[prev_node_id]["label"] += f"\n\nμ(corr) = {avg_corr}"
        if prev_node_id in image_data_dict:
            G.nodes[prev_node_id]["label"] += f"\nRMSE = {image_data_dict[prev_node_id][1]}"

        if avg_corr > best_avg_corr:
            best_avg_corr = avg_corr
            best_path_idx = path_idx

    if visualization_type == "best" and best_path_idx is not None:
        best_path = paths[best_path_idx]
        keep_nodes: Set[str] = {qS}
        keep_edges: Set[Tuple[str, str]] = set()
        prev = qS

        for step_i in range(1, len(best_path)):
            a2 = best_path[step_i]
            is_final = step_i == len(best_path) - 1
            nid = f"{a2}_p{best_path_idx}_final" if is_final else f"{a2}_p{best_path_idx}_s{step_i}"
            keep_nodes.add(nid)
            keep_edges.add((prev, nid))
            prev = nid

        for n in list(G.nodes):
            if n not in keep_nodes:
                G.remove_node(n)
        for e in list(G.edges):
            if e not in keep_edges:
                G.remove_edge(*e)

        image_data_dict = {k: v for k, v in image_data_dict.items() if k in keep_nodes}
        models = {k: v for k, v in models.items() if k[2] == best_path_idx}

    return G, image_data_dict, models, paths, best_path_idx


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(
    G: nx.DiGraph,
    image_data_dict: Dict[str, Tuple[str, float]],
    output_html: str = DEFAULT_HTML,
    open_html: bool = False,
) -> int:
    """
    Render the graph as an interactive HTML file.
    All layout parameters scale automatically with the number of paths and graph depth.
    Returns the canvas height in pixels (used by capture_pyvis_html_as_png).
    """
    num_paths = max(1, sum(1 for n in G.nodes() if "_final" in str(n)))

    try:
        max_depth = nx.dag_longest_path_length(G)
    except Exception:
        max_depth = 2

    # All layout values derived from num_paths and max_depth so the graph
    # never clips or overlaps regardless of dataset size.
    node_spacing = max(180, min(500, 1600 // num_paths))
    level_sep    = max(400, min(900, 900 - max_depth * 40))
    height_px    = max(1000, num_paths * node_spacing + 500)
    img_size     = max(60,  min(130, 260 // num_paths))
    font_size    = max(14,  min(30,  img_size // 4))

    print(
        f"[visualize] paths={num_paths}  depth={max_depth}  "
        f"height={height_px}px  node_spacing={node_spacing}  "
        f"level_sep={level_sep}  img_size={img_size}"
    )

    net = Network(height=f"{height_px}px", width="100%", directed=True)

    for node_id, data in G.nodes(data=True):
        label = data.get("label", node_id)
        img_info = image_data_dict.get(node_id)

        if img_info and data.get("shape") == "square":
            img_b64, rmse = img_info
            net.add_node(
                node_id,
                label=label if "μ" in label else f"{label}\n\nRMSE = {rmse}",
                title=label,
                shape="image",
                image=f"data:image/png;base64,{img_b64}",
                size=img_size,
                color={"border": "black"},
                font={"size": font_size, "background": "white"},
            )
        else:
            net.add_node(
                node_id,
                label=label,
                title=label,
                shape="dot",
                size=20,
                font={"size": font_size, "background": "white"},
                color={
                    "background": "black",
                    "border": "black",
                    "highlight": {"background": "white", "border": "black"},
                },
            )

    for u, v, d in G.edges(data=True):
        net.add_edge(
            u, v,
            label=str(d.get("correlation", "")),
            arrows="to",
            font={"size": font_size, "background": "white"},
        )

    net.set_options(f"""{{
      "layout": {{
        "hierarchical": {{
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": {level_sep},
          "nodeSpacing": {node_spacing},
          "blockShifting": false,
          "edgeMinimization": false,
          "parentCentralization": false
        }}
      }},
      "physics": {{"enabled": false}},
      "edges": {{
        "smooth": {{"enabled": true, "type": "diagonal", "roundness": 0.5}},
        "width": 2,
        "font": {{"size": {font_size}, "align": "middle", "background": "white"}}
      }},
      "nodes": {{
        "font": {{"size": {font_size}, "background": "white"}}
      }}
    }}""")

    net.write_html(output_html)
    if open_html:
        webbrowser.open("file://" + os.path.realpath(output_html))

    return height_px


# ── HTML → PNG capture ────────────────────────────────────────────────────────

def capture_pyvis_html_as_png(
    html_path: str,
    out_png: str = "regAutomata_exact.png",
    canvas_height_px: int = 1200,
    extra_left_px: int = 50,
    extra_right_px: int = 50,
    extra_top_px: int = 50,
    extra_bottom_px: int = 50,
    device_scale: int = 2,
    wait_after_load: float = 0.5,
) -> str:
    """
    Screenshot a pyvis HTML file to PNG via Playwright.
    canvas_height_px should be the value returned by visualize().
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && playwright install"
        ) from exc

    html_uri = Path(html_path).absolute().as_uri()
    tmp_png = Path("__pyvis_tmp_screenshot.png")

    # Viewport sized to exactly match the canvas — do NOT resize canvas.height
    # after load, that is a full canvas clear in the HTML5 API.
    viewport_h = canvas_height_px + extra_top_px + extra_bottom_px + 100

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": viewport_h},
            device_scale_factor=device_scale,
        )
        page = context.new_page()
        page.goto(html_uri)
        page.wait_for_load_state("networkidle")

        # vis.js draws asynchronously — poll pixel data until rendering is done.
        page.wait_for_function("""() => {
            const c = document.querySelector('#mynetwork canvas');
            if (!c) return false;
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            return d.some(v => v !== 0);
        }""", timeout=15000)

        time.sleep(wait_after_load)
        page.screenshot(path=str(tmp_png), full_page=True)
        browser.close()

    im = Image.open(tmp_png).convert("RGBA")

    new_w = im.size[0] + extra_left_px + extra_right_px
    new_h = im.size[1] + extra_top_px + extra_bottom_px
    canvas = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    canvas.paste(im, (extra_left_px, extra_top_px), mask=im)
    im = canvas

    if im.size[1] > im.size[0]:
        nw = int(im.size[1] * 1.2)
        widened = Image.new("RGBA", (nw, im.size[1]), (255, 255, 255, 255))
        widened.paste(im, ((nw - im.size[0]) // 2, 0), mask=im)
        im = widened

    im.convert("RGB").save(out_png, format="PNG")
    try:
        tmp_png.unlink()
    except Exception:
        pass

    return out_png


# ── Prediction ────────────────────────────────────────────────────────────────

def predictRegAutomata(
    artifacts_or_path,
    x0: float,
    qS: str,
    save_html: Optional[str] = None,
    save_png_path: Optional[str] = None,
) -> Tuple[List[Tuple[str, str, float]], float]:
    """
    Walk the best path using fitted models, predicting sequentially from x0 at qS.
    Returns (sequence of (a1, a2, predicted_value) tuples, final predicted value).
    """
    if isinstance(artifacts_or_path, (str, Path)):
        with open(artifacts_or_path, "rb") as f:
            artifacts = pickle.load(f)
    else:
        artifacts = artifacts_or_path

    models = artifacts["models"]
    paths = artifacts["paths"]
    best_idx = artifacts["best_path_index"]
    G = artifacts["G"].copy()
    image_data_dict = copy.copy(artifacts["image_data_dict"])

    if best_idx is None:
        raise ValueError("Artifacts contain no best_path_index.")

    path = paths[best_idx]
    cur = float(x0)
    seq: List[Tuple[str, str, float]] = []
    node_values: Dict[str, float] = {path[0]: cur}

    for step_i in range(1, len(path)):
        a1, a2 = path[step_i - 1], path[step_i]
        key = (a1, a2, best_idx)
        model_entry = models.get(key) or next(
            (v for k, v in models.items() if k[0] == a1 and k[1] == a2), None
        )
        if model_entry is None:
            raise KeyError(f"No regression model found for edge ({a1} -> {a2}).")

        yhat = manual_eval_model(model_entry["model_info"], cur)
        seq.append((a1, a2, float(yhat)))
        cur = yhat
        node_values[a2] = cur

    for node_id in list(G.nodes):
        for attr, val in node_values.items():
            if node_id.startswith(attr + "_p") or node_id == attr:
                lines = G.nodes[node_id].get("label", attr).split("\n")
                attr_line = next((l for l in lines if l.strip()), attr)
                rest = [l for l in lines if l.startswith("μ") or l.startswith("RMSE")]
                G.nodes[node_id]["label"] = "\n".join([attr_line, f"p = {val:.4f}"] + rest)
                break

    if save_html:
        canvas_h = visualize(G, image_data_dict, output_html=save_html, open_html=False)
        if save_png_path:
            capture_pyvis_html_as_png(
                save_html,
                out_png=save_png_path,
                canvas_height_px=canvas_h,
                extra_left_px=80,
                extra_right_px=80,
                extra_bottom_px=80,
                device_scale=2,
            )

    return seq, cur


# ── Persistence ───────────────────────────────────────────────────────────────

def save_artifacts(artifacts: Dict, path: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)


def load_artifacts(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_regAutomata(
    dataset_csv: str,
    qS: str,
    qF: str,
    regression_type: str = "linear",
    correlation_method: Optional[str] = None,
    visualization_type: str = "full",
    output_html: str = DEFAULT_HTML,
    open_html: bool = False,
    save_artifacts_path: Optional[str] = None,
    save_png: bool = False,
    png_kwargs: Optional[Dict] = None,
    use_prefilter: bool = True,
) -> Dict:
    """
    End-to-end pipeline: load CSV -> pre-filter pairs -> build graph -> visualise -> save artifacts.

    Parameters
    ----------
    dataset_csv        : path to CSV file
    qS                 : source attribute (initial state)
    qF                 : target attribute (final state)
    regression_type    : one of SUPPORTED_REGRESSION
    correlation_method : one of SUPPORTED_CORRELATION (auto-selected if None)
    visualization_type : "full" | "best"
    output_html        : path for the interactive HTML output
    open_html          : open browser after generation
    save_artifacts_path: pickles full artifacts and a *_best.pkl variant
    save_png           : capture HTML to PNG via Playwright
    png_kwargs         : extra kwargs forwarded to capture_pyvis_html_as_png
    use_prefilter      : remove low-correlation pairs before path generation
    """
    if regression_type not in SUPPORTED_REGRESSION:
        raise ValueError(f"regression_type {regression_type!r} not supported. Choose from {SUPPORTED_REGRESSION}")

    df = pd.read_csv(dataset_csv)
    missing = [c for c in (qS, qF) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataset: {missing}")

    if correlation_method is None:
        correlation_method = REGRESSION_CORR_DEFAULTS[regression_type]

    allowed_pairs: Optional[Set[Tuple[str, str]]] = None
    if use_prefilter:
        allowed_pairs = prefilter_pairs(df, method=correlation_method)
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        for col in num_cols:
            if col != qS:
                allowed_pairs.add((col, qF))
            if col != qF:
                allowed_pairs.add((qS, col))
        allowed_pairs.add((qS, qF))
        n_possible = len(num_cols) * (len(num_cols) - 1)
        print(f"[prefilter] {len(allowed_pairs)}/{n_possible} directed pairs retained (threshold=(max+mean)/2 of |corr|)")

    G_full, image_data_full, models_full, paths_full, best_path_idx = build_graph(
        df, qS, qF, correlation_method, regression_type,
        visualization_type="full",
        allowed_pairs=allowed_pairs,
    )

    print(
        f"[build_graph] {len(paths_full)} paths explored | "
        f"best path index: {best_path_idx} | "
        f"path: {paths_full[best_path_idx] if best_path_idx is not None else 'N/A'}"
    )

    canvas_h = visualize(G_full, image_data_full, output_html=output_html, open_html=open_html)

    artifacts_full: Dict = {
        "G": G_full,
        "image_data_dict": image_data_full,
        "models": models_full,
        "paths": paths_full,
        "best_path_index": best_path_idx,
        "qS": qS,
        "qF": qF,
        "regression_type": regression_type,
        "correlation_method": correlation_method,
        "visualization_type": "full",
        "dataset_csv": dataset_csv,
        "allowed_pairs": allowed_pairs,
    }

    if save_artifacts_path:
        save_artifacts(artifacts_full, save_artifacts_path)

        base, ext = os.path.splitext(save_artifacts_path)
        best_pkl = f"{base}_best{ext}"

        G_best, image_best, models_best, paths_best, _ = build_graph(
            df, qS, qF, correlation_method, regression_type,
            visualization_type="best",
            allowed_pairs=allowed_pairs,
        )
        artifacts_best: Dict = {
            **artifacts_full,
            "G": G_best,
            "image_data_dict": image_best,
            "models": models_best,
            "paths": paths_best,
            "visualization_type": "best",
        }
        save_artifacts(artifacts_best, best_pkl)
        print(f"[artifacts] saved: {save_artifacts_path}  |  best: {best_pkl}")

    if save_png:
        kwargs = {
            "canvas_height_px": canvas_h,
            "extra_left_px": 50,
            "extra_right_px": 50,
            "extra_top_px": 50,
            "extra_bottom_px": 50,
            "device_scale": 2,
        }
        if png_kwargs:
            kwargs.update(png_kwargs)
        time.sleep(0.2)
        capture_pyvis_html_as_png(output_html, **kwargs)

    return artifacts_full