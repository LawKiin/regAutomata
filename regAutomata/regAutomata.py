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
import shutil
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

RESULTS_DIR = Path("regAutomata_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

SUPPORTED_REGRESSION  = list(REGRESSION_CORR_DEFAULTS.keys())
SUPPORTED_CORRELATION = ("pearson", "spearman", "kendall")

# ── Utilities ─────────────────────────────────────────────────────────────────

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


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
    Reduces path count on wide datasets without a fixed cutoff.
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
    x, y,
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

    fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
    ax.scatter(x_arr, y_arr, s=50, alpha=0.65, zorder=3, label="Data")
    x_line = np.linspace(x_arr.min(), x_arr.max(), 400)
    ax.plot(x_line, predict_model(model_info, x_line), linewidth=2.5,
            color="tab:red", label=regression_type.title())
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=11)
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
    tmp_dir: Optional[Path] = None,
) -> Tuple[nx.DiGraph, Dict[str, Tuple[str, float]], Dict[Tuple[str, str], Dict], List[List[str]], Optional[int]]:
    """
    Build the regression automaton graph.

    Node ID = "{prev_node_id}__{a2}" — encodes the full path prefix so nodes
    are only shared when the complete history of the path is identical.

    Scatter PNGs are written to tmp_dir, encoded to base64 immediately, then
    deleted. They never accumulate between calls.
    """
    if tmp_dir is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())

    tmp_dir.mkdir(parents=True, exist_ok=True)

    G = nx.DiGraph()
    image_data_dict: Dict[str, Tuple[str, float]] = {}
    models: Dict[Tuple[str, str], Dict] = {}
    pair_cache: Dict[Tuple[str, str], Tuple[np.ndarray, object, float, str]] = {}

    paths = generate_paths(list(df.columns), qS, qF, allowed_pairs)
    best_path_idx: Optional[int] = None
    best_avg_corr: float = -1.0
    final_node_to_path: Dict[str, int] = {}

    G.add_node(qS, label=qS, shape="dot", size=50)

    for path_idx, path in enumerate(paths):
        prev_node_id = qS
        path_corrs: List[float] = []

        for step_i in range(1, len(path)):
            a1, a2 = path[step_i - 1], path[step_i]
            is_final = (step_i == len(path) - 1)
            node2_id = f"{prev_node_id}__{a2}__final" if is_final else f"{prev_node_id}__{a2}"

            if (a1, a2) not in pair_cache:
                tmp_png = tmp_dir / f"{a1}_to_{a2}_{regression_type}.png"
                pred, model_info = create_regression_plot(
                    df[a1], df[a2], a1, a2, regression_type, str(tmp_png)
                )
                rmse = round(calculate_rmse(df[a2].to_numpy(), pred), 4)
                img_b64 = encode_image_from_file(str(tmp_png))
                try:
                    tmp_png.unlink()
                except Exception:
                    pass
                pair_cache[(a1, a2)] = (pred, model_info, rmse, img_b64)

            _, model_info, rmse, img_b64 = pair_cache[(a1, a2)]

            if node2_id not in G:
                G.add_node(node2_id, label=a2, shape="square")
                image_data_dict[node2_id] = (img_b64, rmse)

            if (a1, a2) not in models:
                models[(a1, a2)] = {"model_info": model_info, "rmse": rmse}

            corr = round(calculate_correlation(df[a1], df[a2], correlation_method), 4)
            path_corrs.append(abs(corr))
            if not G.has_edge(prev_node_id, node2_id):
                G.add_edge(prev_node_id, node2_id, correlation=f"{corr}")
            prev_node_id = node2_id

        avg_corr = round(float(np.mean(path_corrs)), 4) if path_corrs else 0.0
        G.nodes[prev_node_id]["label"] += f"\n\nμ(corr) = {avg_corr}"
        if prev_node_id in image_data_dict:
            G.nodes[prev_node_id]["label"] += f"\nRMSE = {image_data_dict[prev_node_id][1]}"
        final_node_to_path[prev_node_id] = path_idx

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
            nid = f"{prev}__{a2}__final" if is_final else f"{prev}__{a2}"
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
        best_pairs = {(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)}
        models = {k: v for k, v in models.items() if k in best_pairs}

    return G, image_data_dict, models, paths, best_path_idx


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(
    G: nx.DiGraph,
    image_data_dict: Dict[str, Tuple[str, float]],
    output_html: str = "regAutomata.html",
    open_html: bool = False,
) -> int:
    """
    Render the graph as an interactive HTML file.
    All layout parameters scale automatically with the number of paths and depth.
    Returns the canvas height in pixels (needed by capture_pyvis_html_as_png).
    """
    num_paths = max(1, sum(1 for n in G.nodes() if str(n).endswith("_final")))

    try:
        max_depth = nx.dag_longest_path_length(G)
    except Exception:
        max_depth = 2

    node_spacing = max(250, min(600, 2000 // num_paths))
    level_sep    = max(300, min(750, 750 - max_depth * 40))
    # height_px only needs to be large enough for the browser to render all nodes.
    # PNG capture calls network.fit() before screenshotting, so actual empty
    # space is eliminated by content-aware crop rather than this formula.
    height_px    = max(900, num_paths * node_spacing + 300)
    img_size     = max(60,  min(130, 260 // num_paths))
    font_size    = max(16,  min(36,  img_size // 4))

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
          "treeSpacing": {node_spacing},
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": true
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

    _ensure_parent_dir(output_html)
    net.write_html(output_html)
    if open_html:
        webbrowser.open("file://" + os.path.realpath(output_html))

    return height_px


# ── HTML → PNG capture ────────────────────────────────────────────────────────

def capture_pyvis_html_as_png(
    html_path: str,
    out_png: str = "regAutomata.png",
    canvas_height_px: int = 1200,
    pad_h: int = 140,
    pad_v: int = 60,
    device_scale: int = 2,
    wait_after_load: float = 0.8,
) -> str:
    """
    Screenshot a pyvis HTML file to PNG via Playwright.

    Strategy
    --------
    1. Load the HTML in a headless browser with a viewport tall enough for
       vis.js to lay out all nodes (canvas_height_px from visualize()).
    2. Wait for vis.js to finish drawing (pixel-poll on the canvas).
    3. Call network.fit({animation:false}) — zooms/pans so all nodes are
       packed tightly inside the visible canvas area.  This is what eliminates
       the large empty areas: without fit() vis.js leaves the original
       hierarchical layout centred in a mostly-blank canvas.
    4. Screenshot just the <canvas> element (not the full div).
       The canvas pixel buffer contains exactly what vis.js drew.
    5. Content-aware crop: find the bounding box of pixels that differ from
       the canvas background colour (threshold < 230 to catch faint edges
       and light-coloured scatter plot dots), then re-add pad_h / pad_v.
       pad_h is wider than pad_v because node labels extend horizontally
       past the node thumbnail.

    canvas_height_px must equal the value returned by visualize() so the
    viewport is tall enough for vis.js to complete layout before fit().
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && playwright install"
        ) from exc

    html_uri = Path(html_path).absolute().as_uri()
    tmp_png  = Path("__pyvis_tmp_screenshot.png")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": canvas_height_px + 400},
            device_scale_factor=device_scale,
        )
        page = context.new_page()
        page.goto(html_uri)
        page.wait_for_load_state("networkidle")

        # vis.js draws asynchronously — wait until the canvas has at least one
        # non-zero (non-background) pixel.
        page.wait_for_function("""() => {
            const c = document.querySelector('#mynetwork canvas');
            if (!c) return false;
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            for (let i = 0; i < d.length; i += 4) {
                if (d[i] < 250 || d[i+1] < 250 || d[i+2] < 250) return true;
            }
            return false;
        }""", timeout=20000)

        time.sleep(wait_after_load)

        # network.fit() packs all nodes into the visible canvas area, which
        # directly removes the empty space that appears when the hierarchical
        # layout places nodes in a small region of a large canvas.
        page.evaluate("""() => {
            if (typeof network !== 'undefined') {
                network.fit({ animation: false });
            }
        }""")
        # Small pause for the fit() reflow to complete before screenshotting.
        time.sleep(0.4)

        # Screenshot the <canvas> element — its pixel buffer is exactly what
        # vis.js drew, with no surrounding div whitespace.
        page.locator("#mynetwork canvas").screenshot(path=str(tmp_png))
        browser.close()

    # Content-aware crop: threshold at 230 (not 245) to reliably catch faint
    # grey edges and light-coloured scatter plot points.
    im  = Image.open(tmp_png).convert("RGB")
    arr = np.array(im)

    non_bg = np.any(arr < 230, axis=2)
    rows   = np.where(non_bg.any(axis=1))[0]
    cols   = np.where(non_bg.any(axis=0))[0]

    if rows.size and cols.size:
        top    = max(0,            int(rows[0])  - pad_v)
        bottom = min(arr.shape[0], int(rows[-1]) + pad_v)
        left   = max(0,            int(cols[0])  - pad_h)
        right  = min(arr.shape[1], int(cols[-1]) + pad_h)
        im = im.crop((left, top, right, bottom))

    _ensure_parent_dir(out_png)
    im.save(out_png, format="PNG")
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
    Walk the best path using fitted models, predicting sequentially from x0.
    artifacts_or_path may be a file path string/Path or an already-loaded dict.
    Returns (sequence of (a1, a2, predicted_value) tuples, final predicted value).
    """
    if isinstance(artifacts_or_path, (str, Path)):
        with open(artifacts_or_path, "rb") as f:
            artifacts = pickle.load(f)
    else:
        artifacts = artifacts_or_path

    models    = artifacts["models"]
    paths     = artifacts["paths"]
    best_idx  = artifacts["best_path_index"]
    G         = artifacts["G"].copy()
    image_data_dict = copy.copy(artifacts["image_data_dict"])

    if best_idx is None:
        raise ValueError("Artifacts contain no best_path_index.")

    path = paths[best_idx]
    cur  = float(x0)
    seq: List[Tuple[str, str, float]] = []
    node_values: Dict[str, float]     = {path[0]: cur}

    for step_i in range(1, len(path)):
        a1, a2 = path[step_i - 1], path[step_i]
        model_entry = models.get((a1, a2))
        if model_entry is None:
            raise KeyError(f"No regression model found for edge ({a1} -> {a2}).")

        yhat = manual_eval_model(model_entry["model_info"], cur)
        seq.append((a1, a2, float(yhat)))
        cur = yhat
        node_values[a2] = cur

    for node_id in list(G.nodes):
        for attr, val in node_values.items():
            if (node_id == attr
                    or f"__{attr}__" in node_id
                    or node_id.endswith(f"__{attr}")
                    or node_id.endswith(f"__{attr}__final")):
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
                pad_h=140,
                pad_v=60,
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
    output_dir: Optional[str] = None,
    open_html: bool = False,
    save_png: bool = False,
    png_kwargs: Optional[Dict] = None,
    use_prefilter: bool = True,
) -> Dict:
    """
    End-to-end pipeline: load CSV -> pre-filter -> build graph -> visualise -> save.

    Creates one self-contained folder per run:
        {output_dir}/{dataset}__{qS}_to_{qF}__{regression}/
            graph.html          — full graph, interactive
            graph.png           — full graph PNG  (if save_png=True)
            graph_best.html     — best-path graph, interactive
            graph_best.png      — best-path graph PNG  (if save_png=True)
            artifacts.pkl       — full artifacts
            artifacts_best.pkl  — best-path artifacts (used by predictRegAutomata)

    Scatter PNGs are written to _tmp/ inside the run folder, encoded to base64,
    then _tmp/ is deleted before the function returns.

    The returned dict includes a "run_dir" key with the path to the run folder,
    so downstream code can locate artifacts without re-constructing the path.

    Parameters
    ----------
    dataset_csv      : path to CSV file
    qS               : source attribute (initial state)
    qF               : target attribute (final state)
    regression_type  : one of SUPPORTED_REGRESSION
    correlation_method : one of SUPPORTED_CORRELATION  (auto-selected if None)
    visualization_type : "full" | "best"  — scope of the returned full graph
    output_dir       : root results folder  (default: "regAutomata_results")
    open_html        : open browser after generation
    save_png         : capture HTML to PNG via Playwright
    png_kwargs       : extra kwargs forwarded to capture_pyvis_html_as_png
                       (e.g. pad_h, pad_v, device_scale, wait_after_load)
    use_prefilter    : remove low-correlation pairs before path generation
    """
    if regression_type not in SUPPORTED_REGRESSION:
        raise ValueError(
            f"regression_type {regression_type!r} not supported. "
            f"Choose from {SUPPORTED_REGRESSION}"
        )

    df = pd.read_csv(dataset_csv)
    missing = [c for c in (qS, qF) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataset: {missing}")

    if correlation_method is None:
        correlation_method = REGRESSION_CORR_DEFAULTS[regression_type]

    ds_stem  = Path(dataset_csv).stem
    qs_safe  = qS.replace(" ", "_").replace("/", "_")
    qf_safe  = qF.replace(" ", "_").replace("/", "_")
    run_name = f"{ds_stem}__{qs_safe}_to_{qf_safe}__{regression_type}"

    run_dir = (Path(output_dir) if output_dir else RESULTS_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

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
        print(
            f"[prefilter] {len(allowed_pairs)}/{n_possible} directed pairs retained "
            f"(threshold=(max+mean)/2 of |corr|)"
        )

    # ── Full graph ────────────────────────────────────────────────────────────
    G_full, img_full, models_full, paths_full, best_idx = build_graph(
        df, qS, qF, correlation_method, regression_type,
        visualization_type=visualization_type,
        allowed_pairs=allowed_pairs,
        tmp_dir=tmp_dir,
    )

    print(
        f"[build_graph] {len(paths_full)} paths explored | "
        f"best index: {best_idx} | "
        f"path: {paths_full[best_idx] if best_idx is not None else 'N/A'}"
    )

    html_full  = str(run_dir / "graph.html")
    canvas_h_full = visualize(G_full, img_full, output_html=html_full, open_html=open_html)

    artifacts_full: Dict = {
        "G":                  G_full,
        "image_data_dict":    img_full,
        "models":             models_full,
        "paths":              paths_full,
        "best_path_index":    best_idx,
        "qS":                 qS,
        "qF":                 qF,
        "regression_type":    regression_type,
        "correlation_method": correlation_method,
        "visualization_type": visualization_type,
        "dataset_csv":        dataset_csv,
        "allowed_pairs":      allowed_pairs,
        "run_dir":            str(run_dir),
    }
    save_artifacts(artifacts_full, str(run_dir / "artifacts.pkl"))

    # ── Best-path graph ───────────────────────────────────────────────────────
    G_best, img_best, models_best, paths_best, _ = build_graph(
        df, qS, qF, correlation_method, regression_type,
        visualization_type="best",
        allowed_pairs=allowed_pairs,
        tmp_dir=tmp_dir,
    )

    # All scatter PNGs are now encoded in memory — delete _tmp/.
    shutil.rmtree(tmp_dir, ignore_errors=True)

    html_best    = str(run_dir / "graph_best.html")
    canvas_h_best = visualize(G_best, img_best, output_html=html_best, open_html=False)

    artifacts_best: Dict = {
        **artifacts_full,
        "G":                  G_best,
        "image_data_dict":    img_best,
        "models":             models_best,
        "paths":              paths_best,
        "visualization_type": "best",
    }
    save_artifacts(artifacts_best, str(run_dir / "artifacts_best.pkl"))
    print(
        f"[artifacts] {run_dir / 'artifacts.pkl'}  |  "
        f"best: {run_dir / 'artifacts_best.pkl'}"
    )

    # ── PNG capture ───────────────────────────────────────────────────────────
    if save_png:
        kw: Dict = {"pad_h": 140, "pad_v": 60, "device_scale": 2}
        if png_kwargs:
            kw.update(png_kwargs)
        time.sleep(0.2)
        capture_pyvis_html_as_png(
            html_full, out_png=str(run_dir / "graph.png"),
            canvas_height_px=canvas_h_full, **kw,
        )
        capture_pyvis_html_as_png(
            html_best, out_png=str(run_dir / "graph_best.png"),
            canvas_height_px=canvas_h_best, **kw,
        )
        print(f"[png] {run_dir / 'graph.png'}  |  {run_dir / 'graph_best.png'}")

    print(f"[done] {run_dir}")
    return artifacts_full
