"""
Common experimental protocol shared by every member of the group.

Everything that must be identical across experiments lives here so results
are directly comparable:
  * data loading (identifier dropped, malignant = positive class 1)
  * one fixed stratified train/test split (80/20, seed 42)
  * one 5-fold stratified CV splitter (seed 42)
  * one set of evaluation metrics (incl. specificity)
  * helpers for tuning, CV evaluation, test evaluation and saving results

The held-out test set is only touched in notebook 07 (final evaluation),
after every model has been selected and tuned using CV on the training set.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "wdbc.data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Model selection criterion used by every GridSearchCV in the project.
# F1 of the malignant class balances malignant recall (missed cancers) against
# precision (false alarms); pure recall would favour degenerate "always
# malignant" models. Malignant recall is still reported everywhere.
REFIT_METRIC = "f1"

_STATS = ["radius", "texture", "perimeter", "area", "smoothness",
          "compactness", "concavity", "concave_points", "symmetry",
          "fractal_dimension"]
FEATURE_NAMES = ([f"{s}_mean" for s in _STATS]
                 + [f"{s}_se" for s in _STATS]
                 + [f"{s}_worst" for s in _STATS])
CLASS_NAMES = ["benign", "malignant"]  # index = label


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Raw UCI file: id, diagnosis (M/B), 30 numeric features."""
    return pd.read_csv(DATA_PATH, header=None,
                       names=["id", "diagnosis"] + FEATURE_NAMES)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Features X (30 cols, id removed) and target y (1 = malignant)."""
    df = load_raw()
    X = df[FEATURE_NAMES].astype(float)
    y = (df["diagnosis"] == "M").astype(int).rename("malignant")
    return X, y


def get_split():
    """The ONE fixed stratified train/test split used by the whole project."""
    X, y = load_data()
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",                       # malignant sensitivity
    "specificity": make_scorer(specificity_score),
    "f1": "f1",
    "roc_auc": "roc_auc",
}
METRICS = list(SCORING)


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": (roc_auc_score(y_true, y_score)
                    if y_score is not None and len(np.unique(y_score)) > 1
                    else 0.5),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def cv_evaluate(model, X, y, name: str | None = None, **extra) -> dict:
    """5-fold stratified CV on the training set -> mean and std per metric."""
    res = cross_validate(model, X, y, cv=get_cv(), scoring=SCORING, n_jobs=-1)
    row = {"model": name} if name else {}
    row.update(extra)
    for m in METRICS:
        row[f"cv_{m}"] = res[f"test_{m}"].mean()
        row[f"cv_{m}_std"] = res[f"test_{m}"].std()
    return row


def tune(pipeline, param_grid, X, y, refit: str = REFIT_METRIC) -> GridSearchCV:
    """Grid search with the common CV splitter and metrics."""
    gs = GridSearchCV(pipeline, param_grid, cv=get_cv(), scoring=SCORING,
                      refit=refit, n_jobs=-1)
    gs.fit(X, y)
    return gs


def best_cv_row(gs: GridSearchCV, name: str, **extra) -> dict:
    """Mean/std of every metric for the best grid-search candidate."""
    i = gs.best_index_
    row = {"model": name, **extra}
    for m in METRICS:
        row[f"cv_{m}"] = gs.cv_results_[f"mean_test_{m}"][i]
        row[f"cv_{m}_std"] = gs.cv_results_[f"std_test_{m}"][i]
    return row


def test_evaluate(model, X_test, y_test, name: str, **extra) -> dict:
    """ONLY used in the final notebook, after all tuning is done."""
    y_pred = model.predict(X_test)
    row = {"model": name, **extra}
    row.update(compute_metrics(y_test, y_pred, _scores(model, X_test)))
    return row


def summarize(rows, cols=("accuracy", "precision", "recall", "specificity",
                          "f1", "roc_auc"), prefix="cv_") -> pd.DataFrame:
    """Readable table: 'mean ± std' strings for CV rows."""
    df = pd.DataFrame(rows)
    out = df[[c for c in df.columns if not c.startswith(prefix)]].copy()
    for c in cols:
        if f"{prefix}{c}_std" in df:
            out[c] = [f"{m:.4f} ± {s:.4f}" for m, s in
                      zip(df[f"{prefix}{c}"], df[f"{prefix}{c}_std"])]
        elif f"{prefix}{c}" in df:
            out[c] = df[f"{prefix}{c}"].round(4)
    return out


# --------------------------------------------------------------------------
# Persisting results (how members share work with each other)
# --------------------------------------------------------------------------
def save_rows(rows, filename: str) -> Path:
    path = RESULTS_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def save_best_params(model_key: str, params: dict) -> None:
    """Store tuned hyper-parameters so Members 5 & 6 and the final notebook
    reuse them instead of re-tuning (proposal section 8)."""
    path = RESULTS_DIR / "best_params.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    store[model_key] = {k: _jsonable(v) for k, v in params.items()}
    path.write_text(json.dumps(store, indent=2, sort_keys=True))


def load_best_params(model_key: str | None = None) -> dict:
    store = json.loads((RESULTS_DIR / "best_params.json").read_text())
    return store if model_key is None else store[model_key]


def savefig(fig, name: str):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
