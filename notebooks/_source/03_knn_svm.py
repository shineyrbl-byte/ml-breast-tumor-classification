# %% [markdown]
# # 03 — K-Nearest Neighbours and Support Vector Machine
# **Member 3.** Implement and tune KNN and SVM on all 30 standardised features using
# the common protocol. The test set is not used here.

# %%
import sys
sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
from common import *
from models import build_pipeline, display_name, GRIDS

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160, "display.max_columns", 40)
X_train, X_test, y_train, y_test = get_split()
del X_test, y_test

# %% [markdown]
# ## 1. KNN
# Grid: number of neighbours *k* (odd values 1–31, to avoid ties), uniform vs
# distance weighting, and Euclidean vs Manhattan distance.
# Standardisation is essential for KNN — without it `area_*` (values in the
# thousands) would dominate every distance. The cell below shows this directly.

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
summarize([
    cv_evaluate(Pipeline([("clf", KNeighborsClassifier())]), X_train, y_train, name="KNN k=5, NOT scaled"),
    cv_evaluate(build_pipeline("knn"), X_train, y_train, name="KNN k=5, standardised"),
])

# %%
knn_grid = GRIDS["knn"]
print(knn_grid)
knn_gs = tune(build_pipeline("knn"), knn_grid, X_train, y_train)
print("best params:", knn_gs.best_params_)
print(f"best CV F1: {knn_gs.best_score_:.4f}")

# %%
r = pd.DataFrame(knn_gs.cv_results_)
r["config"] = r["param_clf__weights"] + ", " + r["param_clf__p"].map({1: "manhattan", 2: "euclidean"})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
for ax, metric in zip(axes, ["f1", "recall"]):
    for cfg, g in r.groupby("config"):
        ax.plot(g["param_clf__n_neighbors"], g[f"mean_test_{metric}"], "o-", ms=4, label=cfg)
    ax.set_xlabel("k (neighbours)"); ax.set_ylabel(f"CV {metric}"); ax.set_title(f"KNN — CV {metric} vs k")
axes[0].legend(); fig.tight_layout(); savefig(fig, "03_knn_k_curve"); plt.show()

# %% [markdown]
# The best *k* is small (k = 3 with Manhattan distance). k = 1 overfits (it memorises the
# training fold), and from k ≈ 5 upwards performance declines steadily. The drop shows
# up mostly in **recall**: as the neighbourhood grows, the minority (malignant) class gets
# outvoted near the boundary.

# %% [markdown]
# ## 2. SVM
# Grid: linear and RBF kernels, `C` (margin softness), `gamma` (RBF width) and
# `class_weight`.

# %%
svm_grid = GRIDS["svm"]
svm_gs = tune(build_pipeline("svm"), svm_grid, X_train, y_train)
print("best params:", svm_gs.best_params_)
print(f"best CV F1: {svm_gs.best_score_:.4f}")

# %%
r = pd.DataFrame(svm_gs.cv_results_)
rbf = r[(r["param_clf__kernel"] == "rbf")]
cw_best = svm_gs.best_params_["clf__class_weight"]
rbf = rbf[rbf["param_clf__class_weight"].astype(str) == str(cw_best)]
heat = rbf.pivot_table(index="param_clf__C", columns="param_clf__gamma", values="mean_test_f1")
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis", ax=ax,
            xticklabels=[f"{g:.0e}" for g in heat.columns], yticklabels=[f"{c:g}" for c in heat.index])
ax.set_xlabel("gamma"); ax.set_ylabel("C")
ax.set_title(f"RBF SVM — CV F1 (class_weight={cw_best})")
savefig(fig, "03_svm_rbf_heatmap"); plt.show()

# %%
lin = r[r["param_clf__kernel"] == "linear"].copy()
lin["cw"] = lin["param_clf__class_weight"].fillna("none")
fig, ax = plt.subplots(figsize=(7, 3.8))
for cw, g in lin.groupby("cw"):
    ax.errorbar(g["param_clf__C"].astype(float), g["mean_test_f1"], yerr=g["std_test_f1"], marker="o", ms=4,
                capsize=2, label=f"class_weight={cw}")
ax.set_xscale("log"); ax.set_xlabel("C"); ax.set_ylabel("CV F1"); ax.set_ylim(0.85, 1); ax.legend()
ax.set_title("Linear SVM — CV F1 vs C")
savefig(fig, "03_svm_linear_C"); plt.show()

# %% [markdown]
# ### Best linear vs best RBF kernel

# %%
rows = []
for kern in ["linear", "rbf"]:
    sub = r[r["param_clf__kernel"] == kern]
    i = sub["mean_test_f1"].idxmax()
    row = {"model": f"SVM ({kern})", "params": {k.replace("clf__", ""): v for k, v in r.loc[i, "params"].items()}}
    for m in METRICS:
        row[f"cv_{m}"] = r.loc[i, f"mean_test_{m}"]; row[f"cv_{m}_std"] = r.loc[i, f"std_test_{m}"]
    rows.append(row)
summarize(rows)

# %% [markdown]
# ## 3. Comparison

# %%
rows = [best_cv_row(knn_gs, display_name("knn"), key="knn", feature_set="all 30"),
        best_cv_row(svm_gs, display_name("svm"), key="svm", feature_set="all 30")]
save_rows(rows, "cv_03_knn_svm.csv")
save_best_params("knn", knn_gs.best_params_)
save_best_params("svm", svm_gs.best_params_)
summarize(rows)

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
oof = {}
for ax, (key, gs) in zip(axes, [("knn", knn_gs), ("svm", svm_gs)]):
    oof[key] = cross_val_predict(gs.best_estimator_, X_train, y_train, cv=get_cv())
    ConfusionMatrixDisplay.from_predictions(y_train, oof[key], display_labels=CLASS_NAMES, cmap="Blues",
                                            colorbar=False, ax=ax)
    ax.set_title(f"{display_name(key)} — out-of-fold")
fig.tight_layout(); savefig(fig, "03_confusion"); plt.show()

# %%
err = pd.DataFrame({"true": y_train, "knn": oof["knn"], "svm": oof["svm"]})
print("errors — KNN:", int((err.true != err.knn).sum()), " SVM:", int((err.true != err.svm).sum()),
      " both:", int(((err.true != err.knn) & (err.true != err.svm)).sum()))
print("KNN false negatives (missed malignant):", int(((err.true == 1) & (err.knn == 0)).sum()))
print("SVM false negatives (missed malignant):", int(((err.true == 1) & (err.svm == 0)).sum()))
