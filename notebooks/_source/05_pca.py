# %% [markdown]
# # 05 — PCA-based Dimensionality Reduction
# **Member 5.** Transform the 30 standardised features into a smaller number of
# principal components and test whether classifiers keep their performance.
#
# * Settings: **2, 5, 10 components** and the number that keeps **≈95 % of the
#   training variance**, compared with the same model on all 30 original features.
# * Classifiers: Logistic Regression, KNN and SVM (as in the proposal), plus
#   Gaussian NB, which is interesting here because PCA components are uncorrelated,
#   matching NB's independence assumption.
# * Following the coordination plan, classifiers reuse the hyper-parameters tuned by
#   Members 2–3 (`results/best_params.json`). A re-tuning sanity check is included at the end.
# * `StandardScaler → PCA → classifier` lives in one `Pipeline`, so PCA is re-fitted
#   on each CV training fold only. The test set is not used here.

# %%
import sys
sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import *
from models import build_pipeline, display_name, GRIDS

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160, "display.max_columns", 40)
X_train, X_test, y_train, y_test = get_split()
del X_test, y_test
PARAMS = load_best_params()
CLASSIFIERS = ["logreg", "knn", "svm", "gnb"]

# %% [markdown]
# ## 1. Explained variance (training set)

# %%
Z = StandardScaler().fit_transform(X_train)
pca_full = PCA().fit(Z)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
n95 = int(np.searchsorted(cum, 0.95) + 1)
print(f"components needed for 95% variance: {n95}")
pd.DataFrame({"PC": range(1, 16), "explained_var": evr[:15], "cumulative": cum[:15]}).round(4)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(1, 31), evr, color="#7A6FD6", label="individual")
ax.step(range(1, 31), cum, where="mid", color="#E8684C", label="cumulative")
for t in [0.95]:
    ax.axhline(t, ls="--", color="grey", lw=1); ax.text(30.5, t, f"{t:.0%}", va="center")
for k in [2, 5, 10]:
    ax.annotate(f"{k} PCs: {cum[k-1]:.1%}", (k, cum[k-1]), xytext=(k + 1.2, cum[k-1] - 0.12),
                arrowprops=dict(arrowstyle="->", lw=.8), fontsize=9)
ax.set_xlabel("principal component"); ax.set_ylabel("explained variance ratio"); ax.legend(loc="center right")
ax.set_title("PCA scree plot (standardised training data)")
savefig(fig, "05_scree"); plt.show()

# %% [markdown]
# Variance is concentrated in the first few components: PC1 alone explains about 44 %
# and PC1–PC2 about 63 %. Ten components keep ≈95 %. This confirms how
# redundant the 30 features are.

# %% [markdown]
# ## 2. What do the components mean?

# %%
load = pd.DataFrame(pca_full.components_[:5].T, index=FEATURE_NAMES, columns=[f"PC{i}" for i in range(1, 6)])
fig, ax = plt.subplots(figsize=(6, 10))
sns.heatmap(load, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax, cbar_kws={"shrink": .5})
ax.set_title("Loadings of the first 5 principal components")
savefig(fig, "05_loadings"); plt.show()

# %%
for pc in ["PC1", "PC2"]:
    s = load[pc].abs().sort_values(ascending=False)
    print(pc, "top loadings:", ", ".join(f"{f} ({load.loc[f, pc]:+.2f})" for f in s.index[:6]))

# %% [markdown]
# * **PC1** loads positively on almost every feature, most strongly concavity, concave
#   points, compactness and size. It is an overall "large and irregular nucleus" axis.
# * **PC2** contrasts fractal dimension (all three versions) and `compactness_se`
#   (positive) with size (`radius_mean`, `area_mean` negative). Roughly, it captures "small but
#   with a complex, finely irregular boundary".
#
# Each component mixes all 30 original measurements, so a model built on PCs is much harder
# to explain in clinical terms than one built on selected features (notebook 06).

# %%
pcs = PCA(n_components=2).fit_transform(Z)
fig, ax = plt.subplots(figsize=(7, 5))
for lab, col in [(0, "#4C9BE8"), (1, "#E8684C")]:
    m = y_train.values == lab
    ax.scatter(pcs[m, 0], pcs[m, 1], s=14, alpha=.65, c=col, label=CLASS_NAMES[lab])
ax.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr[1]:.1%})"); ax.legend()
ax.set_title("Training data projected on the first two components")
savefig(fig, "05_pc1_pc2_scatter"); plt.show()

# %% [markdown]
# Even in 2-D the classes are almost linearly separable along PC1. PCA is
# *unsupervised*: it was not told the labels, yet the direction of maximum variance
# happens to be the direction that separates benign from malignant.

# %% [markdown]
# ## 3. Main experiment: 2 / 5 / 10 / 95 % components vs all 30 features

# %%
settings = [("all 30 (no PCA)", None), ("PCA 2", 2), ("PCA 5", 5), ("PCA 10", 10), ("PCA 95% var", 0.95)]
rows = []
for key in CLASSIFIERS:
    for label, n in settings:
        reducer = None if n is None else PCA(n_components=n, random_state=RANDOM_STATE)
        pipe = build_pipeline(key, PARAMS[key], reducer=reducer)
        n_feat = 30 if n is None else (n95 if n == 0.95 else n)
        rows.append(cv_evaluate(pipe, X_train, y_train, name=display_name(key), key=key,
                                method="none" if n is None else "PCA", setting=label,
                                n_components=str(n), n_features=n_feat))
pca_df = pd.DataFrame(rows)
save_rows(rows, "cv_05_pca.csv")
summarize(rows)[["model", "setting", "n_features", "accuracy", "recall", "specificity", "f1", "roc_auc"]]

# %%
order = [s for s, _ in settings]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), sharey=False)
for ax, m in zip(axes, ["accuracy", "recall", "roc_auc"]):
    piv = pca_df.pivot(index="setting", columns="model", values=f"cv_{m}").loc[order]
    piv.plot(kind="bar", ax=ax, width=.8, rot=20, legend=ax is axes[0])
    lo = max(0.8, piv.values.min() - 0.02); ax.set_ylim(lo, 1.0)
    ax.set_title(f"CV {m}"); ax.set_xlabel("")
fig.suptitle("PCA vs original features (reused hyper-parameters)", y=1.02); fig.tight_layout()
savefig(fig, "05_pca_settings"); plt.show()

# %% [markdown]
# **Findings (reused hyper-parameters).**
# * The 95 % variance rule picks exactly **10 components**, so "PCA 10" and "PCA 95 % var"
#   are the same model here.
# * **2 components are not enough.** Every model loses 2–7 F1 points, and KNN and GNB lose the
#   most recall.
# * **5 components recover most of the performance.** LR and SVM are within ≈1.5 F1 points
#   of their 30-feature versions.
# * **10 components match or slightly beat 30 features** for LR (F1 0.968 vs 0.967) and
#   SVM (0.976 vs 0.976, with *higher* malignant recall 0.971 vs 0.965 and a much smaller
#   fold-to-fold std). KNN is about 2 points worse at 10 PCs with its old *k*.
# * **Gaussian NB does *not* benefit** from decorrelated inputs; it gets worse. Making the
#   features uncorrelated does not make them *independent given the class*, and the
#   low-variance components it keeps are close to noise. Its weakness on this data is the
#   class-conditional Gaussian model itself.

# %% [markdown]
# ## 4. Performance as a function of the number of components

# %%
curve = []
for key in CLASSIFIERS:
    for n in range(1, 31):
        pipe = build_pipeline(key, PARAMS[key], reducer=PCA(n_components=n, random_state=RANDOM_STATE))
        curve.append(cv_evaluate(pipe, X_train, y_train, name=display_name(key), key=key, n_features=n))
curve = pd.DataFrame(curve)
curve.to_csv(RESULTS_DIR / "cv_05_pca_curve.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
for ax, m in zip(axes, ["f1", "recall"]):
    for key, g in curve.groupby("key", sort=False):
        ax.plot(g["n_features"], g[f"cv_{m}"], "o-", ms=3, label=display_name(key))
    ax.axvline(n95, ls=":", color="grey"); ax.text(n95 + .3, 0.86, "95% var", color="grey")
    ax.set_ylim(0.85, 1.0); ax.set_xlabel("number of principal components"); ax.set_ylabel(f"CV {m}")
    ax.set_title(f"CV {m} vs number of components")
axes[0].legend(); fig.tight_layout(); savefig(fig, "05_pca_curve"); plt.show()

# %%
best_n = curve.loc[curve.groupby("key")["cv_f1"].idxmax(), ["model", "n_features", "cv_f1", "cv_recall", "cv_accuracy"]]
full = pca_df[pca_df.method == "none"].set_index("model")[["cv_f1", "cv_recall", "cv_accuracy"]].add_suffix("_all30")
best_n.set_index("model").join(full).round(4)

# %%
# smallest number of components within 1 F1-point (0.01) of the same model on all 30 features
small = []
for key, g in curve.groupby("key", sort=False):
    ref = pca_df[(pca_df.key == key) & (pca_df.method == "none")]["cv_f1"].iloc[0]
    ok = g[g["cv_f1"] >= ref - 0.01]
    small.append({"model": display_name(key), "all30_f1": ref,
                  "min_components_within_0.01": int(ok["n_features"].min()) if len(ok) else None})
pd.DataFrame(small).round(4)

# %% [markdown]
# Logistic Regression needs only **5 components** to get within 0.01 F1 of its full
# 30-feature performance, SVM 8 and KNN 9. Gaussian NB never gets there. Beyond ≈10–11
# components, the extra low-variance directions add nothing (LR, SVM) or slightly hurt
# (KNN, through the curse of dimensionality).

# %% [markdown]
# ## 5. Sanity check: re-tuning the classifiers on PCA features
# Reusing hyper-parameters tuned on 30 features could handicap PCA (e.g. a KNN
# neighbourhood size or RBF `gamma` chosen for 30-D may not suit 5-D). Here each
# classifier is re-tuned, with the same grids as Members 2–3, inside each PCA setting.

# %%
retuned = []
for key in CLASSIFIERS:
    for label, n in settings[1:]:
        gs = tune(build_pipeline(key, reducer=PCA(n_components=n, random_state=RANDOM_STATE)), GRIDS[key],
                  X_train, y_train)
        row = best_cv_row(gs, display_name(key), key=key, setting=label)
        row["reused_f1"] = pca_df[(pca_df.key == key) & (pca_df.setting == label)]["cv_f1"].iloc[0]
        retuned.append(row)
rt = pd.DataFrame(retuned)
rt["gain_from_retuning"] = rt["cv_f1"] - rt["reused_f1"]
rt.to_csv(RESULTS_DIR / "cv_05_pca_retuned.csv", index=False)
rt[["model", "setting", "reused_f1", "cv_f1", "gain_from_retuning", "cv_recall"]].round(4)

# %% [markdown]
# Re-tuning changes little for LR, SVM and GNB (≤ 0.01 F1). **KNN is the exception**:
# with a re-tuned *k* it gains 1–3 F1 points, and at 5 components it matches its
# 30-feature score (0.967). So the reused-parameter results slightly understate PCA for
# KNN. The overall conclusions do not change.
#
# ## Summary
# * 10 PCs keep 95 % of the variance. 5 PCs keep 85 %, and 2 PCs keep 63 %.
# * With **10 components (one third of the dimensions)**, LR and SVM perform as well as with
#   all 30 features, and SVM's malignant recall even improves slightly. 5 components cost
#   about 1 F1 point. 2 components cost 3–7 points.
# * PCA does **not** clearly *improve* performance on this dataset. Its value is compression:
#   the same accuracy from far fewer dimensions.
# * The price is **interpretability**: every component is a mixture of all 30 measurements,
#   and all 30 must still be *measured* to compute the components.

