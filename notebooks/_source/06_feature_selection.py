# %% [markdown]
# # 06 — Feature Selection
# **Member 6.** Keep a *subset of the original measurements*, rather than transformed
# components, and test whether classifiers keep their performance.
#
# Three selectors, each asked for **k = 5, 10, 15** features (all 30 as reference):
#
# | Method | Type | How it ranks features |
# |---|---|---|
# | `SelectKBest(mutual_info_classif)` | filter | univariate mutual information with the label |
# | `RFE(LogisticRegression)` | wrapper | repeatedly fits an L2 logistic regression and drops the feature with the smallest \|coefficient\| |
# | L1 path (`L1PathSelector`, `src/feature_selectors.py`) | embedded | the first *k* features to get a non-zero weight as L1 regularisation is relaxed |
#
# The classifiers (LR, KNN, SVM, Gaussian NB) are the same as in notebook 05 for a direct
# PCA-vs-selection comparison, and they reuse the hyper-parameters tuned by Members 2–3.
# Every selector sits inside the `Pipeline`, so it is re-fitted on each CV training fold
# and selection never sees the validation fold. The test set is not used here.

# %%
import sys
sys.path.insert(0, "../src")
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *
from models import build_pipeline, display_name
from feature_selectors import L1PathSelector, make_selector, METHODS, METHOD_NAMES

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160, "display.max_columns", 40)
X_train, X_test, y_train, y_test = get_split()
del X_test, y_test
PARAMS = load_best_params()
CLASSIFIERS = ["logreg", "knn", "svm", "gnb"]



# %% [markdown]
# ## 1. Which features does each method pick? (full training set)

# %%
Z = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=FEATURE_NAMES, index=X_train.index)
rankings = {}
mi = pd.Series(mutual_info_classif(Z, y_train, random_state=RANDOM_STATE), index=FEATURE_NAMES)
rankings["MI"] = mi.rank(ascending=False, method="first")
rfe_full = RFE(LogisticRegression(solver="liblinear", max_iter=5000, random_state=RANDOM_STATE),
               n_features_to_select=1).fit(Z, y_train)
rankings["RFE"] = pd.Series(rfe_full.ranking_, index=FEATURE_NAMES).astype(float)
# L1 path: order in which features enter the model as C grows
enter = {}
for C in np.logspace(-3, 3, 121):
    coef = LogisticRegression(l1_ratio=1, solver="liblinear", C=C, max_iter=5000).fit(Z, y_train).coef_[0]
    for f, c in zip(FEATURE_NAMES, coef):
        if c != 0 and f not in enter:
            enter[f] = C
rank_l1 = pd.Series({f: enter.get(f, np.inf) for f in FEATURE_NAMES}).rank(method="first")
rankings["L1"] = rank_l1
rank_df = pd.DataFrame(rankings).rename(columns=METHOD_NAMES).astype(int)
rank_df["mean_rank"] = rank_df.mean(axis=1)
rank_df = rank_df.sort_values("mean_rank")
rank_df.to_csv(RESULTS_DIR / "feature_rankings.csv")
rank_df

# %%
fig, ax = plt.subplots(figsize=(6, 10))
sns.heatmap(rank_df.drop(columns="mean_rank"), annot=True, fmt="d", cmap="viridis_r", cbar_kws={"label": "rank (1 = best)"}, ax=ax)
ax.set_title("Feature rank by each selection method\n(full training set, sorted by mean rank)")
savefig(fig, "06_feature_rank_heatmap"); plt.show()

# %%
for k in [5, 10, 15]:
    sets = {m: set(rank_df.index[rank_df[METHOD_NAMES[m]] <= k]) for m in METHODS}
    common_all = set.intersection(*sets.values())
    print(f"k={k:2d}  selected by all three: {len(common_all)}  -> {sorted(common_all)}")
    for m in METHODS:
        print(f"        {METHOD_NAMES[m]:17s}: {sorted(sets[m])}")

# %% [markdown]
# **The methods disagree in an informative way.**
# * **Mutual information** is univariate: it ranks every feature on its own, so its top
#   10 is crowded with near-duplicates (radius / perimeter / area in both `_mean` and
#   `_worst` form, plus concave points). Each one is individually strong, but together they
#   carry largely the same information.
# * **RFE** and the **L1 path** are multivariate: once one size feature is in, its
#   duplicates add little, so they prefer *complementary* features such as texture,
#   smoothness, symmetry and `_se` measurements.
# * At k = 10 all three methods agree on four features: `radius_worst`,
#   `perimeter_worst`, `concave_points_worst` and `concave_points_mean`. These are the
#   size and concavity of the most abnormal nuclei. RFE and the L1 path also both add
#   `texture_worst` and `radius_se`.

# %% [markdown]
# ## 2. Main experiment: k = 5, 10, 15 vs all 30 features

# %%
rows = []
for key in CLASSIFIERS:
    rows.append(cv_evaluate(build_pipeline(key, PARAMS[key]), X_train, y_train, name=display_name(key), key=key,
                            method="none", setting="all 30", n_features=30))
    for method in METHODS:
        for k in [5, 10, 15]:
            pipe = build_pipeline(key, PARAMS[key], reducer=make_selector(method, k))
            rows.append(cv_evaluate(pipe, X_train, y_train, name=display_name(key), key=key,
                                    method=method, setting=f"{METHOD_NAMES[method]} k={k}", n_features=k))
fs_df = pd.DataFrame(rows)
save_rows(rows, "cv_06_feature_selection.csv")
summarize(rows)[["model", "setting", "accuracy", "recall", "specificity", "f1", "roc_auc"]]

# %%
fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), sharey=True)
for ax, key in zip(axes, CLASSIFIERS):
    g = fs_df[fs_df.key == key]
    full = g[g.method == "none"]["cv_f1"].iloc[0]
    for method in METHODS:
        h = g[g.method == method]
        ax.errorbar(h["n_features"], h["cv_f1"], yerr=h["cv_f1_std"], marker="o", capsize=3, label=METHOD_NAMES[method])
    ax.axhline(full, color="k", ls="--", lw=1, label="all 30 features")
    ax.set_xticks([5, 10, 15]); ax.set_xlabel("k selected features"); ax.set_title(display_name(key))
axes[0].set_ylabel("CV F1"); axes[0].legend(fontsize=8); axes[0].set_ylim(0.85, 1.0)
fig.suptitle("Feature selection vs all 30 features (reused hyper-parameters)", y=1.02); fig.tight_layout()
savefig(fig, "06_fs_main"); plt.show()

# %% [markdown]
# **Findings (reused hyper-parameters).**
# * **The method matters more than k.** The multivariate selectors (RFE, L1 path) are far
#   better than the univariate MI filter for every classifier. Even with 15 features, MI
#   is 4–6 F1 points below the full feature set for LR, KNN and SVM, because its picks are
#   largely redundant copies of tumour size.
# * **The L1 path with only 5 features** (`radius_worst`, `perimeter_worst`,
#   `concave_points_worst`, `concave_points_mean`, `texture_worst`) gives LR an F1 of 0.965,
#   within 0.003 of the 30-feature LR (0.967), with malignant recall 0.965 vs 0.959.
#   Five original measurements are enough for a linear model.
# * **SVM loses 1.5–2 F1 points** with 5–15 selected features (0.955–0.961 vs 0.976). KNN loses
#   1–2.5 points.
# * **Gaussian NB improves** with selection (L1 path k=5: F1 0.940 vs 0.921 on all 30).
#   Removing redundant features *does* help NB once whole redundant groups go, not just
#   perimeter and area. This revises the result of notebook 02.

# %% [markdown]
# ## 3. Performance as a function of k

# %%
curve = []
for key in CLASSIFIERS:
    for method in METHODS:
        for k in range(1, 31):
            pipe = build_pipeline(key, PARAMS[key], reducer=make_selector(method, k))
            curve.append(cv_evaluate(pipe, X_train, y_train, name=display_name(key), key=key, method=method, n_features=k))
curve = pd.DataFrame(curve)
curve.to_csv(RESULTS_DIR / "cv_06_fs_curve.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), sharey=True)
for ax, key in zip(axes, CLASSIFIERS):
    for method in METHODS:
        h = curve[(curve.key == key) & (curve.method == method)]
        ax.plot(h["n_features"], h["cv_f1"], "o-", ms=3, label=METHOD_NAMES[method])
    ax.axhline(fs_df[(fs_df.key == key) & (fs_df.method == "none")]["cv_f1"].iloc[0], color="k", ls="--", lw=1)
    ax.set_title(display_name(key)); ax.set_xlabel("k selected features")
axes[0].set_ylabel("CV F1"); axes[0].set_ylim(0.8, 1.0); axes[0].legend(fontsize=8)
fig.suptitle("CV F1 vs number of selected features (dashed = all 30)", y=1.02); fig.tight_layout()
savefig(fig, "06_fs_curve"); plt.show()

# %%
small = []
for (key, method), g in curve.groupby(["key", "method"], sort=False):
    ref = fs_df[(fs_df.key == key) & (fs_df.method == "none")]["cv_f1"].iloc[0]
    ok = g[g["cv_f1"] >= ref - 0.01]
    best = g.loc[g["cv_f1"].idxmax()]
    small.append({"model": display_name(key), "method": METHOD_NAMES[method], "all30_f1": ref,
                  "best_f1": best["cv_f1"], "best_k": int(best["n_features"]),
                  "min_k_within_0.01": int(ok["n_features"].min()) if len(ok) else None})
small = pd.DataFrame(small)
small.round(4)

# %% [markdown]
# With RFE or the L1 path, LR needs only 5–9 features to get within 0.01 F1 of all 30,
# and GNB 2–3 (it gets *better*). SVM needs 18–19 and KNN 11–14. The non-linear models
# exploit the extra, weaker features more. MI needs 20+ features for every model except GNB.

# %% [markdown]
# ## 4. Selection stability: are the same features chosen on every fold?
# Each selector (k = 10) is fitted on each of the 5 CV training folds. A feature chosen 5/5
# times is a stable choice; one that flips in and out depends on the particular sample.

# %%
freq = pd.DataFrame(0, index=FEATURE_NAMES, columns=[METHOD_NAMES[m] for m in METHODS])
for tr, _ in get_cv().split(X_train, y_train):
    sc = StandardScaler().fit(X_train.iloc[tr])
    Zt = sc.transform(X_train.iloc[tr])
    for m in METHODS:
        sel = make_selector(m, 10).fit(Zt, y_train.iloc[tr])
        freq.loc[np.array(FEATURE_NAMES)[sel.get_support()], METHOD_NAMES[m]] += 1
freq["total"] = freq.sum(axis=1)
freq = freq[freq.total > 0].sort_values("total", ascending=False)
freq.to_csv(RESULTS_DIR / "fs_selection_frequency.csv")

fig, ax = plt.subplots(figsize=(6, 0.32 * len(freq) + 1))
sns.heatmap(freq.drop(columns="total"), annot=True, fmt="d", cmap="Greens", vmin=0, vmax=5, cbar=False, ax=ax)
ax.set_title("Times selected across 5 CV folds (k = 10)")
savefig(fig, "06_selection_stability"); plt.show()

# %%
consistent = freq[(freq.drop(columns="total") == 5).all(axis=1)].index.tolist()
print("Selected in 5/5 folds by ALL three methods:", consistent)
stable_per_method = {c: int((freq[c] == 5).sum()) for c in freq.columns[:-1]}
print("Features selected in every fold, per method:", stable_per_method)

# %% [markdown]
# * **MI is perfectly stable** (the same 10 features in every fold) but picks redundant features.
#   Stability is not the same as usefulness.
# * RFE and the L1 path are less stable, because among near-duplicates (e.g. `perimeter_worst`
#   vs `radius_worst` vs `area_worst`) which one wins can flip between folds. Any of them carries
#   the same information.
# * **The consistently important measurements** across methods and folds are `radius_worst`,
#   `concave_points_mean`, `concave_points_worst`, `area_worst` and `perimeter_worst`, followed by
#   `texture_worst` and `radius_se`. This matches the tree importances of notebook 04.

# %% [markdown]
# ## 5. L1 regularisation for classification vs L1-based feature selection
# Coordination with Member 2:
# * **(a) L1 as the classifier.** Member 2's best L1-penalised logistic regression learns
#   sparse weights and classifies with them in one step.
# * **(b) L1 as a selector.** The L1 path only chooses the features. A separately tuned
#   classifier (L2 LR or RBF SVM) is then trained on them, with the same number of
#   features as (a) uses.

# %%
l1p = PARAMS["logreg_l1"]
l1_clf = build_pipeline("logreg", l1p)
nz = int((l1_clf.fit(X_train, y_train).named_steps["clf"].coef_ != 0).sum())
print(f"L1 logistic regression (C={l1p['clf__C']:.3g}) uses {nz} non-zero features on the full training set")
cmp_rows = [cv_evaluate(l1_clf, X_train, y_train, name=f"(a) L1-penalised LR classifier (~{nz} features)")]
for key in ["logreg", "svm"]:
    pipe = build_pipeline(key, PARAMS[key], reducer=L1PathSelector(k=nz))
    cmp_rows.append(cv_evaluate(pipe, X_train, y_train, name=f"(b) L1-path select {nz} → {display_name(key)}"))
cmp_rows.append(cv_evaluate(build_pipeline("logreg", PARAMS["logreg"]), X_train, y_train, name="L2 LR, all 30 (reference)"))
save_rows(cmp_rows, "cv_06_l1_comparison.csv")
summarize(cmp_rows)

# %% [markdown]
# L1 used as the classifier and L1 used as a selector perform almost the same here
# (F1 0.964 vs 0.961), and both are within ≈0.006 of the dense L2 model on all 30 features.
# The one-step L1 classifier is the simpler choice. Using L1 only to pick features is useful
# when a *different* downstream model is wanted, but with ~9 features the RBF SVM gains
# nothing over the linear model.

# %% [markdown]
# ## 6. Sanity check: re-tuning the classifiers on the selected features
# As in notebook 05, hyper-parameters tuned on 30 features may not suit 5–15 features.
# RFE and L1-path settings are re-tuned with the members' grids. MI is excluded because
# re-fitting MI inside every grid candidate is expensive, and MI is clearly the weakest
# selector anyway.

# %%
from models import GRIDS
retuned = []
for key in CLASSIFIERS:
    for method in ["RFE", "L1"]:
        for k in [5, 10, 15]:
            gs = tune(build_pipeline(key, reducer=make_selector(method, k)), GRIDS[key], X_train, y_train)
            row = best_cv_row(gs, display_name(key), key=key, method=method, n_features=k)
            row["reused_f1"] = fs_df[(fs_df.key == key) & (fs_df.method == method) & (fs_df.n_features == k)]["cv_f1"].iloc[0]
            row["best_params"] = {p.replace("clf__", ""): v for p, v in gs.best_params_.items()}
            retuned.append(row)
rt = pd.DataFrame(retuned)
rt["gain_from_retuning"] = rt["cv_f1"] - rt["reused_f1"]
rt.to_csv(RESULTS_DIR / "cv_06_fs_retuned.csv", index=False)
rt[["model", "method", "n_features", "reused_f1", "cv_f1", "gain_from_retuning", "cv_recall", "best_params"]].round(4)

# %% [markdown]
# **Re-tuning matters more here than for PCA.** It recovers 1–1.5 F1 points for several
# configurations. With 10 L1-path features, LR reaches CV F1 0.970 and a (now linear) SVM
# 0.971, essentially equal to the 30-feature models (0.967 / 0.976). Part of the gap seen
# with reused parameters therefore came from hyper-parameters tuned for 30 dimensions (e.g.
# the RBF `gamma`), not from lost information. The ranking of methods is unchanged:
# L1 path ≥ RFE ≫ MI.
#
# ## Summary
# * **Univariate filtering (MI) is the wrong tool** for strongly correlated features. It
#   selects many copies of "tumour size" and never matches the full feature set at k ≤ 15.
# * **Multivariate selection works.** The L1 path with 5 features is enough for logistic
#   regression (CV F1 0.965 vs 0.967), and with re-tuning, 10 selected features reach ≈0.97
#   for LR and SVM.
# * **Consistently important measurements:** `radius_worst`, `perimeter_worst`,
#   `area_worst`, `concave_points_worst`, `concave_points_mean`, then `texture_worst` and
#   `radius_se`.
# * Unlike PCA, the result is a short list of *named, measurable* features, so a model on
#   5–10 of them is both cheaper to apply and easier to explain.

