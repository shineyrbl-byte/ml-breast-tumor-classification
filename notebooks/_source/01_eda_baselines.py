# %% [markdown]
# # 01 — Data Preprocessing, EDA and Baseline Models
# **Member 1.** Data quality checks, exploratory analysis, the common experimental
# protocol, and two reference models: a majority-class baseline and Linear
# Discriminant Analysis (LDA).
#
# The protocol itself is implemented in `src/common.py` and imported by every notebook:
#
# | Item | Choice |
# |---|---|
# | Positive class | malignant = 1 (benign = 0) |
# | Train / test split | stratified 80 / 20, `random_state=42` (455 / 114 samples) |
# | Model development | 5-fold stratified CV on the training set only, `random_state=42` |
# | Metrics | accuracy, precision, malignant recall (sensitivity), specificity, F1, ROC-AUC, confusion matrix |
# | Tuning criterion | malignant-class F1 (balances missed cancers against false alarms) |
# | Test set | used once, in notebook 07, after all tuning |
# | Leakage control | scaling / PCA / feature selection live inside sklearn `Pipeline`s |

# %%
import sys
sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common import *
from models import build_pipeline, display_name

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160, "display.max_columns", 40)

# %% [markdown]
# ## 1. Data quality

# %%
raw = load_raw()
print("shape:", raw.shape)
raw.head()

# %%
print("missing values per column (total):", int(raw.isna().sum().sum()))
print("duplicate rows:", int(raw.duplicated().sum()))
print("duplicate rows ignoring id:", int(raw.drop(columns="id").duplicated().sum()))
print("duplicate ids:", int(raw["id"].duplicated().sum()))
print("diagnosis values:", raw["diagnosis"].unique())
print("negative feature values:", int((raw[FEATURE_NAMES] < 0).sum().sum()))
print("dtypes:", raw[FEATURE_NAMES].dtypes.unique())

# %% [markdown]
# The data are clean: no missing values, no duplicate records, only `M`/`B` labels and
# no negative measurements. A handful of features contain exact zeros
# (`concavity_*`, `concave_points_*`) — these are legitimate (nuclei with no concave
# segments), not missing values, so they are kept.

# %%
zeros = (raw[FEATURE_NAMES] == 0).sum()
zeros[zeros > 0]

# %% [markdown]
# The `id` column is an identifier with no predictive meaning and is dropped
# (`load_data()` returns only the 30 numeric features).

# %%
X, y = load_data()
X_train, X_test, y_train, y_test = get_split()
print("X:", X.shape, " train:", X_train.shape, " test:", X_test.shape)

# %% [markdown]
# ## 2. Class balance

# %%
bal = pd.DataFrame({
    "full": y.value_counts(),
    "train": y_train.value_counts(),
    "test": y_test.value_counts(),
}).rename(index={0: "benign", 1: "malignant"})
bal.loc["% malignant"] = (bal.loc["malignant"] / bal.sum() * 100).round(1)
bal

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
counts = y.map({0: "benign", 1: "malignant"}).value_counts()
ax.bar(counts.index, counts.values, color=["#4C9BE8", "#E8684C"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, f"{v} ({v / len(y):.0%})", ha="center")
ax.set_ylabel("samples"); ax.set_title("Class distribution")
savefig(fig, "01_class_distribution"); plt.show()

# %% [markdown]
# The classes are moderately imbalanced (≈63 % benign, 37 % malignant). That is why
# accuracy alone is not enough and why a majority-class model already scores ≈63 %.
# The stratified split keeps the same ratio in train and test.
#
# **From here on, all exploratory analysis uses the training set only**, so nothing we
# learn about the data comes from the test set.

# %% [markdown]
# ## 3. Feature distributions

# %%
train = X_train.assign(diagnosis=y_train.map({0: "benign", 1: "malignant"}))
X_train.describe().T.round(3)

# %% [markdown]
# Feature scales differ by several orders of magnitude (e.g. `area_worst` up to
# ~4000 vs `fractal_dimension_se` ≈ 0.003), so standardisation is required for
# distance- and margin-based models (KNN, SVM), regularised models (logistic
# regression), and PCA.

# %%
mean_feats = [f for f in FEATURE_NAMES if f.endswith("_mean")]
fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
for ax, f in zip(axes.flat, mean_feats):
    sns.kdeplot(data=train, x=f, hue="diagnosis", fill=True, common_norm=False,
                palette={"benign": "#4C9BE8", "malignant": "#E8684C"}, ax=ax, legend=ax is axes.flat[0])
    ax.set_title(f); ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("Distribution of the 10 *_mean features by class (training set)", y=1.01)
fig.tight_layout(); savefig(fig, "01_mean_feature_kde"); plt.show()

# %%
from scipy.stats import skew
sk = X_train.apply(skew).sort_values(ascending=False)
print("Most right-skewed features:\n", sk.head(8).round(2))

# %%
z = (X_train - X_train.mean()) / X_train.std()
long = z.assign(diagnosis=train["diagnosis"]).melt(id_vars="diagnosis", var_name="feature", value_name="z")
fig, ax = plt.subplots(figsize=(18, 5))
sns.boxplot(data=long, x="feature", y="z", hue="diagnosis", fliersize=1.5,
            palette={"benign": "#4C9BE8", "malignant": "#E8684C"}, ax=ax)
ax.tick_params(axis="x", rotation=90); ax.set_ylabel("standardised value"); ax.set_xlabel("")
ax.set_title("All 30 features (standardised) by class — training set")
savefig(fig, "01_boxplots"); plt.show()

# %% [markdown]
# **Observations.** Size-related features (radius, perimeter, area) and shape
# irregularity (concavity, concave points, compactness) separate the classes strongly —
# malignant nuclei are larger and more irregular. Texture separates moderately.
# Smoothness, symmetry and fractal dimension (especially their `_se` versions) overlap
# heavily and look weakly informative on their own. Many features (area, `_se`
# features) are right-skewed with outliers, mostly on the malignant side.

# %% [markdown]
# ### Univariate separability
# Point-biserial correlation with the target and a univariate ROC-AUC for each feature
# (computed on the training set).

# %%
from sklearn.metrics import roc_auc_score
uni = pd.DataFrame({
    "corr_with_malignant": X_train.corrwith(y_train),
    "univariate_auc": [max(roc_auc_score(y_train, X_train[f]), 1 - roc_auc_score(y_train, X_train[f]))
                       for f in FEATURE_NAMES],
}, index=FEATURE_NAMES).sort_values("univariate_auc", ascending=False)
uni.round(3)

# %%
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(uni.index[::-1], uni["univariate_auc"][::-1], color="#7A6FD6")
ax.axvline(0.5, color="grey", ls="--", lw=1)
ax.set_xlim(0.45, 1); ax.set_xlabel("univariate ROC-AUC (training set)")
ax.set_title("How well does each feature separate the classes alone?")
savefig(fig, "01_univariate_auc"); plt.show()

# %% [markdown]
# ## 4. Correlations between features

# %%
corr = X_train.corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={"shrink": .6}, ax=ax)
ax.set_title("Pearson correlation between the 30 features (training set)")
savefig(fig, "01_correlation_heatmap"); plt.show()

# %%
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
pairs = upper.stack().dropna().rename("r").reset_index().rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
pairs["|r|"] = pairs["r"].abs()
high = pairs.sort_values("|r|", ascending=False)
print("pairs with |r| > 0.9:", int((high["|r|"] > 0.9).sum()), "of", len(high))
high.head(15).round(3)

# %% [markdown]
# **Strong multicollinearity.** Radius, perimeter and area are almost perfectly
# correlated (r > 0.99) because they are geometric functions of each other, and the
# `_mean` and `_worst` version of the same measurement are also highly correlated.
# 21 feature pairs have |r| > 0.9. This redundancy is the main motivation for the
# project: the 30 features carry much less than 30 dimensions of independent
# information, so PCA (notebook 05) and feature selection (notebook 06) should be able
# to compress them substantially.

# %%
sel = ["radius_worst", "concave_points_worst", "texture_mean", "smoothness_worst"]
g = sns.pairplot(train[sel + ["diagnosis"]], hue="diagnosis", corner=True, plot_kws={"s": 12, "alpha": .6},
                 palette={"benign": "#4C9BE8", "malignant": "#E8684C"})
g.figure.suptitle("Pairwise view of four representative features", y=1.02)
savefig(g.figure, "01_pairplot"); plt.show()

# %% [markdown]
# ## 5. Baseline models
# Both are evaluated with the common 5-fold stratified CV on the training set.
# LDA has no hyper-parameters worth tuning here (we use the default SVD solver), so it
# serves as a strong, simple linear reference. LDA is reported only in this notebook
# (it was listed in both Step 2 and Step 3 of the proposal).

# %%
rows = []
for key in ["dummy", "lda"]:
    pipe = build_pipeline(key)
    rows.append(cv_evaluate(pipe, X_train, y_train, name=display_name(key), key=key, feature_set="all 30"))
    save_best_params(key, {})
save_rows(rows, "cv_01_baselines.csv")
summarize(rows)

# %% [markdown]
# * The **majority baseline** always predicts *benign*: 62.6 % accuracy but **0 %
#   malignant recall** — it misses every cancer. This is the clearest illustration of why
#   accuracy alone is misleading here.
# * **LDA** already reaches ≈95–96 % accuracy and ROC-AUC ≈ 0.99, so this is an
#   "easy" dataset for linear models. Its weak point is malignant recall (≈ 0.89): it
#   misses about 1 in 9 malignant tumours in CV. Every model below has to beat LDA to
#   be interesting.

# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, key in zip(axes, ["dummy", "lda"]):
    pred = cross_val_predict(build_pipeline(key), X_train, y_train, cv=get_cv())
    ConfusionMatrixDisplay.from_predictions(y_train, pred, display_labels=CLASS_NAMES, cmap="Blues",
                                            colorbar=False, ax=ax)
    ax.set_title(f"{display_name(key)} — out-of-fold (CV)")
fig.tight_layout(); savefig(fig, "01_baseline_confusion"); plt.show()

# %% [markdown]
# ## Summary
# * 569 samples, 30 numeric features, no missing values or duplicates; `id` dropped.
# * Moderate imbalance (37 % malignant) → stratified splitting; recall/specificity/F1
#   reported alongside accuracy.
# * Features are on very different scales and skewed → `StandardScaler` inside every
#   pipeline for scale-sensitive models.
# * Heavy redundancy (21 pairs with |r| > 0.9) → strong case for PCA / feature selection.
# * Reference performance: majority 62.6 % accuracy / 0 recall; LDA ≈ 95–96 % accuracy.
