# %% [markdown]
# # 08: Implementing the algorithms from scratch
# **All members.** Notebooks 01–07 use scikit-learn. Here every member re-implements the
# algorithm from their own part using only NumPy (`src/scratch.py`) and checks it against
# scikit-learn on the same data, split, folds and hyper-parameters. Then we run three
# heavier computational experiments: a runtime benchmark, nested cross-validation and a
# permutation test.
#
# | Member | Implemented from scratch |
# |---|---|
# | 1 | standardisation, stratified K-fold splitter, nested CV, permutation test |
# | 2 | logistic regression (accelerated gradient descent), Gaussian Naive Bayes |
# | 3 | KNN (vectorised distances), kernel SVM (SMO dual solver) |
# | 4 | CART decision tree, Random Forest with out-of-bag score |
# | 5 | PCA (covariance eigendecomposition and power iteration) |
# | 6 | recursive feature elimination (RFE) |
#
# The scratch classes follow the scikit-learn API, so they run through our common
# protocol (`cv_evaluate`, Pipelines) unchanged. The comparison is like for like.

# %%
import sys, time
sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, cross_val_score, permutation_test_score, StratifiedKFold
from common import *
from scratch import *

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160, "display.max_columns", 40)
X_train, X_test, y_train, y_test = get_split()
Xtr, Xte, ytr, yte = X_train.values, X_test.values, y_train.values, y_test.values
BP = {k: {p.replace("clf__", ""): v for p, v in d.items()} for k, d in load_best_params().items()}
sc = StandardScaler().fit(Xtr)
A, B = sc.transform(Xtr), sc.transform(Xte)   # scaled train / test for direct comparisons
BP

# %% [markdown]
# ## 1. Member 1: standardisation and the stratified K-fold splitter
# The scaler learns the mean and standard deviation on the training data only. The splitter
# shuffles each class separately and deals its samples to the folds in turn, so every fold
# keeps the 37/63 malignant/benign ratio.

# %%
diff = np.abs(StandardScalerScratch().fit(Xtr).transform(Xte) - B).max()
print(f"max |scratch scaler - sklearn scaler| on the test set: {diff:.1e}")
rows = []
for name, splits in [("scratch", stratified_kfold_scratch(ytr, 5, seed=RANDOM_STATE)),
                     ("sklearn", get_cv().split(Xtr, ytr))]:
    for k, (tr, te) in enumerate(splits):
        rows.append({"splitter": name, "fold": k + 1, "test size": len(te),
                     "malignant %": round(100 * ytr[te].mean(), 1)})
pd.DataFrame(rows).pivot(index="fold", columns="splitter", values=["test size", "malignant %"])

# %%
lr_s = Pipeline([("scaler", StandardScalerScratch()),
                 ("clf", LogisticRegressionScratch(C=BP["logreg"]["C"], class_weight=BP["logreg"]["class_weight"]))])
f1_own = cross_val_score(lr_s, Xtr, ytr, cv=list(stratified_kfold_scratch(ytr, 5, RANDOM_STATE)), scoring="f1")
f1_skl = cross_val_score(lr_s, Xtr, ytr, cv=get_cv(), scoring="f1")
print(f"scratch LR, CV F1 with our splitter:     {f1_own.mean():.4f} ± {f1_own.std():.4f}")
print(f"scratch LR, CV F1 with sklearn splitter: {f1_skl.mean():.4f} ± {f1_skl.std():.4f}")

# %% [markdown]
# Both splitters give perfectly stratified folds. Their CV estimates differ only through
# which samples land in which fold. That difference is itself a reminder of how noisy a
# single 5-fold estimate is (we quantify this in Section 9).

# %% [markdown]
# ## 2. Member 2: logistic regression and Gaussian Naive Bayes
# **Logistic regression.** We minimise the regularised log-loss with gradient descent:
#
# $$J(\beta) = \frac{1}{n}\sum_i w_i\left[\log(1+e^{z_i}) - y_i z_i\right] + \frac{1}{2Cn}\lVert\beta\rVert^2,
# \qquad z_i = x_i^\top\beta + \beta_0, \qquad \nabla J = \frac{1}{n}X^\top\big(w\odot(\sigma(z)-y)\big) + \frac{\beta}{Cn}$$
#
# This is scikit-learn's objective divided by $n$, so the same `C` gives the same model.
# The step size is $1/L$, where $L$ bounds the gradient's Lipschitz constant. Nesterov
# momentum speeds up convergence.

# %%
p = BP["logreg"]
lr_scr = LogisticRegressionScratch(C=p["C"], class_weight=p["class_weight"]).fit(A, ytr)
lr_skl = LogisticRegression(C=p["C"], class_weight=p["class_weight"], tol=1e-10, max_iter=10000).fit(A, ytr)
print(f"gradient descent converged in {lr_scr.n_iter_} iterations")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
lc = np.array(lr_scr.loss_curve_)
ax[0].semilogy(lc - lc.min() + 1e-12)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("J(β) − J*  (log scale)")
ax[0].set_title("Gradient descent convergence")
ax[1].scatter(lr_skl.coef_[0], lr_scr.coef_[0])
lim = np.abs(lr_skl.coef_).max() * 1.1
ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax[1].set_xlabel("sklearn coefficient"); ax[1].set_ylabel("scratch coefficient")
ax[1].set_title("Learned coefficients (30 features)")
savefig(fig, "08_logreg_scratch"); plt.show()
print(f"max coefficient difference: {np.abs(lr_scr.coef_ - lr_skl.coef_).max():.1e}")
print(f"test-set predictions identical: {np.mean(lr_scr.predict(B) == lr_skl.predict(B)):.0%}")

# %% [markdown]
# **Gaussian Naive Bayes.** For each class we estimate a prior and, for each feature, a
# mean and a variance. Prediction maximises $\log P(c) + \sum_j \log \mathcal N(x_j;\mu_{cj},\sigma^2_{cj})$.
# The log-sum-exp trick keeps the probabilities from underflowing.

# %%
gnb_scr = GaussianNBScratch(**BP["gnb"]).fit(A, ytr)
gnb_skl = GaussianNB(**BP["gnb"]).fit(A, ytr)
print(f"max probability difference: {np.abs(gnb_scr.predict_proba(B) - gnb_skl.predict_proba(B)).max():.1e}")

# %% [markdown]
# ## 3. Member 3: KNN and kernel SVM
# **KNN.** The Euclidean distance matrix is one matrix product,
# $\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\top b$. The $k$ nearest
# neighbours are found with `argpartition`, which takes $O(n)$ time rather than a full sort.

# %%
knn_scr = KNNScratch(**BP["knn"]).fit(A, ytr)
knn_skl = KNeighborsClassifier(**BP["knn"]).fit(A, ytr)
print(f"KNN (k={BP['knn']['n_neighbors']}, p={BP['knn']['p']}) probability difference: "
      f"{np.abs(knn_scr.predict_proba(B) - knn_skl.predict_proba(B)).max():.1e}")

# %% [markdown]
# **SVM via SMO.** We solve the soft-margin dual problem
#
# $$\min_\alpha \tfrac12\alpha^\top Q\alpha - \mathbf 1^\top\alpha \quad\text{s.t.}\quad 0\le\alpha_i\le C_i,\ \ \textstyle\sum_i t_i\alpha_i = 0,
# \qquad Q_{ij} = t_i t_j K(x_i,x_j),\ K(x,x') = e^{-\gamma\lVert x-x'\rVert^2}$$
#
# with Sequential Minimal Optimisation, the algorithm inside LIBSVM. Each step picks the
# pair $(i,j)$ that most violates the KKT optimality conditions, using second-order
# working-set selection, and solves that 2-variable problem exactly. It stops when the
# violation gap drops below $10^{-3}$.

# %%
p = BP["svm"]
svm_scr = SVMScratch(C=p["C"], gamma=p["gamma"], class_weight=p["class_weight"]).fit(A, ytr)
svm_skl = SVC(C=p["C"], gamma=p["gamma"], class_weight=p["class_weight"]).fit(A, ytr)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].semilogy(svm_scr.gap_curve_); ax[0].axhline(1e-3, color="k", ls="--", lw=1)
ax[0].set_xlabel("SMO iteration"); ax[0].set_ylabel("KKT violation gap (log)")
ax[0].set_title(f"SMO convergence ({svm_scr.n_iter_} iterations)")
ax[1].scatter(svm_skl.decision_function(B), svm_scr.decision_function(B), c=yte, cmap="coolwarm")
ax[1].plot([-4, 4], [-4, 4], "k--", lw=1)
ax[1].set_xlabel("sklearn decision value"); ax[1].set_ylabel("scratch decision value")
ax[1].set_title("RBF SVM decision values on the test set")
savefig(fig, "08_svm_smo"); plt.show()
print(f"support vectors: scratch {len(svm_scr.support_)}, sklearn {len(svm_skl.support_)}")
print(f"test-set predictions identical: {np.mean(svm_scr.predict(B) == svm_skl.predict(B)):.0%}")

# %% [markdown]
# ## 4. Member 4: CART decision tree and Random Forest
# **CART.** At every node, for every feature, we sort the samples once and score all
# thresholds in one pass using cumulative class counts. That costs $O(d\,n\log n)$ per node
# instead of $O(d\,n^2)$. The split with the largest weighted Gini decrease wins.
# Prediction sends all samples down the tree together.

# %%
rows = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, None]:
    for name, est in [("scratch", DecisionTreeScratch(max_depth=d)),
                      ("sklearn", DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE))]:
        cv = cross_val_score(est, Xtr, ytr, cv=get_cv(), scoring="accuracy")
        rows.append({"max_depth": str(d), "impl": name, "train": est.fit(Xtr, ytr).score(Xtr, ytr),
                     "cv": cv.mean()})
dd = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7, 4))
for (name, g), m in zip(dd.groupby("impl"), ["o", "s"]):
    ax.plot(g["max_depth"], g["train"], m + "--", label=f"{name} train")
    ax.plot(g["max_depth"], g["cv"], m + "-", label=f"{name} CV")
ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(ncol=2)
ax.set_title("CART from scratch reproduces the overfitting curve")
savefig(fig, "08_tree_depth_scratch"); plt.show()
order = ["1", "2", "3", "4", "5", "6", "8", "10", "None"]
dd.pivot(index="max_depth", columns="impl", values=["train", "cv"]).loc[order].round(3)

# %% [markdown]
# **Random Forest.** Each tree is trained on a bootstrap sample and tries a random subset
# of features at every split. The samples a tree never saw (about 37%) give a free
# **out-of-bag** estimate of accuracy, with no separate validation set needed.

# %%
p = BP["rf"]
t0 = time.time()
rf_scr = RandomForestScratch(n_estimators=p["n_estimators"], max_features=p["max_features"],
                             min_samples_leaf=p["min_samples_leaf"], random_state=RANDOM_STATE).fit(Xtr, ytr)
print(f"scratch RF ({p['n_estimators']} trees) trained in {time.time() - t0:.1f}s, OOB accuracy {rf_scr.oob_score_:.4f}")
rf_skl = RandomForestClassifier(**p, oob_score=True, random_state=RANDOM_STATE).fit(Xtr, ytr)
print(f"sklearn RF OOB accuracy {rf_skl.oob_score_:.4f}")

oob = []
for n in [1, 5, 10, 25, 50, 100, 200]:
    m = RandomForestScratch(n_estimators=n, max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                            random_state=RANDOM_STATE).fit(Xtr, ytr)
    oob.append((n, m.oob_score_))
imp = pd.DataFrame({"scratch": rf_scr.feature_importances_, "sklearn": rf_skl.feature_importances_},
                   index=FEATURE_NAMES).sort_values("sklearn", ascending=False).head(10)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(*zip(*oob), "o-"); ax[0].set_xscale("log")
ax[0].set_xlabel("number of trees"); ax[0].set_ylabel("OOB accuracy")
ax[0].set_title("Scratch RF: more trees → lower variance")
imp[::-1].plot.barh(ax=ax[1]); ax[1].set_title("Top-10 Gini importances")
savefig(fig, "08_rf_scratch"); plt.show()

# %% [markdown]
# ## 5. Member 5: PCA from the covariance matrix
# PCA = eigendecomposition of the sample covariance $\Sigma = \frac{1}{n-1}X_c^\top X_c$.
# The eigenvectors are the principal axes and the eigenvalues are the variance along each.
# We implement two solvers: NumPy's exact symmetric eigensolver (`eigh`), and **power
# iteration with deflation** written by hand: repeat $v \leftarrow \Sigma v/\lVert\Sigma v\rVert$,
# then subtract $\lambda vv^\top$ and find the next component.

# %%
pca_skl = PCA(10).fit(A)
pca_eig = PCAScratch(10).fit(A)
pca_pow = PCAScratch(10, solver="power").fit(A)
evr = pd.DataFrame({"sklearn (SVD)": pca_skl.explained_variance_ratio_,
                    "scratch eigh": pca_eig.explained_variance_ratio_,
                    "scratch power": pca_pow.explained_variance_ratio_,
                    "power iterations": pca_pow.power_iters_}, index=[f"PC{i+1}" for i in range(10)])
lam = PCAScratch().fit(A).explained_variance_
evr["eigengap λ(k+1)/λ(k)"] = lam[1:11] / lam[:10]
print(f"max |component| difference vs sklearn: eigh {np.abs(np.abs(pca_eig.components_) - np.abs(pca_skl.components_)).max():.1e}, "
      f"power {np.abs(np.abs(pca_pow.components_) - np.abs(pca_skl.components_)).max():.1e}")
print("components for 95% variance:", PCAScratch(0.95).fit(A).n_components_)
evr.round(4)

# %% [markdown]
# Power iteration converges at a rate set by the ratio $\lambda_{k+1}/\lambda_k$. When two
# eigenvalues are close (ratio near 1), it needs many more iterations, as the table shows.

# %%
ks = range(1, 31)
err = [np.mean((A - PCAScratch(k).fit(A).inverse_transform(PCAScratch(k).fit(A).transform(A))) ** 2) for k in ks]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(ks, err, "o-", ms=3); ax.axvline(10, color="k", ls="--", lw=1)
ax.set_xlabel("number of components k"); ax.set_ylabel("mean squared reconstruction error")
ax.set_title("Reconstruction error = 1 − cumulative variance (standardised data)")
savefig(fig, "08_pca_reconstruction"); plt.show()

# %% [markdown]
# ## 6. Member 6: recursive feature elimination
# Fit logistic regression, drop the feature with the smallest |coefficient|, refit, and
# repeat until $k$ features remain. Our RFE uses our own logistic regression inside.

# %%
rfe_scr = RFEScratch(LogisticRegressionScratch(C=1.0), n_features_to_select=5).fit(Xtr, ytr)
rfe_skl = RFE(LogisticRegression(C=1.0, max_iter=5000), n_features_to_select=5).fit(A, ytr)
print("scratch RFE keeps:", [FEATURE_NAMES[i] for i in np.flatnonzero(rfe_scr.support_)])
print("sklearn RFE keeps:", [FEATURE_NAMES[i] for i in np.flatnonzero(rfe_skl.support_)])
rank = pd.DataFrame({"scratch": rfe_scr.ranking_, "sklearn rank (1 = kept)": rfe_skl.ranking_}, index=FEATURE_NAMES)
print(f"Spearman correlation of elimination order: {rank.corr(method='spearman').iloc[0, 1]:.3f}")

rows = []
for k in [5, 10, 15]:
    pipe = Pipeline([("rfe", RFEScratch(LogisticRegressionScratch(C=1.0), k)),
                     ("scaler", StandardScalerScratch()),
                     ("clf", LogisticRegressionScratch(C=BP["logreg"]["C"], class_weight="balanced"))])
    rows.append(cv_evaluate(pipe, Xtr, ytr, name="scratch RFE → scratch LR", k=k))
summarize(rows)[["model", "k", "accuracy", "recall", "f1"]]

# %% [markdown]
# Selection happens inside each CV fold, as part of the pipeline, so there is no leakage.
# A fully hand-written pipeline (RFE, scaler, classifier) with 5 features reaches roughly
# the same CV F1 as notebook 06's scikit-learn version.

# %% [markdown]
# ## 7. Head-to-head: every scratch model vs scikit-learn
# Same tuned hyper-parameters (from `best_params.json`), same 5 folds, same metrics. We
# also report how often the two implementations give the same prediction on the held-out
# test set. Nothing is tuned here, so looking at the test set is safe.

# %%
def pair(key):
    p = BP[key]
    if key == "logreg":
        return (LogisticRegressionScratch(C=p["C"], class_weight=p["class_weight"]),
                LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=10000), True)
    if key == "gnb":
        return GaussianNBScratch(**p), GaussianNB(**p), True
    if key == "knn":
        return KNNScratch(**p), KNeighborsClassifier(**p), True
    if key == "svm":
        return (SVMScratch(C=p["C"], gamma=p["gamma"], class_weight=p["class_weight"]),
                SVC(C=p["C"], gamma=p["gamma"], class_weight=p["class_weight"]), True)
    if key == "dtree":
        q = {k: v for k, v in p.items() if k != "ccp_alpha"}
        return DecisionTreeScratch(**q), DecisionTreeClassifier(**q, random_state=RANDOM_STATE), False
    if key == "rf":
        q = dict(n_estimators=p["n_estimators"], max_features=p["max_features"],
                 min_samples_leaf=p["min_samples_leaf"], max_depth=p["max_depth"], class_weight=p["class_weight"])
        return RandomForestScratch(**q, random_state=RANDOM_STATE), RandomForestClassifier(**q, random_state=RANDOM_STATE), False

rows, agree = [], {}
for key in ["logreg", "gnb", "knn", "svm", "dtree", "rf"]:
    scr, skl, scale = pair(key)
    pipes = {"scratch": Pipeline(([("scaler", StandardScalerScratch())] if scale else []) + [("clf", scr)]),
             "sklearn": Pipeline(([("scaler", StandardScaler())] if scale else []) + [("clf", skl)])}
    for impl, pipe in pipes.items():
        rows.append(cv_evaluate(pipe, Xtr, ytr, name=key, impl=impl))
    preds = [pp.fit(Xtr, ytr).predict(Xte) for pp in pipes.values()]
    agree[key] = np.mean(preds[0] == preds[1])
h2h = pd.DataFrame(rows)
h2h["model"] = h2h["model"].map({"logreg": "Logistic Regression", "gnb": "Gaussian NB", "knn": "KNN",
                                 "svm": "SVM (RBF)", "dtree": "Decision Tree", "rf": "Random Forest"})
save_rows(rows, "cv_08_scratch_vs_sklearn.csv")
tbl = h2h.pivot(index="model", columns="impl", values="cv_f1").round(4)
tbl["|Δ F1|"] = (tbl["scratch"] - tbl["sklearn"]).abs().round(4)
tbl["test predictions identical"] = [f"{agree[k]:.1%}" for k in
                                     tbl.index.map({"Logistic Regression": "logreg", "Gaussian NB": "gnb", "KNN": "knn",
                                                    "SVM (RBF)": "svm", "Decision Tree": "dtree", "Random Forest": "rf"})]
tbl

# %%
fig, ax = plt.subplots(figsize=(10, 4.2))
pv = h2h.pivot(index="model", columns="impl", values="cv_f1")
er = h2h.pivot(index="model", columns="impl", values="cv_f1_std")
pv.plot.bar(yerr=er, ax=ax, capsize=3, rot=15)
ax.set_ylim(0.85, 1.02); ax.set_ylabel("CV F1 (malignant)"); ax.set_xlabel("")
ax.legend(ncol=2, loc="upper left")
ax.set_title("From-scratch implementations match scikit-learn")
savefig(fig, "08_scratch_vs_sklearn"); plt.show()

# %% [markdown]
# The deterministic algorithms (LR, GNB, KNN, SVM) match scikit-learn to numerical
# precision: the same F1 to four decimals and identical test predictions. The trees differ
# slightly because there are ties. When two splits are equally good, scikit-learn breaks the
# tie with a random feature order and we take the first one. Random Forests also use
# different random numbers. The gaps (0.010 F1 for the tree, 0.007 for the forest) are well
# inside the fold-to-fold standard deviation (0.03), and the test predictions agree on 99–100%.

# %% [markdown]
# ## 8. Runtime benchmark
# Median fit + predict time over 5 runs on the 455-sample training set, then how the cost
# grows with the number of training samples $n$.

# %%
def timeit(est, Xa, Xb, y, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); est.fit(Xa, y); est.predict(Xb); ts.append(time.perf_counter() - t)
    return np.median(ts) * 1000

rows = []
for key in ["logreg", "gnb", "knn", "svm", "dtree", "rf"]:
    scr, skl, scale = pair(key)
    Xa, Xb = (A, B) if scale else (Xtr, Xte)
    reps = 2 if key == "rf" else 5
    rows.append({"model": key, "scratch ms": timeit(scr, Xa, Xb, ytr, reps), "sklearn ms": timeit(skl, Xa, Xb, ytr, reps)})
rt = pd.DataFrame(rows)
rt["slowdown ×"] = rt["scratch ms"] / rt["sklearn ms"]
save_rows(rt.to_dict("records"), "runtime_08.csv")
rt.round(2)

# %%
sizes = [50, 100, 200, 300, 455]
curves = {k: [] for k in ["KNN", "SVM (SMO)", "CART", "LR (GD)"]}
rng = np.random.default_rng(RANDOM_STATE)
for n in sizes:
    idx = np.sort(rng.choice(len(ytr), n, replace=False)) if n < len(ytr) else np.arange(len(ytr))
    curves["KNN"].append(timeit(KNNScratch(3, p=1), A[idx], A, ytr[idx], 3))
    curves["SVM (SMO)"].append(timeit(SVMScratch(**{k: BP["svm"][k] for k in ["C", "gamma", "class_weight"]}), A[idx], B, ytr[idx], 3))
    curves["CART"].append(timeit(DecisionTreeScratch(), Xtr[idx], Xte, ytr[idx], 3))
    curves["LR (GD)"].append(timeit(LogisticRegressionScratch(C=0.1, class_weight="balanced"), A[idx], B, ytr[idx], 3))
fig, ax = plt.subplots(figsize=(7, 4))
for k, v in curves.items():
    ax.loglog(sizes, v, "o-", label=k)
ax.set_xlabel("training samples n"); ax.set_ylabel("fit + predict time (ms)")
ax.set_title("How our implementations scale with n"); ax.legend()
savefig(fig, "08_runtime_scaling"); plt.show()
slopes = {k: np.polyfit(np.log(sizes), np.log(v), 1)[0] for k, v in curves.items()}
pd.Series(slopes, name="empirical exponent (time ∝ n^a)").round(2)

# %% [markdown]
# On the log–log plot, the slope is the empirical complexity exponent. In theory KNN
# prediction is $O(n)$ per query, the SVM kernel matrix is $O(n^2)$ and CART is
# $O(d\,n\log n)$ per tree level. At our small sizes ($n \le 455$) fixed per-iteration
# Python overhead dominates, so all the measured exponents are around 1.1–1.4. The
# quadratic kernel cost would only show at much larger $n$.
#
# Most of our NumPy implementations are within 2–4× of scikit-learn's compiled C/Cython
# code. Our Gaussian NB is even faster because it skips input validation. The exception is
# logistic regression (about 20× slower): first-order gradient descent needs about 1,500
# iterations, while scikit-learn's solver uses curvature information and converges in a
# handful of steps. That is a good illustration of why optimisers matter.

# %% [markdown]
# ## 9. Nested cross-validation: is our tuned CV score optimistic?
# In notebooks 02–04 the hyper-parameters were chosen to maximise the 5-fold CV F1, and we
# reported that same maximum. Choosing the best of many candidates biases that number
# upwards. **Nested CV** removes the bias. An inner 5-fold grid search tunes the model
# using only the outer-training part of each of 5 outer folds, and the tuned model is then
# scored on the outer-test fold it never saw. For the RBF SVM that is
# 5 outer × 162 candidates × 5 inner = **4,050 model fits**, plus refits.

# %%
from models import build_pipeline, GRIDS
rows = []
for key in ["logreg", "knn", "svm", "rf"]:
    outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 1)
    t = time.time()
    inner = GridSearchCV(build_pipeline(key), GRIDS[key], cv=get_cv(), scoring="f1", n_jobs=-1)
    nested = cross_val_score(inner, Xtr, ytr, cv=outer, scoring="f1")
    flat = tune(build_pipeline(key), GRIDS[key], Xtr, ytr).best_score_
    rows.append({"model": key, "non-nested (best grid CV F1)": flat, "nested CV F1": nested.mean(),
                 "nested std": nested.std(), "optimism": flat - nested.mean(), "seconds": time.time() - t})
nest = pd.DataFrame(rows)
save_rows(rows, "nested_cv_08.csv")
nest.round(4)

# %% [markdown]
# The optimism column shows how much the "best grid score" overstates performance on unseen
# data: about 0.01–0.02 F1 for every model. That is roughly the size of one fold's standard
# deviation, so it is worth reporting honestly. It does not change the conclusions: the RBF
# SVM and logistic regression stay on top (nested F1 about 0.955) and Random Forest stays
# below them. This is also why notebook 07 judges the final models on the untouched test set.

# %% [markdown]
# ## 10. Permutation test: is the model learning anything real?
# We shuffle the labels 1,000 times, re-run the full 5-fold CV of our **from-scratch**
# logistic regression each time, and see where the real score falls in that "no signal"
# distribution. That is 5,000 model fits.

# %%
t = time.time()
pipe = Pipeline([("scaler", StandardScalerScratch()),
                 ("clf", LogisticRegressionScratch(C=BP["logreg"]["C"], class_weight="balanced", max_iter=2000))])
score, perm, pval = permutation_test_score(pipe, Xtr, ytr, cv=get_cv(), scoring="f1",
                                           n_permutations=1000, random_state=RANDOM_STATE, n_jobs=-1)
print(f"{time.time() - t:.0f}s; real CV F1 = {score:.4f}; permuted mean = {perm.mean():.4f} ± {perm.std():.4f}; "
      f"max permuted = {perm.max():.4f}; p-value = {pval:.4f}")
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(perm, bins=40, color="grey", label="F1 with shuffled labels")
ax.axvline(score, color="crimson", lw=2, label=f"real labels ({score:.3f})")
ax.set_xlabel("CV F1"); ax.set_ylabel("count"); ax.legend()
ax.set_title(f"Permutation test (1,000 shuffles), p = {pval:.4f}")
savefig(fig, "08_permutation_test"); plt.show()

# %% [markdown]
# With shuffled labels, F1 is about 0.42 (chance level for a balanced classifier). Our real score is far beyond every
# permuted score, so p = 1/1001, the smallest value 1,000 permutations can give. The
# relationship between the measurements and the diagnosis is real and not an artefact of
# the CV procedure.
#
# ## Summary
# * All six members' algorithms were implemented in NumPy and plug into the same pipeline.
# * The deterministic models reproduce scikit-learn exactly: coefficients and probabilities
#   to $10^{-6}$ or better, SVM decision values to the SMO stopping tolerance, and the same CV
#   F1 and test predictions. The trees and forests agree within CV noise.
# * Convergence analyses: GD loss curve, SMO KKT gap, power-iteration speed vs eigengap,
#   and OOB accuracy vs number of trees.
# * Heavier experiments: a runtime/complexity benchmark, nested CV (about 4k fits per
#   model) showing that tuning optimism is about 0.01–0.02 F1, and a permutation test
#   (5k fits) with p ≈ 0.001.
