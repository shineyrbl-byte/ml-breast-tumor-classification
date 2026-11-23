"""
From-scratch NumPy implementations of every algorithm used in the project.

Each class follows the scikit-learn estimator API (fit / predict /
predict_proba or decision_function, get_params / set_params), so it plugs
straight into our common protocol: Pipelines, cross_validate, GridSearchCV.
Notebook 08 checks every implementation against scikit-learn on the same
split and folds.

Only NumPy is used inside the algorithms. scikit-learn supplies the base
classes, which give us get_params/set_params and cloning.

  Member 1  StandardScalerScratch, stratified K-fold splitter
  Member 2  LogisticRegressionScratch (gradient descent), GaussianNBScratch
  Member 3  KNNScratch, SVMScratch (SMO dual solver, linear + RBF kernels)
  Member 4  DecisionTreeScratch (CART), RandomForestScratch (bagging)
  Member 5  PCAScratch (covariance eigendecomposition + power iteration)
  Member 6  RFEScratch (recursive feature elimination)
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


def _as_array(X):
    return np.asarray(X, dtype=float)


def _class_weights(y, class_weight):
    """Per-sample weights; 'balanced' = n / (2 * n_c) as in scikit-learn."""
    y = np.asarray(y)
    if class_weight is None:
        return np.ones(len(y))
    if class_weight == "balanced":
        counts = np.bincount(y, minlength=2)
        w = len(y) / (2.0 * counts)
        return w[y]
    return np.array([class_weight.get(c, 1.0) for c in y], dtype=float)


# ==========================================================================
# Member 1: preprocessing and the CV splitter
# ==========================================================================
class StandardScalerScratch(BaseEstimator, TransformerMixin):
    """z = (x - mean) / std, with mean and std learned on training data only."""

    def fit(self, X, y=None):
        X = _as_array(X)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        return (_as_array(X) - self.mean_) / self.scale_


def stratified_kfold_scratch(y, n_splits=5, seed=42):
    """Stratified K-fold: shuffle each class, deal its indices round-robin to folds."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        for i, j in enumerate(idx):
            folds[i % n_splits].append(j)
    all_idx = np.arange(len(y))
    for f in folds:
        test = np.sort(np.array(f))
        yield np.setdiff1d(all_idx, test), test


# ==========================================================================
# Member 2: Logistic Regression and Gaussian Naive Bayes
# ==========================================================================
class LogisticRegressionScratch(BaseEstimator, ClassifierMixin):
    """L2-regularised logistic regression trained by full-batch gradient descent.

    Minimises   (1/n) * sum_i w_i * logloss(y_i, sigmoid(x_i.b + b0))
              + (1 / (2 C n)) * ||b||^2
    which is scikit-learn's objective divided by n, so the same C means the
    same model. Gradient descent uses Nesterov momentum and a step of 1/L,
    where L bounds the Lipschitz constant of the gradient.
    """

    def __init__(self, C=1.0, class_weight=None, max_iter=5000, tol=1e-7):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _sigmoid(z):
        return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable

    def _loss_grad(self, theta, Xb, y, sw):
        n = len(y)
        z = Xb @ theta
        p = self._sigmoid(z)
        # log(1 + e^z) - y z, computed stably
        loss = np.sum(sw * (np.logaddexp(0, z) - y * z)) / n
        reg = np.r_[theta[:-1], 0.0]            # intercept not penalised
        loss += reg @ reg / (2 * self.C * n)
        grad = Xb.T @ (sw * (p - y)) / n + reg / (self.C * n)
        return loss, grad

    def fit(self, X, y):
        X, y = _as_array(X), np.asarray(y)
        self.classes_ = np.array([0, 1])
        n, d = X.shape
        Xb = np.c_[X, np.ones(n)]
        sw = _class_weights(y, self.class_weight)
        # Lipschitz bound of the gradient: max(sw) * ||Xb||_2^2 / (4n) + 1/(Cn)
        L = sw.max() * np.linalg.norm(Xb, 2) ** 2 / (4 * n) + 1 / (self.C * n)
        step = 1.0 / L
        theta = np.zeros(d + 1)
        v = theta.copy()
        t = 1.0
        self.loss_curve_ = []
        for it in range(self.max_iter):
            loss, grad = self._loss_grad(v, Xb, y, sw)
            theta_new = v - step * grad
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            v = theta_new + ((t - 1) / t_new) * (theta_new - theta)
            if np.linalg.norm(theta_new - theta) < self.tol * (1 + np.linalg.norm(theta)):
                theta = theta_new
                break
            theta, t = theta_new, t_new
            self.loss_curve_.append(loss)
        self.n_iter_ = it + 1
        self.coef_ = theta[:-1][None, :]
        self.intercept_ = np.array([theta[-1]])
        return self

    def decision_function(self, X):
        return _as_array(X) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = self._sigmoid(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class GaussianNBScratch(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes: class priors x product of per-feature normals.

    var_smoothing adds  var_smoothing * max(feature variance)  to every
    variance, exactly like scikit-learn. Prediction works in log space.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = _as_array(X), np.asarray(y)
        self.classes_ = np.array([0, 1])
        eps = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + eps
        self.class_prior_ = np.array([(y == c).mean() for c in self.classes_])
        return self

    def _joint_log_likelihood(self, X):
        X = _as_array(X)
        out = []
        for k in range(2):
            ll = -0.5 * np.sum(np.log(2 * np.pi * self.var_[k]))
            ll = ll - 0.5 * np.sum((X - self.theta_[k]) ** 2 / self.var_[k], axis=1)
            out.append(np.log(self.class_prior_[k]) + ll)
        return np.column_stack(out)

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)    # log-sum-exp trick
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self._joint_log_likelihood(X).argmax(axis=1)


# ==========================================================================
# Member 3: KNN and SVM
# ==========================================================================
class KNNScratch(BaseEstimator, ClassifierMixin):
    """k-nearest neighbours with a vectorised Minkowski distance matrix."""

    def __init__(self, n_neighbors=5, p=2, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        self.X_ = _as_array(X)
        self.y_ = np.asarray(y)
        self.classes_ = np.array([0, 1])
        return self

    def _distances(self, X):
        X = _as_array(X)
        if self.p == 2:   # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (one matrix product)
            sq = (X ** 2).sum(1)[:, None] + (self.X_ ** 2).sum(1)[None, :] - 2 * X @ self.X_.T
            return np.sqrt(np.maximum(sq, 0))
        diff = np.abs(X[:, None, :] - self.X_[None, :, :])
        return (diff ** self.p).sum(axis=2) ** (1 / self.p)

    def predict_proba(self, X):
        D = self._distances(X)
        k = self.n_neighbors
        nn = np.argpartition(D, k - 1, axis=1)[:, :k]          # O(n) selection
        d_nn = np.take_along_axis(D, nn, axis=1)
        if self.weights == "distance":
            w = 1.0 / np.maximum(d_nn, 1e-12)
        else:
            w = np.ones_like(d_nn)
        p1 = (w * self.y_[nn]).sum(1) / w.sum(1)
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class SVMScratch(BaseEstimator, ClassifierMixin):
    """Soft-margin kernel SVM solved in the dual with SMO.

    Dual:  min_a  1/2 a^T Q a - 1^T a,   Q_ij = t_i t_j K(x_i, x_j),
           s.t.   0 <= a_i <= C_i,  sum_i t_i a_i = 0,    t_i in {-1, +1}.

    Each iteration picks the "maximal violating pair" (i, j) of the KKT
    conditions, using LIBSVM's second-order working-set selection for j,
    and solves the 2-variable subproblem analytically. The gradient is kept
    up to date incrementally, so one iteration costs O(n).
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", class_weight=None,
                 tol=1e-3, max_iter=100_000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.tol = tol
        self.max_iter = max_iter

    def _kernel(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma_ * np.maximum(sq, 0))

    def fit(self, X, y):
        X, y = _as_array(X), np.asarray(y)
        self.classes_ = np.array([0, 1])
        n = len(y)
        t = np.where(y == 1, 1.0, -1.0)
        if self.gamma == "scale":
            self.gamma_ = 1.0 / (X.shape[1] * X.var())
        else:
            self.gamma_ = float(self.gamma)
        Cs = self.C * _class_weights(y, self.class_weight)
        K = self._kernel(X, X)
        Q = (t[:, None] * t[None, :]) * K
        a = np.zeros(n)
        G = -np.ones(n)                        # gradient of the dual: Q a - 1
        tau = 1e-12
        self.gap_curve_ = []                   # KKT violation per iteration
        for it in range(self.max_iter):
            # I_up / I_low index sets from the KKT conditions
            up = ((t > 0) & (a < Cs)) | ((t < 0) & (a > 0))
            low = ((t > 0) & (a > 0)) | ((t < 0) & (a < Cs))
            score = -t * G
            i = np.flatnonzero(up)[np.argmax(score[up])]
            m_up = score[i]
            m_low = score[low].min()
            self.gap_curve_.append(m_up - m_low)
            if m_up - m_low < self.tol:         # KKT satisfied to tolerance
                break
            # second-order choice of j (Fan, Chen & Lin 2005)
            cand = np.flatnonzero(low & (score < m_up))
            b = m_up - score[cand]
            quad = K[i, i] + K[cand, cand] - 2 * K[i, cand]
            quad = np.where(quad > 0, quad, tau)
            j = cand[np.argmax(b * b / quad)]
            # analytic 2-variable update along the feasible direction
            eta = max(K[i, i] + K[j, j] - 2 * K[i, j], tau)
            delta = (m_up - score[j]) / eta
            # clip so both alphas stay in their boxes
            di = t[i] * delta
            dj = -t[j] * delta
            lim = np.inf
            lim = min(lim, (Cs[i] - a[i]) / di if di > 0 else (a[i] / -di if di < 0 else np.inf))
            lim = min(lim, (Cs[j] - a[j]) / dj if dj > 0 else (a[j] / -dj if dj < 0 else np.inf))
            step = min(1.0, lim)
            di, dj = di * step, dj * step
            a[i] += di
            a[j] += dj
            G += Q[:, i] * di + Q[:, j] * dj
        self.n_iter_ = it + 1
        # bias: average over free support vectors (fall back to the midpoint)
        score = -t * G
        free = (a > 1e-8) & (a < Cs - 1e-8)
        if free.any():
            self.intercept_ = score[free].mean()
        else:
            up = ((t > 0) & (a < Cs)) | ((t < 0) & (a > 0))
            low = ((t > 0) & (a > 0)) | ((t < 0) & (a < Cs))
            self.intercept_ = (score[up].max() + score[low].min()) / 2
        sv = a > 1e-8
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (a * t)[sv]
        return self

    def decision_function(self, X):
        return self._kernel(_as_array(X), self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


# ==========================================================================
# Member 4: CART Decision Tree and Random Forest
# ==========================================================================
class DecisionTreeScratch(BaseEstimator, ClassifierMixin):
    """CART classification tree with Gini or entropy impurity.

    For every node and every feature, the samples are sorted once and every
    threshold is scored in one pass using cumulative class-weight sums,
    so finding a split costs O(d * n log n) instead of O(d * n^2).
    Nodes are stored in flat arrays (feature, threshold, left, right, value).
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1,
                 max_features=None, class_weight=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def _impurity(self, w0, w1):
        tot = w0 + w1
        with np.errstate(invalid="ignore", divide="ignore"):
            p0, p1 = w0 / tot, w1 / tot
            if self.criterion == "gini":
                return 1 - p0 ** 2 - p1 ** 2
            return -(np.where(p0 > 0, p0 * np.log2(p0), 0) + np.where(p1 > 0, p1 * np.log2(p1), 0))

    def _n_features_to_try(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return int(mf)

    def _best_split(self, X, y, sw, idx):
        best = (None, None, -np.inf)
        w_parent = np.array([sw[idx][y[idx] == 0].sum(), sw[idx][y[idx] == 1].sum()])
        parent_imp = self._impurity(*w_parent) * w_parent.sum()
        feats = self._rng.choice(X.shape[1], self._mtry, replace=False) \
            if self._mtry < X.shape[1] else np.arange(X.shape[1])
        msl = self.min_samples_leaf
        n = len(idx)
        for f in feats:
            order = idx[np.argsort(X[idx, f], kind="stable")]
            xs = X[order, f]
            w1 = np.cumsum(sw[order] * (y[order] == 1))
            w0 = np.cumsum(sw[order] * (y[order] == 0))
            # candidate cut after position k (left = first k+1 samples)
            k = np.arange(msl - 1, n - msl)
            k = k[xs[k] < xs[k + 1]]                     # only between distinct values
            if len(k) == 0:
                continue
            l0, l1 = w0[k], w1[k]
            r0, r1 = w_parent[0] - l0, w_parent[1] - l1
            child = self._impurity(l0, l1) * (l0 + l1) + self._impurity(r0, r1) * (r0 + r1)
            gain = parent_imp - child
            g = np.argmax(gain)
            if gain[g] > best[2]:
                best = (f, (xs[k[g]] + xs[k[g] + 1]) / 2, gain[g])
        return best

    def fit(self, X, y, sample_weight=None):
        X, y = _as_array(X), np.asarray(y)
        self.classes_ = np.array([0, 1])
        self._rng = np.random.default_rng(self.random_state)
        self._mtry = self._n_features_to_try(X.shape[1])
        sw = _class_weights(y, self.class_weight)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight)
        self.n_features_in_ = X.shape[1]
        feature, threshold, left, right, value, gain_ = [], [], [], [], [], []

        def new_node(idx):
            w = np.array([sw[idx][y[idx] == 0].sum(), sw[idx][y[idx] == 1].sum()])
            feature.append(-1); threshold.append(0.0)
            left.append(-1); right.append(-1)
            value.append(w / w.sum()); gain_.append(0.0)
            return len(feature) - 1

        root = new_node(np.arange(len(y)))
        stack = [(root, np.arange(len(y)), 0)]
        while stack:                                     # depth-first growth
            node, idx, depth = stack.pop()
            if (self.max_depth is not None and depth >= self.max_depth) \
                    or len(idx) < 2 * self.min_samples_leaf \
                    or len(np.unique(y[idx])) == 1:
                continue
            f, thr, gain = self._best_split(X, y, sw, idx)
            if f is None or gain <= 1e-12:
                continue
            go_left = X[idx, f] <= thr
            li, ri = new_node(idx[go_left]), new_node(idx[~go_left])
            feature[node], threshold[node] = f, thr
            left[node], right[node], gain_[node] = li, ri, gain
            stack += [(li, idx[go_left], depth + 1), (ri, idx[~go_left], depth + 1)]

        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left)
        self.right_ = np.array(right)
        self.value_ = np.array(value)
        imp = np.bincount(self.feature_[self.feature_ >= 0],
                          weights=np.array(gain_)[self.feature_ >= 0],
                          minlength=X.shape[1])
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        self.node_count_ = len(feature)
        return self

    def apply(self, X):
        """Leaf index for every sample (all samples descend the tree together)."""
        X = _as_array(X)
        node = np.zeros(len(X), dtype=int)
        active = self.feature_[node] >= 0
        while active.any():
            a = np.flatnonzero(active)
            f = self.feature_[node[a]]
            go_left = X[a, f] <= self.threshold_[node[a]]
            node[a] = np.where(go_left, self.left_[node[a]], self.right_[node[a]])
            active = self.feature_[node] >= 0
        return node

    def get_depth(self):
        depth = np.zeros(self.node_count_, dtype=int)
        for i in range(self.node_count_):
            if self.feature_[i] >= 0:
                depth[self.left_[i]] = depth[self.right_[i]] = depth[i] + 1
        return depth.max()

    def predict_proba(self, X):
        return self.value_[self.apply(X)]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class RandomForestScratch(BaseEstimator, ClassifierMixin):
    """Bagging of CART trees with a random feature subset at every split.

    Also computes the out-of-bag (OOB) score: each training sample is
    predicted only by the trees whose bootstrap sample did not contain it.
    """

    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, class_weight=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        X, y = _as_array(X), np.asarray(y)
        self.classes_ = np.array([0, 1])
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        oob_sum = np.zeros((n, 2))
        oob_cnt = np.zeros(n)
        for _ in range(self.n_estimators):
            boot = rng.integers(0, n, n)
            counts = np.bincount(boot, minlength=n)    # bootstrap = sample weights
            tree = DecisionTreeScratch(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features, class_weight=self.class_weight,
                random_state=int(rng.integers(1 << 31)))
            inbag = counts > 0
            tree.fit(X[inbag], y[inbag], sample_weight=counts[inbag])
            self.estimators_.append(tree)
            oob = ~inbag
            oob_sum[oob] += tree.predict_proba(X[oob])
            oob_cnt[oob] += 1
        seen = oob_cnt > 0
        self.oob_score_ = np.mean(oob_sum[seen].argmax(1) == y[seen])
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        self.feature_importances_ = imp / imp.sum()
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ==========================================================================
# Member 5: PCA
# ==========================================================================
def power_iteration(A, n_components, n_iter=1000, tol=1e-12, seed=0):
    """Top eigenpairs of a symmetric PSD matrix by power iteration + deflation."""
    rng = np.random.default_rng(seed)
    A = A.copy()
    vals, vecs, iters = [], [], []
    for _ in range(n_components):
        v = rng.normal(size=A.shape[0])
        v /= np.linalg.norm(v)
        lam = 0.0
        for it in range(n_iter):
            w = A @ v
            lam_new = v @ w
            v = w / np.linalg.norm(w)
            if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
                break
            lam = lam_new
        vals.append(lam_new); vecs.append(v); iters.append(it + 1)
        A -= lam_new * np.outer(v, v)                    # deflation
    return np.array(vals), np.array(vecs), iters


class PCAScratch(BaseEstimator, TransformerMixin):
    """PCA = eigendecomposition of the covariance matrix of centred data.

    n_components: int, or a float in (0, 1) = fraction of variance to keep.
    solver: "eigh" (exact symmetric eigensolver) or "power" (power iteration).
    Component signs are fixed so the largest-magnitude loading is positive.
    """

    def __init__(self, n_components=None, solver="eigh"):
        self.n_components = n_components
        self.solver = solver

    def fit(self, X, y=None):
        X = _as_array(X)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / (n - 1)
        vals, vecs = np.linalg.eigh(cov)                 # ascending order
        vals, vecs = vals[::-1], vecs[:, ::-1].T
        vals = np.maximum(vals, 0)
        ratio = vals / vals.sum()
        k = self.n_components
        if k is None:
            k = d
        elif isinstance(k, float) and 0 < k < 1:
            k = int(np.searchsorted(np.cumsum(ratio), k) + 1)
        if self.solver == "power":
            vals_k, vecs_k, self.power_iters_ = power_iteration(cov, k)
        else:
            vals_k, vecs_k = vals[:k], vecs[:k]
        signs = np.sign(vecs_k[np.arange(k), np.abs(vecs_k).argmax(1)])
        self.components_ = vecs_k * signs[:, None]
        self.explained_variance_ = vals_k
        self.explained_variance_ratio_ = vals_k / vals.sum()
        self.n_components_ = k
        return self

    def transform(self, X):
        return (_as_array(X) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


# ==========================================================================
# Member 6: Recursive Feature Elimination
# ==========================================================================
class RFEScratch(BaseEstimator, TransformerMixin):
    """Fit, drop the feature with the smallest |coefficient|, repeat until k remain.

    estimator must expose coef_ (e.g. LogisticRegressionScratch). Features are
    standardised inside each step so coefficient magnitudes are comparable.
    """

    def __init__(self, estimator=None, n_features_to_select=5, step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step

    def fit(self, X, y):
        from sklearn.base import clone
        X, y = _as_array(X), np.asarray(y)
        est = self.estimator if self.estimator is not None else LogisticRegressionScratch()
        remaining = list(range(X.shape[1]))
        self.ranking_ = np.ones(X.shape[1], dtype=int)
        rank = X.shape[1]
        while len(remaining) > self.n_features_to_select:
            Xs = StandardScalerScratch().fit_transform(X[:, remaining])
            coef = np.abs(clone(est).fit(Xs, y).coef_[0])
            n_drop = min(self.step, len(remaining) - self.n_features_to_select)
            for pos in sorted(np.argsort(coef)[:n_drop], reverse=True):
                self.ranking_[remaining[pos]] = rank
                rank -= 1
                remaining.pop(pos)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[remaining] = True
        return self

    def transform(self, X):
        return _as_array(X)[:, self.support_]
