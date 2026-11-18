"""Custom feature selector used in notebook 06."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression

from common import RANDOM_STATE


class L1PathSelector(SelectorMixin, BaseEstimator):
    """Select exactly ``k`` features using the L1-regularisation path.

    C is increased from very strong to weak regularisation, and the first
    (smallest) C whose L1 logistic regression has at least ``k`` non-zero
    coefficients is used. The k features with the largest |coefficient| at
    that C are kept, i.e. the first features to "enter" the L1 path.
    If even the weakest C on the grid (C=100) has fewer than k non-zero
    weights, the remaining slots are filled by |coefficient| order at that C.
    Expects standardised inputs (put a StandardScaler before it).
    """

    def __init__(self, k=10, Cs=None):
        self.k = k
        self.Cs = Cs

    def fit(self, X, y):
        Cs = self.Cs if self.Cs is not None else np.logspace(-3, 2, 51)
        X = np.asarray(X)
        coef = None
        if self.k >= X.shape[1]:          # nothing to select
            Cs = Cs[-1:]
        for C in Cs:
            m = LogisticRegression(l1_ratio=1, solver="liblinear", C=C, max_iter=5000,
                                   random_state=RANDOM_STATE).fit(X, y)
            coef = np.abs(m.coef_[0])
            if (coef > 0).sum() >= self.k:
                break
        self.C_ = C
        self.coef_ = coef
        self.n_features_in_ = X.shape[1]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(-coef, kind="stable")[: self.k]] = True
        self.mask_ = mask
        return self

    def _get_support_mask(self):
        return self.mask_


METHODS = ["MI", "RFE", "L1"]
METHOD_NAMES = {"MI": "SelectKBest (MI)", "RFE": "RFE (LR)", "L1": "L1 path"}


def make_selector(method: str, k: int):
    """The three feature-selection methods of the project, asked for k features."""
    from functools import partial
    from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
    if method == "MI":
        return SelectKBest(partial(mutual_info_classif, random_state=RANDOM_STATE), k=k)
    if method == "RFE":
        return RFE(LogisticRegression(solver="liblinear", max_iter=5000, random_state=RANDOM_STATE),
                   n_features_to_select=k, step=1)
    if method == "L1":
        return L1PathSelector(k=k)
    raise ValueError(method)
