"""Pipeline factory so every notebook builds models the same way.

Every pipeline names its classifier step ``clf`` (and its scaler ``scaler``),
so tuned parameters such as ``clf__C`` can be re-applied anywhere, including
pipelines that insert a PCA / feature-selection step called ``reduce``.
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE

# key -> (display name, estimator factory, needs scaling)
MODELS = {
    "dummy":  ("Majority baseline", lambda: DummyClassifier(strategy="most_frequent"), False),
    "lda":    ("LDA", lambda: LinearDiscriminantAnalysis(), True),
    "logreg": ("Logistic Regression", lambda: LogisticRegression(
        solver="liblinear", max_iter=5000, random_state=RANDOM_STATE), True),
    "gnb":    ("Gaussian NB", lambda: GaussianNB(), True),
    "knn":    ("KNN", lambda: KNeighborsClassifier(), True),
    "svm":    ("SVM", lambda: SVC(random_state=RANDOM_STATE), True),
    "dtree":  ("Decision Tree", lambda: DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "rf":     ("Random Forest", lambda: RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1), False),
}


def display_name(key: str) -> str:
    return MODELS[key][0]


def build_pipeline(key: str, params: dict | None = None, reducer=None,
                   scale: bool | None = None) -> Pipeline:
    """Scaler (if needed) -> optional reducer (PCA / selector) -> classifier.

    Scaling, PCA and selection live inside the pipeline, so in CV they are
    re-fitted on each training fold only (no leakage).
    """
    _, factory, needs_scaling = MODELS[key]
    if scale is None:
        # PCA and most selectors are scale-sensitive -> always scale with them
        scale = needs_scaling or reducer is not None
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if reducer is not None:
        steps.append(("reduce", reducer))
    steps.append(("clf", factory()))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**{k: v for k, v in params.items() if k.startswith("clf__")})
    return pipe


# --------------------------------------------------------------------------
# Hyper-parameter grids (used by notebooks 02-04, and by notebook 05's
# re-tuning sanity check, so the search space is identical everywhere)
# --------------------------------------------------------------------------
import numpy as np  # noqa: E402

GRIDS = {
    "logreg": {
        "clf__C": np.logspace(-3, 3, 13),
        "clf__l1_ratio": [0, 1],                 # 0 = L2, 1 = L1
        "clf__class_weight": [None, "balanced"],
    },
    "gnb": {"clf__var_smoothing": np.logspace(-12, 0, 25)},
    "knn": {
        "clf__n_neighbors": list(range(1, 32, 2)),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],                        # 1 = Manhattan, 2 = Euclidean
    },
    "svm": [
        {"clf__kernel": ["linear"], "clf__C": np.logspace(-3, 2, 11),
         "clf__class_weight": [None, "balanced"]},
        {"clf__kernel": ["rbf"], "clf__C": np.logspace(-1, 3, 9),
         "clf__gamma": np.logspace(-4, 0, 9), "clf__class_weight": [None, "balanced"]},
    ],
    "dtree": {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [2, 3, 4, 5, 6, 8, None],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__ccp_alpha": [0.0, 0.002, 0.005, 0.01, 0.02],
        "clf__class_weight": [None, "balanced"],
    },
    "rf": {
        "clf__n_estimators": [200, 500],
        "clf__max_features": ["sqrt", "log2", 0.5],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__class_weight": [None, "balanced"],
    },
}
