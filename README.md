# Breast Tumor Classification — Classical ML, PCA and Feature Selection

Machine Learning course project. We classify breast tumours (UCI Breast Cancer Wisconsin
Diagnostic, 569 samples × 30 features) as benign or malignant. We compare eight classical
classifiers and test whether PCA or feature selection can keep performance with far fewer
inputs.

**Full write-up with all results: [`REPORT.md`](REPORT.md)**

## Headline results

| | CV F1 (5-fold) | Test accuracy | Test malignant recall |
|---|---|---|---|
| RBF SVM, all 30 features | 0.976 | 99.1 % | 0.976 (1 miss) |
| Logistic Regression, all 30 | 0.967 | 99.1 % | 0.976 (1 miss) |
| RBF SVM, 10 PCA components (selected final model) | 0.976 | 98.2 % | 0.952 (2 misses) |
| Logistic Regression, 5 L1-selected features | 0.965 | 96.5 % | 0.952 (2 misses) |
| Final model with high-sensitivity threshold | – | 93.0 % | **1.000 (0 misses, 8 false alarms)** |

* 10 PCs (95 % variance) match all 30 features. 5 well-chosen original features nearly do.
* Univariate (mutual-information) selection fails because the features are highly redundant.
  RFE and L1 selection work.
* Every algorithm (LR, GNB, KNN, SMO-based SVM, CART, Random Forest, PCA, RFE) is also
  **implemented from scratch in NumPy** (`src/scratch.py`) and matches scikit-learn
  (notebook 08). That notebook also has a runtime benchmark, nested CV and a permutation test.
* The most important measurements are worst radius, perimeter and area, worst and mean
  concave points, and worst texture.

## Layout

```
data/wdbc.data                 raw UCI file (id, diagnosis, 30 features)
src/common.py                  shared protocol: data loading, fixed split, CV, metrics, result I/O
src/models.py                  pipeline factory + hyper-parameter grids
src/feature_selectors.py       L1-path selector + selector factory
src/scratch.py                 from-scratch NumPy implementations of every algorithm
src/build_notebooks.py         builds & executes notebooks from notebooks/_source/*.py
notebooks/
  01_eda_baselines.ipynb       Member 1: cleaning, EDA, protocol, majority baseline + LDA
  02_logreg_naive_bayes.ipynb  Member 2: Logistic Regression, Gaussian NB
  03_knn_svm.ipynb             Member 3: KNN, SVM
  04_trees.ipynb               Member 4: Decision Tree, Random Forest, feature importance
  05_pca.ipynb                 Member 5: PCA (2/5/10/95 %) + component curve
  06_feature_selection.ipynb   Member 6: SelectKBest-MI, RFE, L1 path; stability; L1 comparison
  07_final_evaluation.ipynb    Member 1: test-set evaluation, comparisons, error analysis, conclusions
  08_from_scratch.ipynb        Significant computation part: own implementations vs sklearn, runtime, nested CV, permutation test
  _source/                     editable source of each notebook (# %% cell format)
results/                       CSV/JSON outputs (CV tables, tuned params, test results)
figures/                       all plots (PNG)
```

## Running

```bash
pip install -r requirements.txt
python src/build_notebooks.py       # all notebooks (~20 min; 06 and 08 are the slowest)
python src/build_notebooks.py 05    # just one
```

You can also open the `.ipynb` files in Jupyter and run them in order (01 → 08). Notebooks
05–08 read `results/best_params.json`, which notebooks 02–04 write. The held-out test set
is used for evaluation only in notebook 07. Notebook 08 only checks that the scratch and
scikit-learn predictions on it agree.
