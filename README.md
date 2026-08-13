# SPY Decision Tree Classifier

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.3-orange)
![pandas](https://img.shields.io/badge/pandas-%E2%9C%93-150458)
![numpy](https://img.shields.io/badge/numpy-%E2%9C%93-013243)
![License](https://img.shields.io/badge/license-GPL--3.0-green)

This project implements a Decision Tree classifier in Python to predict positive entry days for the SPY ETF. The repository contains two complementary scripts.

> Disclaimer: this project is for educational and research purposes only. It does not constitute financial or investment advice, and past predictive performance does not guarantee future results.

## Repository structure

- `decision_tree_spy.py` — Hyperparameter search pipeline: performs a randomized search with time-aware cross-validation, trains the final model directly from the best estimator found, evaluates it, and visualizes the resulting tree. Loads data from a local CSV file (`SPYV3.csv` by default).
- `spy_decision_tree_classifier.py` — Exploratory pipeline (Colab-friendly): explores the target class distribution, and runs two Decision Tree analyses — one with the full feature set and one with a reduced, pre-selected subset — automatically selecting the optimal tree depth via cross-validation in both cases.
- `README.md` — This file.
- `LICENSE` — Project license (GPL-3.0).

## Methodology

- Chronological train/test split (no shuffling), consistent across both scripts.
- TimeSeriesSplit cross-validation instead of standard K-Fold, both for the randomized hyperparameter search and for the max-depth selection loop, to avoid training on folds that occur after the validation fold.
- The final model in `decision_tree_spy.py` is taken directly from the search's `best_estimator_`; the original script computed a search but then trained the final tree with separate, hand-picked hyperparameters that were never actually connected to the search results.
- Optimal `max_depth` in `spy_decision_tree_classifier.py` is chosen automatically as the depth that maximizes average cross-validated accuracy, instead of being hardcoded from a manual reading of the results.
- `class_weight="balanced"` replaces the original fixed weight (`{0: 3.28}`), which was computed for one specific dataset and would not generalize to a different class distribution.
- Tree visualization uses `sklearn.tree.plot_tree` (matplotlib) instead of `pydot`/Graphviz, removing an external dependency that is frequently a source of installation issues.
- Two feature-importance methods are reported: impurity-based (fast, but biased towards high-cardinality features) and permutation-based (computed on held-out data, more reliable).
- `spy_decision_tree_classifier.py` loads the dataset once, using the union of columns needed for both analyses, instead of uploading and reading the CSV twice.

## Requirements

- Python 3.9+

```text
pandas
numpy
scikit-learn>=1.3
scipy
matplotlib
seaborn
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

`google.colab` is only required for `spy_decision_tree_classifier.py` when run with the `--colab` flag inside a Google Colab notebook. `pydot`/Graphviz are no longer required.

## Data format

Both scripts expect a CSV file with a binary target column named `CLASIFICADOR`.

- `decision_tree_spy.py` uses these feature columns:

```text
1, 31, 42, 46, 47, 48, 60, 68, 76, 77, 93, 171, 173, 191, 221, 225, 237, FECHA.month
```

- `spy_decision_tree_classifier.py` uses a full feature set for its first analysis and a reduced subset for its second:

```text
Full:     2, 42, 45, 48, 68, 75, 88, 139, 171, 179, 187, 218, 221, 223, 231, 237, FECHA.month
Selected: 45, 75, 171
```

Adjust the column constants at the top of each script if you use a different dataset.

## Usage

### decision_tree_spy.py

```bash
python decision_tree_spy.py --data-path SPYV3.csv
```

Optional arguments:

```text
--train-ratio      Proportion of data used for training (default: 0.80)
--n-iter           RandomizedSearchCV iterations (default: 500)
--cv-splits        Number of TimeSeriesSplit folds (default: 5)
--plot-max-depth   Visual depth limit for the tree plot only (default: full tree)
--random-state     Random seed (default: 8)
--output-dir       Output directory for artifacts (default: output)
```

Generated artifacts (under `output/`):

```text
decision_tree_model.joblib     Trained model
best_params.json               Best hyperparameters found
classification_report.txt      Precision / recall / F1 per class
confusion_matrix.png           Confusion matrix heatmap
feature_importance.png         Permutation importance plot
decision_tree.png              Tree visualization
```

### spy_decision_tree_classifier.py

Local execution:

```bash
python spy_decision_tree_classifier.py --data-path SPYV3.csv
```

Inside Google Colab, with interactive upload:

```bash
python spy_decision_tree_classifier.py --colab
```

Optional arguments: `--train-ratio`, `--cv-splits`, `--output-dir` (same defaults as above). Artifacts are saved under `output/` with a `full_` or `selected_` prefix depending on the analysis (e.g. `full_decision_tree.png`, `selected_confusion_matrix.png`), plus a shared `class_distribution.png`.

## Notes and limitations

- Missing values are not imputed automatically.
- The dataset is not included in this repository.
- `n_iter=500` in `decision_tree_spy.py` is a practical default; the original script used 32768, which can take a very long time. Increase `--n-iter` if you have the computational budget.

## License

Distributed under the [GNU General Public License v3.0](LICENSE).
