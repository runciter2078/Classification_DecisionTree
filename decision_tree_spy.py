#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPY Decision Tree Classifier — Hyperparameter Search Pipeline
================================================================

Loads a local CSV dataset, performs a randomized hyperparameter search
for a Decision Tree classifier with time-aware cross-validation, trains
the final model directly from the search's best estimator, evaluates
it, and visualizes the resulting tree.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint as sp_randint
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, precision_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("decision_tree_spy")

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

FEATURE_COLUMNS = [
    "1", "31", "42", "46", "47", "48", "60", "68", "76", "77",
    "93", "171", "173", "191", "221", "225", "237", "FECHA.month",
]
TARGET_COLUMN = "CLASIFICADOR"
USE_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS


@dataclass
class PipelineConfig:
    data_path: str = "SPYV3.csv"
    train_ratio: float = 0.80
    n_iter_search: int = 500
    cv_splits: int = 5
    output_dir: Path = Path("output")
    plot_max_depth: Optional[int] = None
    random_state: int = 8

# ----------------------------------------------------------------------
# Carga y split
# ----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    df = pd.read_csv(file_path, sep=",", usecols=USE_COLUMNS)
    logger.info("Dataset cargado desde %s: %s filas x %s columnas", file_path, *df.shape)
    return df


def split_dataset(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cronológico, sin shuffle: el test contiene las observaciones más recientes."""
    n_train = int(len(df) * train_ratio)
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    logger.info("Train: %d filas | Test: %d filas (ratio=%.2f)", len(train), len(test), train_ratio)
    return train, test

# ----------------------------------------------------------------------
# Búsqueda de hiperparámetros
# ----------------------------------------------------------------------

def build_param_distributions() -> dict:
    return {
        "max_depth": list(range(2, 19)) + [None],
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": sp_randint(2, 105),
        "min_samples_leaf": sp_randint(1, 105),
        "min_weight_fraction_leaf": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "max_leaf_nodes": list(range(2, 21)) + [None],
        "splitter": ["best", "random"],
        "class_weight": ["balanced", None],
        "criterion": ["gini", "entropy"],
    }

def _log_top_results(cv_results: dict, n_top: int = 3) -> None:
    for rank in range(1, n_top + 1):
        candidates = np.flatnonzero(cv_results["rank_test_score"] == rank)
        for candidate in candidates:
            logger.info(
                "Rank %d | score medio: %.3f (std %.3f) | params: %s",
                rank,
                cv_results["mean_test_score"][candidate],
                cv_results["std_test_score"][candidate],
                cv_results["params"][candidate],
            )

def hyperparameter_search(
    x_train: pd.DataFrame, y_train: pd.Series, config: PipelineConfig
) -> RandomizedSearchCV:
    """
    Validación cruzada temporal (TimeSeriesSplit) en vez del K-Fold por
    defecto, para no filtrar información futura durante la búsqueda.
    refit=True (por defecto) reentrena automáticamente el mejor
    estimador sobre todo x_train al finalizar.
    """
    clf = tree.DecisionTreeClassifier(random_state=config.random_state)
    scorer = make_scorer(precision_score, average="binary", zero_division=0)
    cv = TimeSeriesSplit(n_splits=config.cv_splits)

    search = RandomizedSearchCV(
        clf,
        scoring=scorer,
        param_distributions=build_param_distributions(),
        n_iter=config.n_iter_search,
        cv=cv,
        n_jobs=-1,
        random_state=config.random_state,
        verbose=1,
    )
    search.fit(x_train, y_train)
    _log_top_results(search.cv_results_, n_top=3)
    logger.info("Mejores hiperparámetros: %s", search.best_params_)
    return search

# ----------------------------------------------------------------------
# Evaluación, importancia y visualización
# ----------------------------------------------------------------------

def evaluate_model(
    clf: tree.DecisionTreeClassifier, x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path
) -> None:
    preds = clf.predict(x_test)
    report_txt = classification_report(y_test, preds)
    logger.info("Classification report:\n%s", report_txt)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

def analyze_feature_importance(
    clf: tree.DecisionTreeClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    output_dir: Path,
    random_state: int,
) -> pd.DataFrame:
    impurity_imp = pd.Series(clf.feature_importances_, index=features, name="impurity")
    perm_result = permutation_importance(
        clf, x_test, y_test, n_repeats=30, random_state=random_state, n_jobs=-1
    )
    perm_imp = pd.Series(perm_result.importances_mean, index=features, name="permutation")
    importance_df = pd.concat([impurity_imp, perm_imp], axis=1).sort_values(
        "permutation", ascending=False
    )
    logger.info("Importancia de variables:\n%s", importance_df.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    importance_df["permutation"].sort_values().plot.barh(ax=ax, color="#2E86AB")
    ax.set_xlabel("Importancia por permutación")
    ax.set_title("Importancia de variables (test set)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    return importance_df

def visualize_decision_tree(
    clf: tree.DecisionTreeClassifier, features: list[str], output_dir: Path, max_depth: Optional[int]
) -> None:
    """
    sklearn.tree.plot_tree en vez de pydot/Graphviz: sin dependencias
    externas y con el mismo stack de matplotlib que el resto del pipeline.
    `max_depth` aquí es solo visual (--plot-max-depth); no recorta el
    árbol entrenado.
    """
    fig, ax = plt.subplots(figsize=(20, 12))
    tree.plot_tree(
        clf,
        feature_names=features,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree.png", dpi=150)
    plt.close(fig)
    logger.info("Árbol de decisión visualizado y guardado.")


def save_artifacts(clf: tree.DecisionTreeClassifier, best_params: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_dir / "decision_tree_model.joblib")
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2, default=str)
    logger.info("Modelo y parámetros guardados en %s", output_dir)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros y entrenamiento de un Decision Tree para SPY."
    )
    parser.add_argument("--data-path", type=str, default="SPYV3.csv")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument(
        "--plot-max-depth", type=int, default=None,
        help="Profundidad máxima mostrada en la visualización del árbol (no afecta al entrenamiento).",
    )
    parser.add_argument("--random-state", type=int, default=8)
    args = parser.parse_args()
    return PipelineConfig(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        n_iter_search=args.n_iter,
        cv_splits=args.cv_splits,
        output_dir=Path(args.output_dir),
        plot_max_depth=args.plot_max_depth,
        random_state=args.random_state,
    )

def main() -> None:
    config = parse_args()
    df = load_data(config.data_path)
    train, test = split_dataset(df, config.train_ratio)

    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    search = hyperparameter_search(x_train, y_train, config)
    clf = search.best_estimator_  # ya reentrenado sobre x_train completo

    evaluate_model(clf, x_test, y_test, config.output_dir)
    analyze_feature_importance(clf, x_test, y_test, FEATURE_COLUMNS, config.output_dir, config.random_state)
    visualize_decision_tree(clf, FEATURE_COLUMNS, config.output_dir, config.plot_max_depth)
    save_artifacts(clf, search.best_params_, config.output_dir)

if __name__ == "__main__":
    main()
