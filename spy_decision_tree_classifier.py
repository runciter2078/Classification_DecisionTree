#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPY Decision Tree Classifier — Exploratory / Colab-friendly pipeline
======================================================================

Loads a single dataset, explores the target class distribution, and
runs two complementary Decision Tree analyses: one using the full
feature set and one using a reduced, pre-selected feature subset.
Optimal tree depth is chosen via time-aware cross-validation in both
cases.
"""

from __future__ import annotations

import argparse
import io
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
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spy_decision_tree_colab")

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

TARGET_COLUMN = "CLASIFICADOR"
FULL_FEATURE_COLUMNS = [
    "2", "42", "45", "48", "68", "75", "88", "139", "171",
    "179", "187", "218", "221", "223", "231", "237", "FECHA.month",
]
SELECTED_FEATURE_COLUMNS = ["45", "75", "171"]  # subconjunto de FULL_FEATURE_COLUMNS
USE_COLUMNS = [TARGET_COLUMN] + FULL_FEATURE_COLUMNS

FIXED_PARAMS = {
    "criterion": "entropy",
    "min_samples_split": 65,
    "min_samples_leaf": 20,
    "class_weight": "balanced",  # antes: {0: 3.28}, un valor fijo no reutilizable
}

@dataclass
class ExploreConfig:
    data_path: Optional[str] = None
    use_colab_upload: bool = False
    train_ratio: float = 0.75
    cv_splits: int = 10
    output_dir: Path = Path("output")
    random_state: int = 8

# ----------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------

def load_data(config: ExploreConfig) -> pd.DataFrame:
    """
    Carga el dataset una única vez con la unión de columnas necesarias
    para ambos análisis (completo y reducido), evitando subir/leer el
    CSV dos veces como en la versión original.
    """
    if config.use_colab_upload:
        try:
            from google.colab import files  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "--colab fue especificado pero google.colab no está disponible."
            ) from exc
        uploaded = files.upload()
        filename = next(iter(uploaded))
        logger.info("Archivo subido: %s (%d bytes)", filename, len(uploaded[filename]))
        df = pd.read_csv(io.StringIO(uploaded[filename].decode("utf-8")), sep=",", usecols=USE_COLUMNS)
    else:
        if config.data_path is None:
            raise ValueError("Debes indicar --data-path o usar --colab.")
        path = Path(config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        df = pd.read_csv(path, sep=",", usecols=USE_COLUMNS)
        logger.info("Dataset cargado desde %s: %s filas x %s columnas", path, *df.shape)
    return df

def explore_data(df: pd.DataFrame, output_dir: Path) -> None:
    """Distribución de clases: log + gráfico guardado en disco (sin plt.show())."""
    counts = df[TARGET_COLUMN].value_counts()
    logger.info("Distribución de clases:\n%s", counts.to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=TARGET_COLUMN, data=df, ax=ax)
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

# ----------------------------------------------------------------------
# Split cronológico
# ----------------------------------------------------------------------

def split_dataset(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(df) * train_ratio)
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    logger.info("Train: %d filas | Test: %d filas", len(train), len(test))
    return train, test

# ----------------------------------------------------------------------
# Selección de max_depth por validación cruzada temporal
# ----------------------------------------------------------------------

def select_optimal_max_depth(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    fixed_params: dict,
    cv_splits: int,
    random_state: int,
) -> tuple[int, pd.DataFrame]:
    """
    TimeSeriesSplit en vez de KFold: KFold sin shuffle sigue entrenando,
    para un fold intermedio, con folds posteriores a él. Además, el
    max_depth óptimo se selecciona automáticamente maximizando la
    accuracy media, en vez de fijar a mano un valor de ejemplo.
    """
    cv = TimeSeriesSplit(n_splits=cv_splits)
    depth_range = range(1, len(feature_columns) + 1)
    accuracies = []

    x = train_df[feature_columns]
    y = train_df[TARGET_COLUMN]

    for depth in depth_range:
        params = {**fixed_params, "max_depth": depth, "random_state": random_state}
        fold_scores = []
        for train_idx, valid_idx in cv.split(x):
            clf = tree.DecisionTreeClassifier(**params)
            clf.fit(x.iloc[train_idx], y.iloc[train_idx])
            fold_scores.append(clf.score(x.iloc[valid_idx], y.iloc[valid_idx]))
        accuracies.append(np.mean(fold_scores))

    results_df = pd.DataFrame({"max_depth": list(depth_range), "avg_accuracy": accuracies})
    logger.info("Resultados de validación cruzada por max_depth:\n%s", results_df.to_string(index=False))

    optimal_depth = int(results_df.loc[results_df["avg_accuracy"].idxmax(), "max_depth"])
    logger.info("max_depth óptimo seleccionado automáticamente: %d", optimal_depth)
    return optimal_depth, results_df

# ----------------------------------------------------------------------
# Entrenamiento, evaluación y visualización
# ----------------------------------------------------------------------

def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    fixed_params: dict,
    max_depth: int,
    output_dir: Path,
    label: str,
    random_state: int,
) -> tree.DecisionTreeClassifier:
    x_train, y_train = train_df[feature_columns], train_df[TARGET_COLUMN]
    x_test, y_test = test_df[feature_columns], test_df[TARGET_COLUMN]

    params = {**fixed_params, "max_depth": max_depth, "random_state": random_state}
    clf = tree.DecisionTreeClassifier(**params)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    report_txt = classification_report(y_test, preds)
    logger.info("[%s] Classification report:\n%s", label, report_txt)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}_classification_report.txt").write_text(report_txt, encoding="utf-8")

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {label}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    _analyze_feature_importance(clf, x_test, y_test, feature_columns, output_dir, label, random_state)
    _visualize_tree(clf, feature_columns, output_dir, label)

    joblib.dump(clf, output_dir / f"{label}_decision_tree_model.joblib")
    with open(output_dir / f"{label}_params.json", "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, default=str)

    return clf

def _analyze_feature_importance(
    clf: tree.DecisionTreeClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    output_dir: Path,
    label: str,
    random_state: int,
) -> None:
    impurity_imp = pd.Series(clf.feature_importances_, index=features, name="impurity")
    perm_result = permutation_importance(
        clf, x_test, y_test, n_repeats=30, random_state=random_state, n_jobs=-1
    )
    perm_imp = pd.Series(perm_result.importances_mean, index=features, name="permutation")
    importance_df = pd.concat([impurity_imp, perm_imp], axis=1).sort_values(
        "permutation", ascending=False
    )
    logger.info("[%s] Importancia de variables:\n%s", label, importance_df.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    importance_df["permutation"].sort_values().plot.barh(ax=ax, color="#2E86AB")
    ax.set_xlabel("Importancia por permutación")
    ax.set_title(f"Importancia de variables — {label}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_feature_importance.png", dpi=150)
    plt.close(fig)

def _visualize_tree(
    clf: tree.DecisionTreeClassifier, features: list[str], output_dir: Path, label: str
) -> None:
    """
    sklearn.tree.plot_tree en vez de pydot/Graphviz: elimina una
    dependencia externa frágil de instalar y mantiene todo dentro del
    stack de matplotlib.
    """
    fig, ax = plt.subplots(figsize=(20, 12))
    tree.plot_tree(
        clf,
        feature_names=features,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_decision_tree.png", dpi=150)
    plt.close(fig)
    logger.info("[%s] Árbol visualizado y guardado.", label)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> ExploreConfig:
    parser = argparse.ArgumentParser(
        description="Análisis exploratorio con Decision Tree (features completas y reducidas) para SPY."
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--random-state", type=int, default=8)
    args = parser.parse_args()
    return ExploreConfig(
        data_path=args.data_path,
        use_colab_upload=args.colab,
        train_ratio=args.train_ratio,
        cv_splits=args.cv_splits,
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
    )

def main() -> None:
    config = parse_args()
    df = load_data(config)
    explore_data(df, config.output_dir)

    train_df, test_df = split_dataset(df, config.train_ratio)

    # ------------------------ Full feature set ------------------------
    depth_full, _ = select_optimal_max_depth(
        train_df, FULL_FEATURE_COLUMNS, FIXED_PARAMS, config.cv_splits, config.random_state
    )
    train_and_evaluate(
        train_df, test_df, FULL_FEATURE_COLUMNS, FIXED_PARAMS, depth_full,
        config.output_dir, label="full", random_state=config.random_state,
    )

    # ---------------------- Selected feature set -----------------------
    depth_sel, _ = select_optimal_max_depth(
        train_df, SELECTED_FEATURE_COLUMNS, FIXED_PARAMS, config.cv_splits, config.random_state
    )
    train_and_evaluate(
        train_df, test_df, SELECTED_FEATURE_COLUMNS, FIXED_PARAMS, depth_sel,
        config.output_dir, label="selected", random_state=config.random_state,
    )

if __name__ == "__main__":
    main()
