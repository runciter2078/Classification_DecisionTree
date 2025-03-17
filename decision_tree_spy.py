#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Tree Classifier for SPY Data

This script loads a CSV dataset, performs hyperparameter tuning using RandomizedSearchCV,
trains a Decision Tree classifier with chosen hyperparameters, evaluates the model,
and exports a visualization of the tree.
"""

import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn.metrics import precision_score, make_scorer, classification_report
from io import StringIO
import pydot
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load dataset from a CSV file using specific columns.
    """
    cols = ['CLASIFICADOR', '1', '31', '42', '46', '47', '48',
            '60', '68', '76', '77', '93', '171', '173', '191',
            '221', '225', '237', 'FECHA.month']
    df = pd.read_csv(filepath, sep=',', usecols=cols)
    print("Data Head:")
    print(df.head())
    return df


def split_dataset(df, train_ratio=0.80):
    """
    Split the dataset into training and testing sets.
    """
    n_train = int(len(df) * train_ratio)
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    print("Training examples:", len(train))
    print("Testing examples:", len(test))
    return train, test


def report(results, n_top=1):
    """
    Report the top n models from the hyperparameter search.
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {}".format(i))
            print("Mean validation score: {:.3f} (std: {:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {}".format(results['params'][candidate]))
            print("")


def perform_hyperparameter_search(X, y, n_iter_search=32768):
    """
    Perform RandomizedSearchCV for hyperparameter tuning of the Decision Tree.
    """
    clf = tree.DecisionTreeClassifier(random_state=8)
    scorer = make_scorer(precision_score, greater_is_better=True, average="binary")

    param_dist = {
        "max_depth": [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, None],
        "max_features": ['sqrt', 'log2', None],
        "min_samples_split": sp_randint(2, 105),
        "min_samples_leaf": sp_randint(1, 105),
        "min_weight_fraction_leaf": [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                     0.35, 0.40, 0.45, 0.50],
        "max_leaf_nodes": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, None],
        "splitter": ['best', 'random'],
        "class_weight": ['balanced', None],
        "criterion": ["gini", "entropy"],
    }

    random_search = RandomizedSearchCV(clf, scoring=scorer,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       random_state=8)
    random_search.fit(X, y)
    report(random_search.cv_results_)
    return random_search


def train_final_model(x_train, y_train):
    """
    Train the final Decision Tree classifier with the chosen hyperparameters.
    """
    final_clf = tree.DecisionTreeClassifier(
        criterion='entropy',
        max_depth=8,
        max_features='sqrt',
        max_leaf_nodes=9,
        min_samples_leaf=87,
        min_samples_split=45,
        min_weight_fraction_leaf=0.05,
        splitter='best',
        random_state=8
    )
    final_clf.fit(x_train, y_train)
    return final_clf


def visualize_decision_tree(model, feature_names):
    """
    Visualize the trained decision tree and save the output as a PNG file.
    """
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data, proportion=True,
                         feature_names=feature_names, class_names=['0', '1'],
                         rounded=True, filled=True)
    graphs = pydot.graph_from_dot_data(dot_data.getvalue())
    if graphs:
        png_str = graphs[0].create_png()
        with open("decision_tree.png", "wb") as f:
            f.write(png_str)
        print("Decision tree visualization saved as 'decision_tree.png'.")
    else:
        print("Failed to create decision tree visualization.")


def main():
    # Load dataset from CSV file
    filepath = "SPYV3.csv"
    df = load_data(filepath)

    # Split dataset into training and testing sets
    train_df, test_df = split_dataset(df, train_ratio=0.80)

    # Define features and target variable
    features = df.columns[1:]
    x_train = train_df[features]
    y_train = train_df['CLASIFICADOR']
    x_test = test_df[features]
    y_test = test_df['CLASIFICADOR']

    # Hyperparameter search (adjust n_iter_search if necessary)
    print("\nStarting hyperparameter search...")
    perform_hyperparameter_search(x_train, y_train)

    # Train final model with chosen hyperparameters
    final_model = train_final_model(x_train, y_train)

    # Evaluate the final model
    preds = final_model.predict(x_test)
    print("\nDecision Tree Classification Report:\n")
    print(classification_report(y_true=y_test, y_pred=preds))

    # Confusion matrix
    print("Confusion Matrix:")
    confusion = pd.crosstab(test_df['CLASIFICADOR'], preds, rownames=['Actual'], colnames=['Predicted'])
    print(confusion)

    # Display feature importances
    print("\nFeature Importances:")
    importances = pd.DataFrame({'Feature': features, 'Importance': final_model.feature_importances_})
    print(importances)
    print("\nMaximum Feature Importance:", final_model.feature_importances_.max())

    # Visualize decision tree
    visualize_decision_tree(final_model, list(features))


if __name__ == '__main__':
    main()
