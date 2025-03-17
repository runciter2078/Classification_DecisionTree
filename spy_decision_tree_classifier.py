#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPY Decision Tree Classifier for Google Colab

This script uploads a CSV dataset, performs data exploration, splits the dataset into
training and testing sets, trains a Decision Tree classifier, evaluates the model,
and visualizes the decision tree. It also performs cross-validation to find the optimal
max_depth and repeats the analysis using a reduced set of features.
"""

import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from io import StringIO
import pydot


def upload_data():
    """
    Upload a CSV file using Google Colab file upload.
    """
    uploaded = files.upload()
    for fn in uploaded.keys():
        print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
        return fn, io.StringIO(uploaded[fn].decode('utf-8'))
    return None, None


def load_dataset(file_io, columns):
    """
    Load dataset from the uploaded file with specified columns.
    """
    df = pd.read_csv(file_io, sep=',', usecols=columns)
    print("Dataset Head:")
    print(df.head())
    return df


def explore_data(df):
    """
    Explore the dataset by displaying class distribution and a count plot.
    """
    print("\nClass distribution:")
    print(df['CLASIFICADOR'].value_counts())
    plt.figure(figsize=(6, 4))
    sns.countplot(x='CLASIFICADOR', data=df)
    plt.title("Class Distribution")
    plt.show()


def split_dataset(df, train_ratio=0.75):
    """
    Split the dataset into training and testing sets.
    """
    n_train = int(len(df) * train_ratio)
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    print("Training examples:", len(train))
    print("Testing examples:", len(test))
    return train, test


def train_decision_tree(x_train, y_train, params):
    """
    Train a Decision Tree classifier using the given parameters.
    """
    clf = tree.DecisionTreeClassifier(**params)
    clf.fit(x_train, y_train)
    return clf


def cross_validate_max_depth(df, depth_range, fixed_params, n_splits=10):
    """
    Use KFold cross-validation to determine the optimal max_depth.
    """
    cv = KFold(n_splits=n_splits, shuffle=False)
    accuracies = []

    for depth in depth_range:
        fold_accuracies = []
        params = fixed_params.copy()
        params['max_depth'] = depth
        clf = tree.DecisionTreeClassifier(**params)

        for train_index, valid_index in cv.split(df):
            train_fold = df.iloc[train_index]
            valid_fold = df.iloc[valid_index]
            X_train_fold = train_fold.drop(['CLASIFICADOR'], axis=1)
            y_train_fold = train_fold['CLASIFICADOR']
            X_valid_fold = valid_fold.drop(['CLASIFICADOR'], axis=1)
            y_valid_fold = valid_fold['CLASIFICADOR']

            clf.fit(X_train_fold, y_train_fold)
            fold_accuracies.append(clf.score(X_valid_fold, y_valid_fold))

        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        accuracies.append(avg_accuracy)

    results_df = pd.DataFrame({'Max Depth': list(depth_range), 'Average Accuracy': accuracies})
    print("\nCross-validation results for max_depth:")
    print(results_df.to_string(index=False))
    return results_df


def visualize_decision_tree(model, feature_names):
    """
    Visualize the decision tree and save the output as a PNG file.
    """
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data, proportion=True,
                         feature_names=feature_names, class_names=['0', '1'],
                         rounded=True, filled=True)
    graphs = pydot.graph_from_dot_data(dot_data.getvalue())
    if graphs:
        png_str = graphs[0].create_png()
        with open("decision_tree_colab.png", "wb") as f:
            f.write(png_str)
        print("Decision tree visualization saved as 'decision_tree_colab.png'.")
    else:
        print("Failed to create decision tree visualization.")


def main():
    # ----------------------- First Analysis with Full Feature Set -----------------------
    # Upload data file
    filename, file_io = upload_data()
    if filename is None:
        print("No file uploaded.")
        return

    # Define full feature set columns
    full_columns = ['CLASIFICADOR', '2', '42', '45', '48', '68', '75',
                    '88', '139', '171', '179', '187', '218', '221', '223',
                    '231', '237', 'FECHA.month']
    df_full = load_dataset(file_io, full_columns)
    explore_data(df_full)

    # Split dataset
    train_df, test_df = split_dataset(df_full, train_ratio=0.75)
    features_full = df_full.columns[1:]
    x_train_full = train_df[features_full]
    y_train_full = train_df['CLASIFICADOR']
    x_test_full = test_df[features_full]
    y_test_full = test_df['CLASIFICADOR']

    # Train initial Decision Tree classifier without max_depth
    params_initial = {
        'criterion': 'entropy',
        'min_samples_split': 65,
        'min_samples_leaf': 20,
        'class_weight': {0: 3.28}
    }
    clf_initial = train_decision_tree(x_train_full, y_train_full, params_initial)

    # Evaluate initial model
    preds_initial = clf_initial.predict(x_test_full)
    print("\nInitial Decision Tree Classification Report:\n")
    print(classification_report(y_true=test_df['CLASIFICADOR'], y_pred=preds_initial))
    print("Confusion Matrix:")
    print(pd.crosstab(test_df['CLASIFICADOR'], preds_initial, rownames=['Actual'], colnames=['Predicted']))

    # Determine optimal max_depth using cross-validation
    max_depth_range = range(1, len(df_full.columns))
    cross_validate_max_depth(df_full, max_depth_range, params_initial, n_splits=10)

    # Train final model with an example optimal max_depth (e.g., 2)
    optimal_depth_full = 2  # Adjust based on cross-validation results
    params_final = params_initial.copy()
    params_final['max_depth'] = optimal_depth_full
    clf_final_full = train_decision_tree(x_train_full, y_train_full, params_final)

    preds_final_full = clf_final_full.predict(x_test_full)
    print("\nFinal Decision Tree Classification Report (Full Features):\n")
    print(classification_report(y_true=test_df['CLASIFICADOR'], y_pred=preds_final_full))
    print("Confusion Matrix:")
    print(pd.crosstab(test_df['CLASIFICADOR'], preds_final_full, rownames=['Actual'], colnames=['Predicted']))

    # Feature importance analysis
    print("\nFeature Importances (Full Features):")
    print(pd.DataFrame({'Feature': features_full, 'Importance': clf_final_full.feature_importances_}))
    print("Maximum Feature Importance:", clf_final_full.feature_importances_.max())

    # Visualize decision tree for full features
    visualize_decision_tree(clf_final_full, list(features_full))

    # ----------------------- Second Analysis with Selected Features -----------------------
    print("\nReloading data for selected features...")
    filename2, file_io2 = upload_data()
    if filename2 is None:
        print("No file uploaded for selected features.")
        return
    selected_columns = ['CLASIFICADOR', '45', '75', '171']
    df_selected = load_dataset(file_io2, selected_columns)

    # Plot histograms for selected features
    for col in selected_columns[1:]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df_selected[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.show()

    # Split dataset for selected features
    train_sel, test_sel = split_dataset(df_selected, train_ratio=0.75)
    features_sel = df_selected.columns[1:]
    x_train_sel = train_sel[features_sel]
    y_train_sel = train_sel['CLASIFICADOR']
    x_test_sel = test_sel[features_sel]
    y_test_sel = test_sel['CLASIFICADOR']

    # Determine optimal max_depth for selected features via cross-validation
    max_depth_range_sel = range(1, len(df_selected.columns))
    cross_validate_max_depth(df_selected, max_depth_range_sel, params_initial, n_splits=10)

    # Train final model for selected features with an example optimal max_depth (e.g., 4)
    optimal_depth_sel = 4  # Adjust based on cross-validation results
    params_final_sel = params_initial.copy()
    params_final_sel['max_depth'] = optimal_depth_sel
    clf_final_sel = train_decision_tree(x_train_sel, y_train_sel, params_final_sel)

    preds_final_sel = clf_final_sel.predict(x_test_sel)
    print("\nFinal Decision Tree Classification Report (Selected Features):\n")
    print(classification_report(y_true=test_sel['CLASIFICADOR'], y_pred=preds_final_sel))
    print("Confusion Matrix:")
    print(pd.crosstab(test_sel['CLASIFICADOR'], preds_final_sel, rownames=['Actual'], colnames=['Predicted']))

    # Feature importance analysis for selected features
    print("\nFeature Importances (Selected Features):")
    print(pd.DataFrame({'Feature': features_sel, 'Importance': clf_final_sel.feature_importances_}))
    print("Maximum Feature Importance:", clf_final_sel.feature_importances_.max())

    # Visualize decision tree for selected features
    visualize_decision_tree(clf_final_sel, list(features_sel))


if __name__ == '__main__':
    main()
