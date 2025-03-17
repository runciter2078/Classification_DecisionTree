# SPY Decision Tree Classifier

This project implements a Decision Tree classifier in Python to predict positive entry days for the SPY stock index. The repository contains two main scripts:

1. **decision_tree_spy.py**  
   This script performs hyperparameter tuning using RandomizedSearchCV, trains a Decision Tree classifier on a dataset loaded from a CSV file (`SPYV3.csv`), evaluates the model using classification reports and confusion matrices, and visualizes the decision tree.

2. **spy_decision_tree_classifier.py**  
   Designed for Google Colab, this script allows users to upload a CSV file and performs data exploration, splits the dataset into training and testing sets, trains a Decision Tree classifier, and determines the optimal tree depth using KFold cross-validation. It also includes an analysis with a reduced set of features based on the classifier's feature importance.

## Features

- **Hyperparameter Tuning:**  
  Utilizes `RandomizedSearchCV` for exploring various hyperparameters for the Decision Tree classifier.
  
- **Model Evaluation:**  
  Generates classification reports, confusion matrices, and displays feature importances.

- **Decision Tree Visualization:**  
  Exports and visualizes the decision tree using `pydot`.

- **Data Exploration:**  
  Includes data exploration and visualization (e.g., class distribution, histograms) for both full and reduced feature sets.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - scipy
  - pydot
  - matplotlib
  - seaborn
- For the Colab version: access to `google.colab` for file uploads

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/runciter2078/Classification_DecisionTree.git
   ```

2. *(Optional)* Rename the repository folder to `SPY_DecisionTree_Classifier` for clarity.

3. Navigate to the project directory:

   ```bash
   cd Classification_DecisionTree
   ```

## Usage

### decision_tree_spy.py

1. Place the CSV file `SPYV3.csv` in the project directory.
2. Run the script:

   ```bash
   python decision_tree_spy.py
   ```

### spy_decision_tree_classifier.py (Google Colab)

1. Upload your CSV file when prompted.
2. Run the notebook cells sequentially to perform data exploration, model training, and visualization.

## Notes

- **Hyperparameter Search:**  
  The hyperparameter search in `decision_tree_spy.py` uses a high number of iterations (32768) by default, which may require considerable time. Adjust `n_iter_search` if necessary.

- **Optimal Depth Selection:**  
  The optimal `max_depth` values used in the scripts (e.g., 2 for full features, 4 for selected features) are examples. Use the cross-validation results to choose appropriate values for your dataset.

## License

This project is distributed under the [MIT License](LICENSE)..
