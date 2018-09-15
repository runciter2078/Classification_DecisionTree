# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:21:48 2018

@author: Pablo Beret
"""

# Carga de datos de un CSV con delimitador coma y selección de variables.

import pandas as pd

spy = pd.read_csv("SPYV3.csv", sep=',', usecols=['CLASIFICADOR', '1','31',
                                                       '42','46','47','48',
                                                       '60','68','76','77',
                                                       '93','171','173','191',
                                                       '221','225','237', 
                                                       'FECHA.month'])
spy.head()

# División del conjunto en train y test

p_train = 0.80 # Porcentaje de train. Modificar para obtener diferentes conjuntos.

train = spy[:int((len(spy))*p_train)]
test = spy[int((len(spy))*p_train):]

print("Ejemplos usados para entrenar: ", len(train))
print("Ejemplos usados para test: ", len(test))
print("\n")

features = spy.columns[1:]
x_train = train[features]
y_train = train['CLASIFICADOR']

x_test = test[features]
y_test = test['CLASIFICADOR']

# Utilización de RandomizedSearchCV para busqueda de hiperparámetros

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn.metrics import precision_score, make_scorer
import warnings

warnings.filterwarnings('ignore') 

X, y = x_train, y_train # Datos de entrenamiento

clf = tree.DecisionTreeClassifier(random_state=8) # Construcción del clasificador

#Construcción de la métrica

metrica = make_scorer(precision_score, greater_is_better=True, average="binary") 
                     
def report(results, n_top=1): # Función para mostrar resultados
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Parámetros y distribuciones para muestrear
param_dist = {"max_depth": [18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,None], 
              "max_features": ['sqrt', 'log2', None],
              "min_samples_split": sp_randint(2, 105),
              "min_samples_leaf": sp_randint(1, 105),
              "min_weight_fraction_leaf": [0,0.05,0.10,0.15,0.20,0.25,0.30,
                                           0.35,0.40,0.45,0.50],
              "max_leaf_nodes": [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,None],
              "splitter": ['best', 'random'], 
              "class_weight":['balanced', None],
              "criterion": ["gini", "entropy"],
              #"random_state": sp_randint(1, 80)
              }

n_iter_search = 32768 # Ejecución
random_search = RandomizedSearchCV(clf, scoring= metrica, 
                                   param_distributions=param_dist, 
                                   n_iter=n_iter_search)
                                   

random_search.fit(X, y)
report(random_search.cv_results_)

# Creación del modelo Random Forest con los parámetros obtenidos

clf = tree.DecisionTreeClassifier(class_weight= None, criterion='entropy',
                                  max_depth = 8, max_features = 'sqrt',  
                                  max_leaf_nodes = 9, min_samples_leaf = 87,
                                  min_samples_split = 45, 
                                  min_weight_fraction_leaf=0.05, 
                                  splitter='best', random_state=8)
                                  
clf.fit(x_train, y_train) # Construcción del modelo

preds = clf.predict(x_test) # Test del modelo

# Visualización de resultados

from sklearn.metrics import classification_report
print("Árbol de decisión: \n" 
      +classification_report(y_true=test['CLASIFICADOR'], y_pred=preds))

# Matriz de confusión

print("Matriz de confusión:\n")
matriz = pd.crosstab(test['CLASIFICADOR'], preds, rownames=['actual'], colnames=['preds'])
print(matriz)

# Variables relevantes

print("Relevancia de variables:\n")
print(pd.DataFrame({'Indicador': features ,
              'Relevancia': clf.feature_importances_}),"\n")
print("Máxima relevancia DF :" , max(clf.feature_importances_), "\n")

# Visualización del Árbol de Decisión

from IPython.display import Image
import pydot
from sklearn.externals.six import StringIO

dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data, feature_names=list(spy.drop(['CLASIFICADOR'], axis=1)))
tree.export_graphviz(clf, out_file = dot_data, proportion = True,
                     feature_names=list(spy.drop(['CLASIFICADOR'], axis=1)), 
                     class_names = ['0','1'], rounded = True, filled = True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())












