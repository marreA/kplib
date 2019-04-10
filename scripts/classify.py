#!/usr/bin/python3

import pandas as pd
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sys
from pprint import pprint as pprint
import numpy as np
from os import listdir, walk
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


UNCORRELATED = 0
WEAKLY = 1
STRONGLY = 2
INVERSE = 3
ALMOST_STRONG = 4

uncorrelated = "00Uncorrelated/n00050/R01000"
weakly_correlated = "01WeaklyCorrelated/n00050/R01000"
strongly_correlated = "02StronglyCorrelated/n00050/R01000"
inverse_strongly_correlated = "03InverseStronglyCorrelated/n00050/R01000"
almost_strongly_correlated = "04AlmostStronglyCorrelated/n00050/R01000"


def read_data(path, target):
    # Leemos los ficheros disponibles
    #filenames = [join(dp, file) for dp, dn, fn in walk(path) for file in fn]
    filenames = [join(path, file)
                 for file in listdir(path) if isfile(join(path, file))]
    print(f"Found {len(filenames)} filenames in {path}")
    # Leemos los datos de cada fichero
    list_of_data = []
    for filename in filenames:
        data = {}
        with open(filename) as file:
            data_file = [line.split()
                         for line in file if len(line.split()) >= 1]
            # Cogemos el numero de elementos
            data["num"] = int(data_file[0][0])
            data["capacity"] = float(data_file[1][0])
            data["target"] = target
            weights_and_profits = np.asfarray(data_file[2:])
            sum_w_p = np.sum(weights_and_profits, axis=0)
            # Dividimos la suma de los beneficios entre la suma de los pesos
            data["relation"] = sum_w_p[1] / sum_w_p[0]
        list_of_data.append(data)
    # Creamos un dataframe con la lista de elementos
    # En la lista, cada elemento es un diccionario que contiene
    # - Numero de elementos
    # - Capacidad de la mochila
    # - Relacion
    # - Target
    dataframe = pd.DataFrame(list_of_data)
    return dataframe


# Primero leemos los datos y las etiquetas
uncorrelated_df = read_data(uncorrelated, UNCORRELATED)
weakly_df = read_data(weakly_correlated, WEAKLY)
strongly_df = read_data(strongly_correlated, STRONGLY)
inverse_df = read_data(inverse_strongly_correlated, INVERSE)
almost_df = read_data(almost_strongly_correlated, ALMOST_STRONG)

# Concatenamos en un mismo dataframe
dataframe = pd.concat([uncorrelated_df, weakly_df, strongly_df, inverse_df,
                       almost_df])

print(dataframe.head())
x_data = dataframe.drop("target", axis=1)
y_data = dataframe["target"]
print(f"X_data shape is {x_data.shape}\nY_data shape is {y_data.shape}")

# Realizamos la division en dos conjuntos: entrenamiento y testeo
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=1)
print(
    f"After splitting data we have\n - {x_train.shape} instances for training\n - {x_test.shape} instances for testing")
# Aplicamos un modelo Naive Bayes Gaussiano

model = GaussianNB()
model.fit(x_train, y_train)
y_model = model.predict(x_test)

print(f"Model accuracy {accuracy_score(y_test, y_model)}%")
# Mostramos la matriz de confunsion
matrix = confusion_matrix(y_test, y_model)
sns.heatmap(matrix, square=True, annot=True, cbar=False)
plt.xlabel("Predicted value")
plt.ylabel("True Value")
plt.show()
