import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


UNCORRELATED = 0
WEAKLY = 1
STRONGLY = 2
INVERSE = 3
ALMOST_STRONG = 4
SUBSET_SUM = 5
UNCORRELATED_WITH_SIMILAR = 6
SPANNER_UNCORRELATED = 7
SPANNER_WEAKLY_CORRELATED = 8
SPANNER_STRONGLY_CORRELATED = 9
MULTIPLE_STRONGLY_CORRELATED = 10
PROFIT_CEILING = 11
CIRLE = 12
TARGET = "target"
NUM = "num"
CAPACITY = "capacity"
MIN_WEIGHT = "min_weight"
MIN_PROFIT = "min_profit"
MAX_WEIGHT = "max_weight"
MAX_PROFIT = "max_profit"
RELATION = "relation"
SUM_W = "sum_w"
SUM_P = "sum_p"

uncorrelated = "../00Uncorrelated/n00050/R01000"
weakly_correlated = "../01WeaklyCorrelated/n00050/R01000"
strongly_correlated = "../02StronglyCorrelated/n00050/R01000"
inverse_strongly_correlated = "../03InverseStronglyCorrelated/n00050/R01000"
almost_strongly_correlated = "../04AlmostStronglyCorrelated/n00050/R01000"
subset_sum_correlated = "../05SubsetSum/n00050/R01000"
uncorrelated_with_similar_w = "../06UncorrelatedWithSimilarWeights/n00050/R01000"
spanner_uncorrelated_p = "../07SpannerUncorrelated/n00050/R01000"
spanner_weakly_c = "../08SpannerWeaklyCorrelated/n00050/R01000"
spanner_strong = "../09SpannerStronglyCorrelated/n00050/R01000"
multiple_strong = "../10MultipleStronglyCorrelated/n00050/R01000"
profit = "../11ProfitCeiling/n00050/R01000"
circle_p = "../12Circle/n00050/R01000"


def read_data(paths, targets):
    list_of_data = []
    for path, target in zip(paths, targets):
        print(f"{path} --> {target}")
        # Leemos los ficheros disponibles
        filenames = [join(path, file)
                     for file in listdir(path) if isfile(join(path, file))]
        print(f"Found {len(filenames)} filenames")
        # Leemos los datos de cada fichero
        for filename in filenames:
            data = {}
            with open(filename) as file:
                data_file = [line.split()
                             for line in file if len(line.split()) >= 1]
                # Cogemos el numero de elementos
                data[NUM] = int(data_file[0][0])
                data[CAPACITY] = float(data_file[1][0])
                data[TARGET] = target
                weights_and_profits = np.asfarray(data_file[2:])
                minimums = np.amin(weights_and_profits, axis=0)
                maximums = np.amax(weights_and_profits, axis=0)
                data[MIN_WEIGHT] = minimums[0]
                data[MAX_WEIGHT] = maximums[0]
                data[MIN_PROFIT] = minimums[1]
                data[MAX_PROFIT] = maximums[1]
                sum_w_p = np.sum(weights_and_profits, axis=0)
                # Dividimos la suma de los beneficios entre la suma de los pesos
                data[RELATION] = sum_w_p[1] / sum_w_p[0]
                data[SUM_W] = sum_w_p[0]
                data[SUM_P] = sum_w_p[1]
            list_of_data.append(data)
    # Creamos un dataframe con la lista de elementos
    # En la lista, cada elemento es un diccionario que contiene
    # - Numero de elementos
    # - Capacidad de la mochila
    # - Relacion
    # - Target
    dataframe = pd.DataFrame(list_of_data)
    return dataframe


def create_dataset():
    # Primero leemos los datos y les asignamos las etiquetas
    paths = [uncorrelated, weakly_correlated, strongly_correlated, inverse_strongly_correlated,
             almost_strongly_correlated, subset_sum_correlated, uncorrelated_with_similar_w,
             spanner_uncorrelated_p, spanner_weakly_c, spanner_strong, multiple_strong,
             profit, circle_p]

    targets = [UNCORRELATED, WEAKLY, STRONGLY, INVERSE, ALMOST_STRONG, SUBSET_SUM,
               UNCORRELATED_WITH_SIMILAR, SPANNER_UNCORRELATED, SPANNER_WEAKLY_CORRELATED,
               SPANNER_STRONGLY_CORRELATED, MULTIPLE_STRONGLY_CORRELATED, PROFIT_CEILING, CIRLE]
    dataframe = read_data(paths, targets)
    x_data = dataframe.drop(TARGET, axis=1)
    y_data = dataframe[TARGET]
    return x_data, y_data, dataframe