import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.preprocessing import Normalizer
from instances_types import *
sns.set()

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


# Recibe un diccionario de targets y paths
def read_data(paths):
    list_of_data = []
    for target in paths:
        for path in paths[target]:
            print(f"{target} --> {path}")
            # Leemos los ficheros disponibles
            filenames = [join(path, file)
                         for file in listdir(path) if isfile(join(path, file)) and file.endswith(".kp")]
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


def save_as_csv(dataset):
    dataset.to_csv("dataset.csv")


def plot_variables(dataset):
    sns.set(style="whitegrid")
    for column in dataset.columns.drop(["num"]):
        sns.boxplot(x=dataset[column])
        plt.show()


def load_paths():
    paths = {}
    paths[UNCORRELATED] = uncorrelated
    paths[WEAKLY] = weakly_correlated
    paths[STRONGLY] = strongly_correlated
    paths[INVERSE] = inverse_strongly
    paths[SUBSET_SUM] = subset_sum
    paths[UNCORRELATED_WITH_S_W] = uncorrelated_w_w
    paths[SPANNER_U] = spanner_u
    paths[SPANNER_W] = spanner_w
    paths[SPANNER_S] = spanner_s
    paths[MULTIPLE_S] = multiple
    paths[PROFIT_C] = profit
    paths[CIRCLE] = circle
    return paths


def create_dataset(normalize=True, shuff=True):
    # Primero leemos los datos y les asignamos las etiquetas
    paths = load_paths()
    dataframe = read_data(paths)

    # Normalizamos los datos
    if normalize is True:
        normalizer = Normalizer()
        data = normalizer.fit_transform(dataframe.drop(
            columns=["target"]), dataframe["target"])
        data = pd.DataFrame(data)
        data.columns = dataframe.columns.drop("target")
        data["target"] = dataframe["target"]
        dataframe = data
    if shuff is True:
        dataframe = shuffle(dataframe)

    y_data = dataframe[TARGET]
    x_data = dataframe.drop(TARGET, axis=1)
    save_as_csv(dataframe)
    # plot_variables(dataframe)
    return x_data, y_data, dataframe
