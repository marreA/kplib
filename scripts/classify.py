#!/usr/bin/python3

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, validation_curve, cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

GAUSSIAN = "GNB"
DECISION_TREE = "DT"
RANDOM_FOREST = "RF"
KNN = "KNN"

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
                data["num"] = int(data_file[0][0])
                data["capacity"] = float(data_file[1][0])
                data["target"] = target
                weights_and_profits = np.asfarray(data_file[2:])
                sum_w_p = np.sum(weights_and_profits, axis=0)
                # Dividimos la suma de los beneficios entre la suma de los pesos
                data["relation"] = sum_w_p[1] / sum_w_p[0]
                data["sum_w"] = sum_w_p[0]
                data["sum_p"] = sum_w_p[1]
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
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(dataframe)
    x_data = dataframe.drop("target", axis=1)
    y_data = dataframe["target"]
    print(f"X_data shape is {x_data.shape}\nY_data shape is {y_data.shape}")
    return x_data, y_data, dataframe


def create_confusion_matrix(y_test, y_model, title):
    matrix = confusion_matrix(y_test, y_model)
    sns.heatmap(matrix, square=True, annot=True, cbar=False)
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.title(title)
    plt.show()


def apply_model(model, x_train, x_test, y_train, y_test, title):
    print(f"Applying {title} model")
    model.fit(x_train, y_train)
    # Tras entrenar el modelo pasamos al proceso de validacion
    y_predicted = model.predict(x_test)
    score = accuracy_score(y_test, y_predicted)
    model_score = model.score(x_test, y_test)
    train_score = model.score(x_train, y_train)
    print(
        f"- Accuracy score {score}%\n- Using model.score(): {model_score}%\n- Training score: {train_score}%")
    # create_confusion_matrix(y_test, y_predicted, title)
    print(
        f"Classificacion Report: \n{classification_report(y_test, y_predicted)}")
    return score, model_score, train_score


def parameter_setting(model_type, params, x, y, cv, results, model_label):
    for i in params:
        label = f"{model_label}-{i}"
        print(f"Applying {label} with cross-validation {cv}")
        model = model_type(i)
        validation = cross_val_score(model, x, y, cv=cv)
        mean = validation.mean()
        print(f"- Mean: {mean}")
        results[label][cv] = mean


def plot_train_vs_test_accuracy(n, train_scores, test_scores, labels):
    fig, axis = plt.subplots()
    ind = np.arange(n)
    width = 0.45
    train_rects = axis.bar(ind, train_scores, width, color="red")
    test_rects = axis.bar(ind + width, test_scores, width, color="green")
    axis.set_ylabel("Scores")
    axis.set_title("Scores by different models")
    axis.set_xticks(ind + width / 2)
    axis.set_xticklabels(labels)
    axis.legend((train_rects[0], test_rects[0]), ("Train", "Test"))
    plt.show()


if __name__ == "__main__":

    x_data, y_data, dataset = create_dataset()
    # Dividimos los datos en dos conjuntos, entrenamiento y testeo
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, random_state=1)

    # Numeros de estimadores para aplicar RandomForest
    n_estimators = [25, 50, 75, 100, 150, 200, 250]
    # Variamos el parametro K del KNN para comparar resultados
    k_parameters = range(2, 21)

    # Diccionario para almacenar los resultados de los diferentes modelos
    # con cada configuracion y tipo de validacion usada
    results = {}
    knn_labels = [f"{KNN}-{k}" for k in k_parameters]
    forest_labels = [f"{RANDOM_FOREST}-{i}" for i in n_estimators]
    labels = [GAUSSIAN, DECISION_TREE]
    labels += [i for i in knn_labels]
    labels += [i for i in forest_labels]

    results.update([(key, {}) for key in labels])

    # Instanciamos los modelos y los aplicamos
    gaussian_model = GaussianNB()
    decision_tree = DecisionTreeClassifier()

    guassian_score, _, guassian_train = apply_model(gaussian_model, x_train, x_test,
                                                    y_train, y_test, GAUSSIAN)
    tree_score, _, tree_train = apply_model(decision_tree, x_train, x_test,
                                            y_train, y_test, DECISION_TREE)

    # Creamos los arrays con los resultados para luego hacer las graficas
    train_scores = [guassian_train, tree_train]
    test_scores = [guassian_score, tree_score]
    results[GAUSSIAN]["SCORE"] = guassian_score
    results[DECISION_TREE]["SCORE"] = tree_score

    # Variamos el parametro K del KNN y aplicamos
    for k, label in zip(k_parameters, knn_labels):
        knn_model = KNeighborsClassifier(n_neighbors=k, n_jobs=5)
        score, model, train = apply_model(
            knn_model, x_train, x_test, y_train, y_test, label)
        test_scores.append(score)
        train_scores.append(train)
        results[label]["SCORE"] = score

    # Variamos el numero de estimadores en RandomForest
    for i, label in zip(n_estimators, forest_labels):
        forest = RandomForestClassifier(n_estimators=i, random_state=0)
        score, model, train = apply_model(
            forest, x_train, x_test, y_train, y_test, label)
        test_scores.append(score)
        train_scores.append(train)
        results[label]["SCORE"] = score

    # Mostramos una comparacion de resultados
    #plot_train_vs_test_accuracy(len(labels), train_scores, test_scores, labels)

    # Ahora probamos a usar validacion cruzada con los mismos modelos
    cv_range = range(2, 11)
    for cv in cv_range:
        gaussian_mean = cross_val_score(
            gaussian_model, x_data, y_data, cv=cv).mean()
        tree_mean = cross_val_score(
            decision_tree, x_data, y_data, cv=cv).mean()

        results[GAUSSIAN][cv] = gaussian_mean
        results[DECISION_TREE][cv] = tree_mean

        parameter_setting(KNeighborsClassifier, k_parameters,
                          x_data, y_data, cv, results, KNN)
        parameter_setting(RandomForestClassifier, n_estimators,
                          x_data, y_data, cv, results, RANDOM_FOREST)

    # Creamos un dataframe para mejorar la salida de resultados
    cv_dataframe = pd.DataFrame(results)
    cv_dataframe["Best"] = cv_dataframe.idxmax(axis=1)
    cv_dataframe["Max-Mean"] = cv_dataframe.max(axis=1)
    cv_dataframe.to_html("index.html")
