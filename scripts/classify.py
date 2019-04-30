#!/usr/bin/python3

from generate_dataset import read_data, create_dataset
from models import *
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(color_codes=True)

GAUSSIAN = "GNB"
DECISION_TREE = "DT"
RANDOM_FOREST = "RF"
KNN = "KNN"
DECIMALS = 4


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
    score = round(accuracy_score(y_test, y_predicted), DECIMALS)
    model_score = round(model.score(x_test, y_test), DECIMALS)
    train_score = round(model.score(x_train, y_train), DECIMALS)
    print(
        f"- Accuracy score {score}%\n- Using model.score(): {model_score}%\n- Training score: {train_score}%")
    # create_confusion_matrix(y_test, y_predicted, title)
    print(
        f"Classificacion Report: \n{classification_report(y_test, y_predicted)}")
    return score, model_score, train_score


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

    # Numeros de estimadores para aplicar RandomForest
    n_estimators = [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250]
    # Variamos el parametro K del KNN para comparar resultados
    k_parameters = range(2, 21)
    cv_range = range(2, 11)

    models = generate_all_models(k_parameters, n_estimators)
    results = run_experiment(models, x_data, y_data, cv_range)

    # Creamos un dataframe para mejorar la salida de resultados
    cv_dataframe = pd.DataFrame(results)
    cv_dataframe["Best"] = cv_dataframe.idxmax(axis=1)
    cv_dataframe["Max-Mean"] = cv_dataframe.max(axis=1)
    cv_dataframe.to_html("index.html")
