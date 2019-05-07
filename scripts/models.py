from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer


GAUSSIAN = "GNB"
DECISION_TREE = "DT"
RANDOM_FOREST = "RF"
KNN = "KNN"
SCORE = "SCORE"
DECIMALS = 4

labels = [GAUSSIAN, DECISION_TREE]


def generate_all_models(knn_params, forest_params):
    models = []
    models.append(create_gaussian_model())
    models.append(create_decision_tree())
    for k in knn_params:
        models.append(create_parametered_model(k, KNN, KNeighborsClassifier))
    for f in forest_params:
        models.append(create_parametered_model(
            f, RANDOM_FOREST, RandomForestClassifier))
    return models


def create_gaussian_model():
    return [GaussianNB(), GAUSSIAN]


def create_decision_tree():
    return [DecisionTreeClassifier(), DECISION_TREE]


def create_parametered_model(param, label, model_type):
    label = f"{label}-{param}"
    labels.append(label)
    return [model_type(param), label]


def run_experiment(models, x_data, y_data, cv_range, t_size=0.33):
    results = {}
    results.update([(key, {}) for key in labels])

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=t_size, random_state=1)

    apply_models(models, results, x_train, x_test, y_train, y_test)
    run_all_models_with_cv(models, results, x_data, y_data, cv_range)
    return results


def apply_models(models, results, x_train, x_test, y_train, y_test):
    for model in models:
        print(f"Applying {model[1]} model")
        model[0].fit(x_train, y_train)
        # Tras entrenar el modelo pasamos al proceso de validacion
        y_predicted = model[0].predict(x_test)
        score = round(accuracy_score(y_test, y_predicted), DECIMALS)
        model_score = round(model[0].score(x_test, y_test), DECIMALS)
        train_score = round(model[0].score(x_train, y_train), DECIMALS)
        print(f"Report: \n{classification_report(y_test, y_predicted)}")
        results[model[1]][SCORE] = score


def run_all_models_with_cv(models, results, x_data, y_data, cv_range):
    for cv in cv_range:
        for model in models:
            print(f"Applying model: {model[1]} with CV: {cv}")
            model_mean = cross_val_score(
                model[0], x_data, y_data, cv=cv).mean()
            model_mean = round(model_mean, DECIMALS)
            results[model[1]][cv] = model_mean
