
import tensorflow as tf
import numpy as np
import math
from generate_dataset import read_data, create_dataset
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.DEBUG)


if __name__ == "__main__":
    # Hay que normalizar los datos
    x_data, y_data, dataset = create_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=1)

    print(f"Train set before normalize:\n{x_train}")
    normalizer = Normalizer()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)
    print(f"Train set after normalize:\n{x_train}")
    print(f"X shape: {x_train.shape}, Y shape: {y_train.shape}")
    # Creamos el modelo
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(9, )),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=10)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Accuracy: {test_accuracy} - Loss: {test_loss}")
