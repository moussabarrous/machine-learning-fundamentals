import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def plot_dataset(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="x")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, marker="x", label="training")
    plt.scatter(x_cv, y_cv, marker="o", label="cross validation")
    plt.scatter(x_test, y_test, marker="^", label="test")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def plot_train_cv_mses(degrees, train_mses, cv_mses, title):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_mses, marker="o", label="training MSE")
    plt.plot(degrees, cv_mses, marker="o", label="CV MSE")
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def plot_bc_dataset(x, y, title):
    y = np.asarray(y).reshape(-1)
    pos = y == 1
    neg = y == 0
    plt.figure(figsize=(10, 6))
    plt.scatter(x[pos, 0], x[pos, 1], marker="x", label="y=1")
    plt.scatter(x[neg, 0], x[neg, 1], marker="o", facecolors="none", label="y=0")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

def build_models():
    tf.random.set_seed(20)
    model_1 = Sequential([
        Dense(25, activation="relu"),
        Dense(15, activation="relu"),
        Dense(1, activation="linear"),
    ], name="model_1")

    model_2 = Sequential([
        Dense(20, activation="relu"),
        Dense(12, activation="relu"),
        Dense(12, activation="relu"),
        Dense(20, activation="relu"),
        Dense(1, activation="linear"),
    ], name="model_2")

    model_3 = Sequential([
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(4, activation="relu"),
        Dense(12, activation="relu"),
        Dense(1, activation="linear"),
    ], name="model_3")

    return [model_1, model_2, model_3]
