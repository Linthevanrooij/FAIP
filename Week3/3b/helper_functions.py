from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


def plot_dataset(X, y):
    plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')


def plot_decision_boundary(X, y, model):
    X_min = np.min(X[:, :], axis=0)
    X_max = np.max(X[:, :], axis=0)

    x0, x1 = np.meshgrid(
        np.linspace(X_min[0], X_max[0], 500).reshape(-1, 1),
        np.linspace(X_min[1], X_max[1], 200).reshape(-1, 1))

    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_new = model.predict(X_new)

    plot_dataset(X, y)

    zz = y_new.reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, levels=np.array([0.5]), colors='k')


def plot_decision_boundary_twostep(X, y, preprocessing, model):
    model_combined = Pipeline([('preprocessing', preprocessing), ('classifier', model)])
    plot_decision_boundary(X, y, model_combined)

def create_dataset(size, dimension, delta):
    #Size of the dataset
    n = (int(size / 2), int(size / 2))

    # Dimension of the dataset
    dim = dimension

    # Distance between the mean of the classes
    delta = delta

    # Prior probabilities per class
    p = [0.5, 0.5]  # make a list
    n = np.asarray(n)
    p = np.asarray(p)
    c = len(p)
    P = np.cumsum(p)
    P = np.concatenate((np.zeros(1), P))
    N = np.zeros(c, dtype=int)

    x = np.random.rand(np.sum(n))
    for i in range(0, c):
        N[i] = np.sum((x > P[i]) & (x < P[i + 1]))
    x0 = np.random.randn(N[0], dim)
    x1 = np.random.randn(N[1], dim)
    x1[:, 0] = x1[:, 0] + delta  # move data from class 1
    x_features = np.concatenate((x0, x1), axis=0)

    # labels
    lab = (0, 1)
    y_out = np.repeat(lab[0], N[0])
    for i in range(1, len(N)):
        y_out = np.concatenate((y_out, np.repeat(lab[i], N[i])))
    return x_features, y_out


def get_moon_data(size):
    X, y = make_moons(size, noise=0.2)
    return X, y

