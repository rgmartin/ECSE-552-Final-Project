<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ClusterParams:
    def __init__(self, x1_range, x2_range, num_points):
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.N = num_points

    def generate_cluster(self):
        x1_values = (self.x1_range[1] - self.x1_range[0]) * np.random.random([self.N, 1]) + self.x1_range[0]
        x2_values = (self.x2_range[1] - self.x2_range[0]) * np.random.random([self.N, 1]) + self.x2_range[0]
        return np.concatenate((x1_values, x2_values), 1)


def plot_clusters(cluster0, cluster1):
    fig, ax = plt.subplots()
    ax.scatter(cluster0[:, 0], cluster0[:, 1], c='r')
    ax.scatter(cluster1[:, 0], cluster1[:, 1], c='b')
    ax.grid()
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-1, 10])
    plt.show()


def normalize(X: np.ndarray, a: int = -1, b: int = 1, x_min = None, x_max = None):
    """
    Applies Min-Max Feature Scaling.
    :param X: The array that is to be normalised
    :param a: The lower-bound of the scaled data
    :param b: The upper-bound of the scaled data
    """

    if x_min is None:
        x_min = np.min(X, axis=0)

    if x_max is None:
        x_max = np.max(X, axis=0)

    denominator = x_max - x_min
    numerator = (X - x_min)*(b-a)

    return a + numerator/denominator


# number of points per cluster
N = 100
cluster_0_params = ClusterParams([0, 10], [2.5, 4], N)
cluster_1_params = ClusterParams([0, 10], [0, 1.5], N)

# generate X training data
cluster_0 = cluster_0_params.generate_cluster()
cluster_1 = cluster_1_params.generate_cluster()
X_train = normalize(np.concatenate((cluster_0, cluster_1), 0))

# generate X validation data
cluster_0 = cluster_0_params.generate_cluster()
cluster_1 = cluster_1_params.generate_cluster()
X_val = normalize(np.concatenate((cluster_0, cluster_1), 0))

y_train = np.concatenate((np.zeros((N, 1)), np.ones((N, 1))), 0)
y_val = np.concatenate((np.zeros((N, 1)), np.ones((N, 1))), 0)

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(X_train[:N, 0], X_train[:N, 1], c='r', label='C1')
plt.scatter(X_train[N:, 0], X_train[N:, 1], c='b', label='C2')
plt.grid()
plt.title("Training")
plt.legend()
plt.subplot(2, 1, 2)
plt.scatter(X_val[:N, 0], X_val[:N, 1], c='r', label='C1')
plt.scatter(X_val[N:, 0], X_val[N:, 1], c='b', label='C2')
plt.grid()
plt.title("Validation")
plt.legend()
plt.show()

train_dataframe = pd.DataFrame(X_train, columns=['x0', 'x1'])
train_dataframe = train_dataframe.assign(label=y_train)
validation_dataframe = pd.DataFrame(X_val, columns=['x0', 'x1'])
validation_dataframe = train_dataframe.assign(label=y_val)

train_dataframe.to_csv('synthetic_train_data.csv', index=False)
validation_dataframe.to_csv('synthetic_validation_data.csv', index=False)
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ClusterParams:
    def __init__(self, x1_range, x2_range, num_points):
        self.x1_range = x1_range
        self.x2_range = x2_range
        self.N = num_points

    def generate_cluster(self):
        x1_values = (self.x1_range[1] - self.x1_range[0]) * np.random.random([self.N, 1]) + self.x1_range[0]
        x2_values = (self.x2_range[1] - self.x2_range[0]) * np.random.random([self.N, 1]) + self.x2_range[0]
        return np.concatenate((x1_values, x2_values), 1)


def plot_clusters(cluster0, cluster1):
    fig, ax = plt.subplots()
    ax.scatter(cluster0[:, 0], cluster0[:, 1], c='r')
    ax.scatter(cluster1[:, 0], cluster1[:, 1], c='b')
    ax.grid()
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-1, 10])
    plt.show()


def normalize(X: np.ndarray, a: int = -1, b: int = 1, x_min = None, x_max = None):
    """
    Applies Min-Max Feature Scaling.
    :param X: The array that is to be normalised
    :param a: The lower-bound of the scaled data
    :param b: The upper-bound of the scaled data
    """

    if x_min is None:
        x_min = np.min(X, axis=0)

    if x_max is None:
        x_max = np.max(X, axis=0)

    denominator = x_max - x_min
    numerator = (X - x_min)*(b-a)

    return a + numerator/denominator


# number of points per cluster
N = 100
cluster_0_params = ClusterParams([0, 10], [2.5, 4], N)
cluster_1_params = ClusterParams([0, 10], [0, 1.5], N)

# generate X training data
cluster_0 = cluster_0_params.generate_cluster()
cluster_1 = cluster_1_params.generate_cluster()
X_train = normalize(np.concatenate((cluster_0, cluster_1), 0))

# generate X validation data
cluster_0 = cluster_0_params.generate_cluster()
cluster_1 = cluster_1_params.generate_cluster()
X_val = normalize(np.concatenate((cluster_0, cluster_1), 0))

y_train = np.concatenate((np.zeros((N, 1)), np.ones((N, 1))), 0)
y_val = np.concatenate((np.zeros((N, 1)), np.ones((N, 1))), 0)

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(X_train[:N, 0], X_train[:N, 1], c='r', label='C1')
plt.scatter(X_train[N:, 0], X_train[N:, 1], c='b', label='C2')
plt.grid()
plt.title("Training")
plt.legend()
plt.subplot(2, 1, 2)
plt.scatter(X_val[:N, 0], X_val[:N, 1], c='r', label='C1')
plt.scatter(X_val[N:, 0], X_val[N:, 1], c='b', label='C2')
plt.grid()
plt.title("Validation")
plt.legend()
plt.show()

train_dataframe = pd.DataFrame(X_train, columns=['x0', 'x1'])
train_dataframe = train_dataframe.assign(label=y_train)
validation_dataframe = pd.DataFrame(X_val, columns=['x0', 'x1'])
validation_dataframe = train_dataframe.assign(label=y_val)

train_dataframe.to_csv('synthetic_train_data.csv', index=False)
validation_dataframe.to_csv('synthetic_validation_data.csv', index=False)
>>>>>>> e1241c2eff256f868ef210ea42ef25f0ce2e1fc8
