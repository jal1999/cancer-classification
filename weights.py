import numpy as np


def predict(x: np.ndarray, theta: np.ndarray) -> int:
    """
    Computes label prediction.

    :param x: row vector of features of a single example. (1 x N + 1)
    :param theta: column vector of weights for each feature. (N + 1 x 1)
    :return: Label prediction
    """
    sig = 1 / (1 + (np.exp(-np.dot(x, theta))))
    return 1 if sig >= .5 else 0


def gradient_descent(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_its: int) -> np.ndarray:
    """
    Computes optimal weights.

    :param x: feature matrix (M x N+1)
    :param y: column vectors of true labels. (M x 1)
    :param theta: column vector of weights for each feature. (N+1 x 1)
    :param alpha: learning rate.
    :param num_its: number of iterations to run the descent.
    :return: optimal weights.
    """
    for n in range(num_its):
        new_theta = np.zeros(theta.shape)
        for j in range(theta.shape[0]):
            total_sum = 0
            for i in range(x.shape[0]):
                pred = predict(x[i], theta)
                second_part = (pred - y[i]) * x[i][j]
                if j == 0:
                    total_sum += second_part
                else:
                    total_sum += second_part + ((alpha / x.shape[0]) * theta[j])
            new_theta[j] = theta[j] - ((alpha / x.shape[0]) * total_sum)
        theta = new_theta.copy()
    return theta
