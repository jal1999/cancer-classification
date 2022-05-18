import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')


def init_scaling_matrix() -> np.ndarray:
    matrix = np.zeros((data.shape[1] - 2, 3))

    for j in range(matrix.shape[0]):
        curr_max = -float('inf')
        curr_min = float('inf')
        total = 0
        for i in range(data.shape[0]):
            val = data.iloc[i, j + 2]
            total += val
            curr_max = max(val, curr_max)
            curr_min = min(val, curr_min)
        matrix[j][0] = curr_min
        matrix[j][1] = curr_max
        matrix[j][2] = total / data.shape[0]
    return matrix


def init_feature_matrix() -> np.ndarray:
    """
    Creates feature matrix of training data.

    :return: Feature matrix from training data.
    """
    matrix = np.zeros((data.shape[0], data.shape[1] - 1))
    scales = init_scaling_matrix()

    for i in range(matrix.shape[0]):
        matrix[i][0] = 1  # Bias
        for j in range(1, matrix.shape[1]):
            vals = scales[j - 1]
            num = data.iloc[i][j + 1] - vals[2]
            den = vals[1] - vals[0]
            # den = den + .0000000001 if den == 0 else den
            matrix[i][j] = num / den
    return matrix


def init_label_vec() -> np.ndarray:
    """
    Creates column vector of true labels for each training example.

    :return: column vector of labels.
    """
    vec = np.zeros((data.shape[0], 1))

    for i in range(data.shape[0]):
        c = data.loc[i, 'diagnosis']
        vec[i][0] = 1 if c == 'M' else 0
    return vec


def init_theta() -> np.ndarray:
    """
    Creates column vector of weights, all initialized to 0.

    :return: column vector of weights.
    """
    return np.zeros((data.shape[1] - 1, 1))