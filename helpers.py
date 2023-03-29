"""Some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def load_csv_names(data_path):
    with open(data_path) as f:
        reader = csv.reader(f)
        columns = next(reader)[2:]
        name_map = dict(zip(columns, range(len(columns))))
        return name_map


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


# =======================================================================================================================#
# Helper Methods
# =======================================================================================================================#


def standardize(x):
    """Standardize the input matrix. Mean = 0, Std = 1"""
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def remove_outliers(y, x, factor=1.5):
    """Find outliers in dataset"""

    N = x.shape[1]
    # Quantile which we'll consider as Q1,
    # The box corresponding to [q, 1-q]^N will contain
    # half the data, so this is very similar to the IQR method
    # in the univariate case
    q = 0.5 - np.power(1 / 2, 1 / N) / 2

    # This method is very approximate, we could probably have a better
    # discriminator for outliers, but this was pretty easy to implement
    def find_outliers(col):
        Q1 = np.quantile(col, q)
        Q3 = np.quantile(col, 1 - q)
        IQR = Q3 - Q1
        return (col < Q1 - factor * IQR) | (col > Q3 + factor * IQR)

    # Helper function, since we'll remove the outliers for data
    # corresponding to signal and background separately
    def _remove_outliers(y, x):
        flags = np.apply_along_axis(find_outliers, axis=0, arr=x)
        outliers = np.any(flags, axis=1)
        rows = ~outliers
        return y[rows], x[rows]

    # Split data into signal and background and remove outliers in
    # each case
    signal = y == 1
    background = ~signal

    y_sig, x_sig = _remove_outliers(y[signal], x[signal])
    y_bg, x_bg = _remove_outliers(y[background], x[background])

    y_clean = np.concatenate([y_sig, y_bg])
    x_clean = np.concatenate([x_sig, x_bg])

    return y_clean, x_clean


def partition(y, x, ratio=0.8, seed=1):
    """split the dataset based on the split ratio.
    Arguments: y (class labels)
               x (features)
               ratio (split ratio, between 0 and 1, default 0.8)
               seed (seed for the random number generator, default = 1)
    Returns: y_train, x_train, y_test, x_test
    Example: y_train, x_train, y_test, x_test = partition(y, x, ratio=0.8, seed=1)
             #len(y_train) = 0.8*len(y)
             #len(y_test) = 0.2*len(y)

    """
    # set seed
    np.random.seed(seed)
    nb_row = len(y)
    indices = np.random.permutation(
        nb_row
    )  # shuffle the indices, np.random.permutation(10) returns a new array with 10 random numbers between 0 and 9
    split = int(np.floor(ratio * nb_row))
    train_indices, test_indices = indices[:split], indices[split:]
    train_y, test_y = y[train_indices], y[test_indices]
    train_x, test_x = x[train_indices], x[test_indices]
    return train_y, train_x, test_y, test_x


def confusion_matrix(y, y_pred):
    """_summary_ : Compute the confusion matrix

    Args:
        y: True labels
        y_pred: Predicted labels

    Returns:
        Confusion Matrix: 2x2 matrix true positives, false positives, false negatives, true negatives in absulute numbers
    """
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == -1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))
    tn = np.sum((y == -1) & (y_pred == -1))
    return np.array([[tp, fp], [fn, tn]])


def confusion_matrix_ratio(y, y_pred):
    """Compute the confusion matrix in ratio"""
    tp, fp, fn, tn = confusion_matrix(y, y_pred).ravel()
    return np.array(
        [[tp / (tp + fn), fp / (fp + tn)], [fn / (tp + fn), tn / (fp + tn)]]
    )


def precision(tp, fp):
    """Compute the precision of the model by standard definition of precision: True_Positives/(True_Positives + False_Positives)"""
    return tp / (tp + fp)


def recall(tp, fn):
    """Compute the recall of the model by standard definition of recall: True_Positives/(True_Positives + False_Negatives)"""
    return tp / (tp + fn)


def f1_score(tp, fp, fn):
    """Compute the recall of the model by standard definition of f1_score: 2*True_Positives/(2*True_Positives + False_Positives + False_Negatives)"""
    return 2 * tp / (2 * tp + fp + fn)


# def accuracy(tp, fp, fn, tn):
#    """Compute the accuracy of the model by standard definition of accuracy: (True_Positives + True_Negatives)/(True_Positives + True_Negatives + False_Positives + False_Negatives)"""
#    return (tp + tn)/(tp + fp + fn + tn)


def accuracy(y, y_pred):
    """Compute the accuracy of the model by standard definition of accuracy: (True_Positives + True_Negatives)/(True_Positives + True_Negatives + False_Positives + False_Negatives)"""
    return np.sum(y == y_pred) / len(y)


def confusion_metrics(y, y_pred):
    """Compute the confusion matrix and the metrics precision, recall, f1_score and accuracy"""
    tp, fp, fn, tn = confusion_matrix(
        y, y_pred
    ).ravel()  # ravel() returns a view on the flattened array (flatten() returns a copy)
    return precision(tp, fp), recall(tp, fn), f1_score(tp, fp, fn), accuracy(y, y_pred)


def print_data_statistics(y, x):
    """Compute the statistics of the dataset"""
    print("Number of samples: {}".format(len(y)))
    print("Number of features: {}".format(x.shape[1]))
    print("Number of positives: {}".format(np.sum(y == 1)))
    print("Number of negatives: {}".format(np.sum(y == -1)))
    print("Ratio of positives: {}".format(np.sum(y == 1) / len(y)))
    print("Ratio of negatives: {}".format(np.sum(y == -1) / len(y)))
    print("Number of missing values: {}".format(np.sum(np.isnan(x))))
    print(
        "Feature with most missing values: {}".format(
            np.argmax(np.sum(np.isnan(x), axis=0))
        )
    )
    print(
        "Number of missing values in this feature: {}".format(
            np.max(np.sum(np.isnan(x), axis=0))
        )
    )


def build_k_indices(N, k_fold):
    """build k indices for k-fold.

    Parameters:
        N (int): Total number of samples
        k_fold (int): K in K-fold, i.e. the fold num
    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    interval = N // k_fold
    indices = np.random.permutation(N)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def feature_expansion(x, degree=2):
    """Build expanded feature matrix with cross terms
    polynomials of at most specified degree"""

    def build_list_of_powers(N, D):
        if N == 0:
            return [[]]
        powers = []
        for i in range(min(2, D + 1)):
            powers_ = build_list_of_powers(N - 1, D - i)
            for p in powers_:
                p.append(i)
                powers.append(p)
        return powers

    N_features = x.shape[1]
    powers = build_list_of_powers(N_features, 2)

    poly = np.concatenate([x**i for i in range(2, degree + 1)], axis=1)
    sq_root = x / np.sqrt(np.abs(x))
    cb_root = x / (np.cbrt(np.abs(x)) ** 2)
    expo = np.exp(x)
    cross = np.concatenate(
        [np.prod(x**p, axis=1, keepdims=True) for p in powers], axis=1
    )
    return np.concatenate((poly, cross, sq_root, cb_root, expo), axis=1)


def polynomial_expansion(x, degree=2):
    """Expand the features by polynomial expansion of specified degree, default degree is 2"""
    poly = np.concatenate([x**i for i in range(1, degree + 1)], axis=1)
    return np.concatenate((np.ones((x.shape[0], 1)), poly), axis=1)


def isOverfitting(y, y_pred, y_test, y_pred_test):
    """Check if the model is overfitting by comparing the training and test accuracy. True if accuracy of training set is higher than accuracy of test set"""
    return accuracy(y, y_pred) > accuracy(y_test, y_pred_test)


def performance(y, y_pred, y_test, y_pred_test):
    """Compute the performance of the model by computing the accuracy of the training and test set"""
    return accuracy(y, y_pred), accuracy(y_test, y_pred_test)


def predict_labels(w, data, threshold=0):
    """
    Generates class predictions for y {-1,1} given a threshold.
    For logistic regression, the threshold should be 0.5 -- CHECK if this is correct

    """
    y_pred = np.dot(data, w)
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    return y_pred


def sigmoid(x):
    """
    Apply sigmoid function on x.

    Parameters:
        x: numpy array dtype=float

    Returns:
        numpy array
    """

    # Separate cases to avoid numerical error as much as possible

    def positive_case(x):
        z = np.exp(-x)
        return 1.0 / (1.0 + z)

    def negative_case(x):
        z = np.exp(x)
        return z / (1.0 + z)

    return np.piecewise(x, [x >= 0], [positive_case, negative_case])


def predict_labels_logistic(w, data, threshold=0.5):
    """ """
    # we apply the sigmoid function
    x = data @ w
    y_pred = sigmoid(x)
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    return y_pred


def augment(y, x, noise=0.1):
    """Augment by adding new data points with the same label as the original data point but with a small noise"""
    return np.concatenate((y, y)), np.concatenate(
        (x, x + noise * np.random.randn(*x.shape))
    )
