import numpy as np

from gradient_descent import *
from helpers import sigmoid

# To run tests here from project directory:
# pytest grading_tests --github_link ./
# pytest grading_tests --github_link https://github.com/CS-433/ml-project-1-s/tree/4c3ff25739b196451cb3c11a9b8acf885da981c4 (click 'y' on keyboard to obtain this link from the browser)

# NOTES: this file is a bit more modular than the basic implementations required for the project,
# here we use sigmoid() from helpers and the gradient descent functions from gradient_descent.py
# This is to keep the code modular and readable, and to make it easier to test the functions in this file,
# but also to avoid redundant copy/paste in the other files used to run the project
# We added docstrings to make it easier to understand what the functions do even when they call other functions.


def compute_mse(y, tx, w):
    """
    Compute the mean squared error of the model (with a 1/2 factor)

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix
        w (numpy array of shape=(D, )): The vector of model parameters.

    Returns:
        float: mean squared error of the model
    """
    N = len(y)
    e = y - tx.dot(w)
    mse = 1 / (2 * N) * np.sum(e**2)
    return mse


def compute_mse_gradient(y, tx, w):
    """
    Compute the gradient of the MSE

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix.
        w (numpy array of shape=(D, )): The vector of model parameters.

    Returns:
        numpy array of shape=(D, ): gradient of the mean squared error (w.r.t. w)
    """
    N = len(y)
    e = y - tx.dot(w)
    grad = -1 / N * tx.T.dot(e)
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent for MSE.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix.
        initial_w (numpy array of shape=(D, )): The vector of initial model parameters.
        max_iters (int): The maximum number of iterations to run the algorithm.
        gamma (float): The learning rate.

    Returns:
        numpy array of shape=(D, ): the weights obtained
        float: the loss of the obtained model
    """
    return linear_regression_gd(
        compute_mse_gradient, compute_mse, y, tx, initial_w, max_iters, gamma
    )


def mean_squared_error_sgd(
    y, tx, initial_w, max_iters, gamma, batch_size=1, num_batches=1, shuffle=True
):
    """
    Linear regression using stochastic gradient descent for MSE.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix.
        initial_w (numpy array of shape=(D, )): The vector of initial model parameters.
        max_iters (int): The maximum number of iterations to run the algorithm.
        gamma (float): The learning rate.
        batch_size (int): The size of the batch.
        num_batches (int): The number of batches.
        shuffle (bool): Whether to shuffle the data before splitting it into batches.

    Returns:
        numpy array of shape=(D, ): the weights obtained
        float: the loss of the obtained model
    """
    return linear_regression_sgd(
        compute_mse_gradient,
        compute_mse,
        y,
        tx,
        initial_w,
        max_iters,
        gamma,
        batch_size,
        num_batches,
        shuffle,
    )


def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix

    Returns:
        numpy array of shape=(D, ): the weights obtained
        float: the loss of the obtained model
    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix

    Returns:
        numpy array of shape=(D, ): the weights obtained
        float: the loss of the obtained model
    """

    N, D = tx.shape
    lambda_prime = lambda_ * 2 * N
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss


def compute_logistic_loss(y, tx, w):
    """
    Compute the negative log-likelihood for logistic regression

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix
        w (numpy array of shape=(D, )): The vector of model parameters.

    Returns:
        float: negative log-likelihood for logistic regression
    """
    N = len(y)

    # intermediary variables
    sigmoid_ = sigmoid(tx @ w)

    # small epsilon is added to avoid numerical issues on log(0)
    epsilon = 1e-10
    sigmoid_ = np.maximum(epsilon, sigmoid_)
    sigmoid_ = np.minimum(1 - epsilon, sigmoid_)

    return -1.0 / N * np.sum(y * np.log(sigmoid_) + (1 - y) * np.log(1 - sigmoid_))


def compute_logistic_gradient(y, tx, w):
    """
    Compute the gradient of the loss for logistic regression.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix
        w (numpy array of shape=(D, )): The vector of model parameters.

    Returns:
        numpy array of shape=(D, ): gradient of the loss for logistic regression (w.r.t. w)
    """
    N = len(y)
    return 1.0 / N * (tx.T @ (sigmoid(tx @ w) - y))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(ND)): The feature matrix.
        initial_w (numpy array of shape=(D, )): The initial value of w.
        max_iters (int): The maximum number of iterations.
        gamma (float): The step size.

    Returns:
        numpy array of shape=(D, ): the weights obtained using GD
        float: the loss of the obtained model
    """
    return linear_regression_gd(
        compute_logistic_gradient,
        compute_logistic_loss,
        y,
        tx,
        initial_w,
        max_iters,
        gamma,
    )


def compute_reg_logistic_loss(y, tx, w, lambda_):
    """
    Compute the penalized negative log-likelihood for logistic regression

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix
        w (numpy array of shape=(D, )): The vector of model parameters.
        lambda_ (float): Regularization factor.

    Returns:
        float: Penalized negative log-likelihood for logistic regression
    """
    loss = compute_logistic_loss(y, tx, w)
    penalty = lambda_ * np.sum(w**2)
    return loss + penalty


def compute_reg_logistic_gradient(y, tx, w, lambda_):
    """
    Compute the gradient of the loss for penalized logistic regression.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix
        w (numpy array of shape=(D, )): The vector of model parameters.
        lambda_ (float): Regularization factor.

    Returns:
        numpy array of shape=(D, ): gradient of the loss for penalized logistic regression (w.r.t. w)
    """

    loss_gradient = compute_logistic_gradient(y, tx, w)
    penalty_gradient = 2.0 * lambda_ * w
    return loss_gradient + penalty_gradient


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Parameters:
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N,D)): The feature matrix.
        lambda_ (float): Regularization factor.
        initial_w (numpy array of shape=(D, )): The initial value of w.
        max_iters (int): The maximum number of iterations.
        gamma (float): The step size.

    Returns:
        numpy array of shape=(D, ): the weights obtained using logistic regression
        float: the loss of the obtained model
    """

    def gradient(y, tx, w):
        return compute_reg_logistic_gradient(y, tx, w, lambda_)

    def loss_function(y, tx, w):
        return compute_reg_logistic_loss(y, tx, w, lambda_)

    w, _ = linear_regression_gd(
        gradient, loss_function, y, tx, initial_w, max_iters, gamma
    )
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
