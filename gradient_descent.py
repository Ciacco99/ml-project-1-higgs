import numpy as np


def gradient_descent(gradient, x_0, max_iters, gamma):
    """
    Perform gradient descent

    Parameters:
        gradient (function: numpy array of shape=(D, ) => numpy array of shape=(D, )): The gradient function
        x_0 (numpy array of shape=(D, )): The initial value.
        max_iters (int): The maximum number of iterations.
        gamma (float): The step size.

    Returns:
        numpy array of shape=(D, ): The final value
    """
    x = x_0
    for i in range(max_iters):
        x = x - gamma * gradient(x)
    return x


def linear_regression_gd(gradient, loss_function, y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Parameters:
        gradient (function: (y, tx, w) => numpy array of shape=(D, )): The gradient function
        loss_function (function: (y, tx, w) => scalar): The loss function
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(ND)): The feature matrix.
        initial_w (numpy array of shape=(D, )): The initial value of w.
        max_iters (int): The maximum number of iterations.
        gamma (float): The step size.

    Returns:
        numpy array of shape=(D, ): the weights obtained using GD
        float: the loss of the obtained model
    """
    w = gradient_descent(lambda w: gradient(y, tx, w), initial_w, max_iters, gamma)

    loss = loss_function(y, tx, w)
    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        y = y[shuffle_indices]
        tx = tx[shuffle_indices]

    for batch_num in range(num_batches):
        start_i = batch_num * batch_size
        end_i = min((batch_num + 1) * batch_size, data_size)
        if start_i != end_i:
            yield y[start_i:end_i], tx[start_i:end_i]


def stochastic_gradient_descent(gradient, batches, x_0, max_iters, gamma):
    """
    Perform stochastic gradient descent

    Parameters:
        gradient (function: (batch, numpy array of shape=(D, )) => numpy array of shape=(D, )): The gradient function
        batches (function: int => batch): The batches over which perform SGD
        x_0 (numpy array of shape=(D, )): The initial value.
        max_iters (int): The maximum number of iterations.
        gamma (float): The step size.

    Returns:
        numpy array of shape=(D, ): The final value
    """
    x = x_0
    for i in range(max_iters):
        for batch in batches(i):
            grad = gradient(batch, x)
            x = x - gamma * grad
    return x


def linear_regression_sgd(
    gradient,
    loss_function,
    y,
    tx,
    initial_w,
    max_iters,
    gamma,
    batch_size,
    num_batches,
    shuffle,
):
    """
    Linear regression using stochastic gradient descent gradient descent.

    Parameters:
        gradient (function: (y, tx, w) => numpy array of shape=(D, )): The gradient function
        loss_function (function: (y, tx, w) => scalar): The loss function
        y (numpy array of shape=(N, )): The target variable vector.
        tx (numpy array of shape=(N, D)): The feature matrix.
        initial_w (numpy array of shape=(D, )): The initial value of w.
        max_iters (int): The maximum number of iterations.
        gamma (float): The step size.
        batch_size (int): Size of batch
        num_batches (int): Number of batches to treat per iteration
        shuffle (int): True if shuffling batches

    Returns:
        numpy array of shape=(D, ): the weights obtained using SGD
        float: the loss of the obtained model
    """
    # Stochastic gradient descent
    w = stochastic_gradient_descent(
        lambda batch, x: gradient(batch[0], batch[1], x),
        lambda i: batch_iter(y, tx, batch_size, num_batches, shuffle),
        initial_w,
        max_iters,
        gamma,
    )

    # Compute loss for final w
    loss = loss_function(y, tx, w)
    return w, loss
