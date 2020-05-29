import numpy as np
import scipy.optimize as sco


def Psi(x):

    """
    Transformation used in the robust MEstimator.

    Parameters
    ----------
    x: float
        Robust mean estimate.
    """

    if x >= 0:
        p = np.log(1 + x + 0.5 * x ** 2)
    else:
        p = -np.log(1 - x + 0.5 * x ** 2)

    return p


def Root(x, samples):

    """
    Computes a non-linear function over a set of robust mean estimates.

    Parameters
    ----------
    x: float
        Robust mean estimate.
    samples: np.ndarray
        Array of values over which to compute the robust mean.
    """

    # Set sensitivity parameter
    alpha = np.sqrt(2 / (len(samples) * np.std(samples)))

    # compute input to non-linear function
    inp = alpha * (samples - x)

    # compute psis and sum over them
    F_i = np.array([Psi(i) for i in inp])
    F = np.sum(F_i)

    return F


def MEstimator(samples):

    """
    Computes the robust mean of 'samples'.

    Parameters
    ----------
    samples: np.ndarray
        Array of values over which to compute the robust mean.
    """

    # Find the roots of the non-linear equation
    mu = np.mean(samples)
    data_opt = sco.root(Root, mu, args=samples, tol=1e-14, method="broyden1")

    return data_opt.x
