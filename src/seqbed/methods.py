from GPyOpt.optimization.optimizer import OptLbfgs
from GPyOpt.core.task.space import Design_space
import itertools
import numpy as np

import sklearn.neighbors


def indicator_boundaries(bounds, d):

    """
    Checks if the provided design 'd' is within the specified 'bounds'.

    Parameters
    ----------
    bounds: np.ndarray
        Bounds of the design domain.
    d: np.ndarray
        Proposed design.
    """

    bounds = np.array(bounds)
    check = bounds.T - d
    low = all(i <= 0 for i in check[0])
    high = all(i >= 0 for i in check[1])

    if low and high:
        ind = 1.0
    else:
        ind = 0.0

    return np.array([[ind]])


def fun_dfun(obj, space, d):

    """
    Computes the posterior predictive and posterior predictive gradients of the
    provided GPyOpt object.

    Parameters
    ----------
    obj: GPyOpt object
        The GPyOpt object with a surrogate probabilistic model.
    space: GPyOpt space
        A GPyOpt object that contains information about the design domain.
    d: np.ndarray
        Proposed design.
    """

    mask = space.indicator_constraints(d)

    pred = obj.model.predict_withGradients(d)[0][0][0]
    d_pred = obj.model.predict_withGradients(d)[2][0]

    return float(pred * mask), d_pred * mask


def get_GP_optimum(obj):

    """
    Finds the optimal design by maximising the mean of the surrogate
    probabilistic GP model.

    Parameters
    ----------
    obj: GPyOpt object
        The GPyOpt object with a surrogate probabilistic model.
    """

    # Define space
    space = Design_space(obj.domain, obj.constraints)
    bounds = space.get_bounds()

    # Specify Optimizer --- L-BFGS
    optimizer = OptLbfgs(space.get_bounds(), maxiter=1000)

    # Do the optimisation
    x, _ = optimizer.optimize(
        x0=obj.x_opt,
        f=lambda d: fun_dfun(obj, space, d)[0],
        f_df=lambda d: fun_dfun(obj, space, d),
    )
    # TODO: MULTIPLE RE-STARTS FROM PREVIOUS BEST POINTS

    # Round values if space is discrete
    xtest = space.round_optimum(x)[0]

    if space.indicator_constraints(xtest):
        opt = xtest
    else:
        # Rounding mixed things up, so need to look at neighbours

        # Compute neighbours to optimum
        idx_comb = np.array(
            list(itertools.product([-1, 0, 1], repeat=len(bounds))))
        opt_combs = idx_comb + xtest

        # Evaluate
        GP_evals = list()
        combs = list()
        for idx, d in enumerate(opt_combs):

            cons_check = space.indicator_constraints(d)[0][0]
            bounds_check = indicator_boundaries(bounds, d)[0][0]

            if cons_check * bounds_check == 1:
                pred = obj.model.predict(d)[0][0][0]
                GP_evals.append(pred)
                combs.append(d)
            else:
                pass

        idx_opt = np.where(GP_evals == np.min(GP_evals))[0][0]
        opt = combs[idx_opt]

    return opt


def resampling(params, weights, nsamples=1000, bounds=None):

    """
    Performs the resampling procedure to obtain new samples from the current
    belief distribution.

    Parameters
    ----------
    params: np.ndarray
        The parameter samples from the original belief distribution, e.g.
        prior, which have become degenerate.
    weights: np.ndarray
        The weights corresponding to the parameter samples 'params'.
    nsamples: int
        The number of resampled parameters desired.
    bounds: Boolean or list of lists
        If None, the parameter domain has infinite support. Otherwise, 'bounds'
        needs to be a list of lower and upper bounds: [[low_0, upp_0], ...].
        (default is None)
    """

    # Transform Parameters to be between 0 and 1
    if len(params.shape) == 1:
        params = params.reshape(-1, 1)

    # transform the parameter space
    PD = params.shape[-1]
    params_transf = list()
    for i in range(PD):
        p0 = params.T[i]
        p0_new = ((1 - 0) / (np.max(p0) - np.min(p0))) * (p0 - np.max(p0)) + 1
        params_transf.append(p0_new)
    params_transf = np.array(params_transf).T

    # compute the nearest neighbours
    kdt = sklearn.neighbors.KDTree(
        params_transf, leaf_size=30, metric="euclidean")
    results = kdt.query(params_transf, k=2, return_distance=True)

    # compute the covariance of parameters in the transformed space
    median_scaled = np.median(results[0][:, 1])
    emp_cov_med = np.identity(PD) * median_scaled ** 1

    # Obtain normalised weights
    ws_norm = weights / np.sum(weights)

    # Get indices of selected parameters
    c_list = list()
    for _ in range(nsamples):
        cat = np.random.choice(range(len(weights)), p=ws_norm)
        c_list.append(cat)

    if isinstance(params[0], np.ndarray):
        arr = True
    else:
        arr = False

    # transform the bounds
    if bounds is not None:
        bounds_transf = list()
        for i in range(PD):
            p0 = params.T[i]

            # rescale bounds params
            b = ((1 - 0) / (np.max(p0) - np.min(p0))) * (
                np.array(bounds[i]) - np.max(p0)
            ) + 1
            bounds_transf.append(list(b))

    # Sample from a Mixture of Gaussians
    p_new = list()
    for cat in c_list:

        if bounds is None:
            p_prop = np.random.multivariate_normal(
                params_transf[cat], emp_cov_med, size=1
            )[0]
        else:

            accept = False
            while accept is False:

                # Propose new sample
                p_prop = np.random.multivariate_normal(
                    params_transf[cat], emp_cov_med, size=1
                )[0]
                if arr:
                    decision = 1
                    for idx in range(len(p_prop)):
                        if (
                            bounds_transf[idx][0]
                            <= p_prop[idx]
                            <= bounds_transf[idx][1]
                        ):
                            continue
                        else:
                            decision = 0

                    if decision == 1:
                        accept = True
                    else:
                        continue
                else:

                    if bounds_transf[0] <= p_prop <= bounds_transf[1]:
                        accept = True

        p_new.append(p_prop)

    p_new = np.array(p_new)
    w_new = np.ones(len(p_new))

    # transform the bounds back
    p_new_retransf = list()
    for i in range(PD):
        p0 = params.T[i]
        par0 = p_new.T[i]

        # rescale bounds params
        p = ((np.max(p0) - np.min(p0)) / (1 - 0)) * (par0 - 1) + np.max(p0)
        p_new_retransf.append(p)
    p_new_retransf = np.array(p_new_retransf).T

    if p_new_retransf.shape[-1] == 1:
        p_new_retransf = p_new_retransf.reshape(-1)

    return p_new_retransf, w_new


def abc_likelihood(data, y0, epsilon=0.1):

    """
    Compute ABC likelihood density.

    Parameters
    ----------
    data: np.npdarray
        Array of several (simulated) data points
    y0: np.ndarray
        Single, observed data point.
    epsilon: float >= 0
        ABC threshold for acceptance.
        (default is 0.1)
    """

    # Compute distances between data and observation
    dist = np.linalg.norm(y0 - data, axis=1)

    # Count data that is sufficiently close
    count = np.mean(dist <= epsilon)

    return count


def ABC_resampling(
    data_d, data_y, simobj, prior_samples, weights, prior_pdf, e0=0.1, n_iter=1
):

    """
    Function to resample particles via ABC used for the BD-Opt utility.

    Parameters
    ----------
    data_d: np.ndarray
        Array of previous optimal designs.
    data_y: np.ndarray
        Array of previous real-world observations.
    simobj: simulator object
        Object of the implicit simulator model.
    prior_samples: np.ndarray
        The parameter samples from the original belief distribution, e.g.
        prior, which have become degenerate.
    weights: np.ndarray
        The weights corresponding to the parameter samples 'params'.
    prior_pdf: np.ndarray
        Prior densities corresponding to each entry in 'prior_samples'.
    e0: float
        Initial ABC threshold for accepting/rejecting proposed samples.
        (default is 0.1)
    n_iter: int
        Iteration of the sequential BED algorithm.
        (default is 1)
    """

    # resample parameters
    ws_norm = weights / np.sum(weights)
    p_i = list()
    N_res = len(prior_samples)
    for _ in range(N_res):
        cat = np.random.choice(range(len(ws_norm)), p=ws_norm)
        p_i.append(prior_samples[cat])
    p_i = np.array(p_i)

    # Simulate data for each p_i given _all_ the data
    y_sim = list()
    for d in data_d:
        y_i = simobj.sample_data(d, p_i)
        y_sim.append(y_i)
    y_sim = np.array(y_sim)[np.newaxis].T[0]

    # Compute tuning parameters
    epsilon = e0 / np.sqrt(n_iter)
    cov = np.cov(p_i)
    if isinstance(p_i[0], np.ndarray):
        arr = True
    else:
        arr = False

    # Move particles via a random walk
    pp_new = list()
    for idx, p in enumerate(p_i):
        accept = False

        while accept is False:

            # Propose new sample
            if arr:
                p_prop = np.random.multivariate_normal(p, cov)
                # condition for Death Model / SIR Model
                if (p_prop < 0).any():
                    continue
            else:
                p_prop = np.random.normal(p, cov ** 0.5)
                # condition for Death Model / SIR Model
                if p_prop < 0:
                    continue

            # Compute data for proposed sample
            y_props = list()
            for d in data_d:
                yp = simobj.sample_data(d, p_prop, num=1)
                y_props.append(yp)
            y_props = np.array(y_props)

            # Check if all the data agrees with observations
            agree = 1
            for k in range(len(y_props)):
                if np.linalg.norm(y_props[k] - data_y[k], axis=0) <= epsilon:
                    agree *= 1
                else:
                    agree *= 0

            # if proposed data agrees with observations, continue
            if agree:

                # compute alpha
                prior_ratio = prior_pdf(p_prop) / prior_pdf(p)

                # check if proposed sample is accepted
                if np.random.random() < prior_ratio or prior_ratio > 1:
                    pp_new.append(p_prop)
                    accept = True
                else:
                    pass
            else:
                pass

    pp_new = np.array(pp_new)
    ww_new = np.ones(len(pp_new))

    return pp_new, ww_new
