import numpy as np

# Own Libraries
import seqbed.inference as inference
import seqbed.robust as robust

# for parallel processing
from joblib import Parallel, delayed
import multiprocessing


class Utility:

    """
    Base class to implement a utility function.

    Attributes
    ----------
    prior_samples: np.ndarray
        Array of prior parameter samples.
    weights: np.ndarray
        Array of weights corresponding to each prior sample.
    simobj: simulator object
        Object of the implicit simulator model.
    """

    def __init__(self, prior_samples, weights, simobj):

        """
        Parameters
        ----------
        prior_samples: np.ndarray
            Array of prior parameter samples.
        weights: np.ndarray
            Array of weights corresponding to each prior sample.
        simobj: simulator object
            Object of the implicit simulator model.
        """

        self.prior_samples = prior_samples
        self.weights = weights
        self.simobj = simobj


class MutualInformation(Utility):

    """
    Child class that implements the mutual information utility function.

    Attributes
    ----------
    prior_samples: np.ndarray
        Array of prior parameter samples.
    weights: np.ndarray
        Array of weights corresponding to each prior sample.
    simobj: simulator object
        Object of the implicit simulator model.
    evalmethod: str
        Method to evaluate the density ratio. Currently only LFIRE is
        implemented.
    smc: Boolean
        Choose whether or not to use SMC methods to evaluate the utility. If
        'True' a single ratio is computed for each prior sample. If 'False',
        several ratios may be computed for a single prior sample.
    r_obs: np.ndarray
        Array of LFIRE ratios obtained for each prior sample through the
        compute_final() method.
    b_obs: np.ndarray
        Array of LFIRE coefficients obtained for each prior sample through the
        compute_final() method.

    Methods
    -------
    compute:
        Computes the mutual information via LFIRE density ratios.
    compute_final:
        Computes the LFIRE ratios and coefficients given some real-world
        observations.
    """

    def __init__(
            self, prior_samples, weights, simobj, evalmethod="lfire",
            smc=False
    ):

        """
        Parameters
        ----------
        prior_samples: np.ndarray
            Array of prior parameter samples.
        weights: np.ndarray
            Array of weights corresponding to each prior sample.
        simobj: simulator object
            Object of the implicit simulator model.
        evalmethod: str
            Method to evaluate the density ratio. Currently only LFIRE is
            implemented.
            (default is 'lfire')
        smc: Boolean
            Choose whether or not to use SMC methods to evaluate the utility.
            If 'True' a single ratio is computed for each prior sample. If
            'False', several ratios may be computed for a single prior sample.
            (default is False)
        """

        super(MutualInformation, self).__init__(prior_samples, weights, simobj)

        self.evalmethod = evalmethod
        self.smc = smc

    def _mean_eval(self, U):
        return np.mean(U)

    def _median_eval(self, U):
        return np.median(U)

    def _robust_eval(self, U):
        return robust.MEstimator(U)

    def compute(self, d, numsamp=10000, evaltype="robust", verbose=True):

        """
        Computes the mutual information via LFIRE density ratios.

        Parameters
        ----------
        d: np.ndarray
            The current design at which to evaluate the utility.
        numsamp: int
            Number of parameter samples to be used in evaluating the utility.
            (default is 10000)
        evaltype: str
            Type of Monte-Carlo evaluation; can currently take 'mean', 'median'
            or 'robust' as input.
            (default 'robust')
        verbose: Boolean
            If True, prints the current design to stdout.
            (default is True)
        """

        # GPyOpt wraps the design point in a weird double array // hacky fix
        d = d[0]

        if verbose:
            print("Design point: ", d)

        if self.evalmethod == "lfire":

            # compute the LFIRE ratios for 'numsamp' prior samples, where some
            # may be repeated; set to default 10000 for now.
            if self.smc:
                infobj = inference.LFIRE_SMC(
                    d, self.prior_samples, self.weights, self.simobj
                )
                logr, _ = infobj.ratios()
                utils = np.array(logr) * self.weights
            else:
                # Define LFIRE object
                infobj = inference.LFIRE(
                    d, self.prior_samples, self.weights, self.simobj
                )
                utils, _ = infobj.ratios(numsamp=numsamp)

            self.utils = np.array(utils)
            # self.coefs = coefs
        else:
            raise NotImplementedError()

        if evaltype == "mean":
            mutualinfo = self._mean_eval(self.utils)
        elif evaltype == "median":
            mutualinfo = self._median_eval(self.utils)
        elif evaltype == "robust":
            mutualinfo = self._robust_eval(self.utils)
        else:
            raise NotImplementedError()

        return mutualinfo

    def compute_final(self, d_opt, y_obs, num_cores=1):

        """
        Computes the LFIRE ratios and coefficients given some real-world
        observations.

        Parameters
        ----------
        d_opt: np.ndarray
            Optimal design after optimisation.
        y_obs: np.darray
            Real-world observation at 'd_opt'.
        num_cores: int
            Number of CPU cores to be used to evaluate the ratios.
            (default is 1)
        """

        if self.evalmethod == "lfire":

            if self.smc:
                # Define LFIRE object
                infobj = inference.LFIRE_SMC(
                    d_opt, self.prior_samples, self.weights, self.simobj
                )
            else:
                # Define LFIRE object
                infobj = inference.LFIRE(
                    d_opt, self.prior_samples, self.weights, self.simobj
                )

            # Take summary statistics of observed data
            if len(y_obs.shape) > 2:
                psi_obs = self.simobj.summary(y_obs)
            else:
                psi_obs = self.simobj.summary(y_obs.reshape(1, -1))

            # Compute coefficients for each prior sample
            tmp_bl = Parallel(n_jobs=int(num_cores))(
                delayed(infobj._logistic_regression)(p) for p in self.prior_samples
            )
            self.b_obs = np.array(tmp_bl)

            # Compute ratios for each coefficient
            self.r_obs = np.array(
                [
                    np.exp(psi_obs.reshape(1, -1) @ b[1:] + b[0])[0][0]
                    for b in self.b_obs
                ]
            )

            return self.r_obs, self.b_obs
        else:
            raise NotImplementedError()


class DOptimality(Utility):

    """
    Child class that implements the Bayesian D-Optimality utility function.

    Attributes
    ----------
    prior_samples: np.ndarray
        Array of prior parameter samples.
    weights: np.ndarray
        Array of weights corresponding to each prior sample.
    simobj: simulator object
        Object of the implicit simulator model.
    evalmethod: str
        Method to evaluate the density ratio. Currently only ABC is
        implemented.
    utils: np.ndarray
        Array containing the utility evaluations for each marginal data sample.
        The final utility evaluation is the expectation of this.

    Methods
    -------
    abc_likelihood:
        Evaluates the ABC likelihood.
    compute:
        Computes the Bayesian D-Optimality utility via ABC.
    compute_final:
        Final likelihood-free inference for the optimal design. Produces
        particle set of model parameters and posterior weights.
    """

    def __init__(self, prior_samples, weights, simobj, evalmethod="abc"):

        """
        Parameters
        ----------
        prior_samples: np.ndarray
            Array of prior parameter samples.
        weights: np.ndarray
            Array of weights corresponding to each prior sample.
        simobj: simulator object
            Object of the implicit simulator model.
        evalmethod: str
            Method to evaluate the density ratio. Currently only ABC is
            implemented.
            (default is 'abc')
        """

        self.evalmethod = evalmethod

        super().__init__(prior_samples, weights, simobj)

    def _mean_eval(self, U):
        return np.mean(U)

    def _median_eval(self, U):
        return np.median(U)

    def _robust_eval(self, U):
        return robust.MEstimator(U)

    def _precision(self, samples):

        """Returns the precision of 'samples'."""

        cov = np.cov(samples)

        # univariate example
        if cov.size == 1:
            prec = 1 / cov
        else:
            det = np.linalg.det(cov)
            prec = 1 / det

        return prec

    def abc_likelihood(self, data, y0, epsilon=0.1, summary=True):

        """
        Compute ABC likelihood used to compute D-Optimality.

        Parameters
        ----------
        data: np.npdarray
            Array of several (simulated) data points
        y0: np.ndarray
            Single, observed data point.
        epsilon: float >= 0
            ABC threshold for acceptance.
            (default is 0.1)
        summary: Boolean
            Compute the ABC likelihood using summary statistics and not just
            raw data. Useful when dimensions of y are high.
            (default is True)
        """

        # check if you want to use summary stats
        if summary:
            data = self.simobj.summary(data)
            y0 = self.simobj.summary(y0)
        else:
            pass

        # Check for correct dimensions
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Compute distances between data and observation
        dist = np.linalg.norm(y0 - data, axis=1)

        # Count data that is sufficiently close
        count = np.mean(dist <= epsilon)

        return count

    def compute(
            self, d, M=1000, epsilon=0.1, evaltype="robust", verbose=True,
            summary=True
    ):

        """
        Computes the Bayesian D-Optimality utility via ABC.

        Parameters
        ----------
        d: np.ndarray
            Design at which to evaluate the BD-Opt utility.
        M: int
            Number of prior samples in ABC.
            (default is 1000)
        epsilon: float >= 0
            ABC threshold for acceptance.
            (default is 0.1)
        evaltype: str
            Type of Monte-Carlo evaluation; can currently take 'mean', 'median'
            or 'robust' as input.
            (default 'robust')
        verbose: Boolean
            If True, prints the current design to stdout.
            (default is True)
        summary: Boolean
            Compute the ABC likelihood using summary statistics and not just
            raw data. Useful when dimensions of y are high.
            (default is True)
        """

        # sample K data points from the current particle set
        K = len(self.weights)

        if np.sum(self.weights) == 0:
            raise ValueError("Your weights are all zero!")
        w_k = self.weights / np.sum(self.weights)
        p_k = list()
        for _ in range(K):
            cat = np.random.choice(range(len(w_k)), p=w_k)
            p_k.append(self.prior_samples[cat])
        p_k = np.array(p_k)

        # Simulate corresponding data
        y_k = self.simobj.sample_data(d, p_k)

        if verbose:
            print("Design point: ", d)

        if self.evalmethod == "abc":

            # Define ABC object + distance measure

            us = list()
            for idx, y in enumerate(y_k):

                w_post = list()
                for i in range(len(self.weights)):

                    # Simulate data from prior at theta_i
                    y_i = self.simobj.sample_data(
                        d, self.prior_samples[i], num=M)

                    # compute ABC likelihood
                    abc_like = self.abc_likelihood(
                        y_i, y, epsilon=epsilon, summary=summary
                    )
                    # Compute posterior new weights
                    wi_new = self.weights[i] * abc_like
                    w_post.append(wi_new)
                w_post = np.array(w_post)

                # Generate posterior samples
                if np.sum(w_post) == 0:
                    raise ValueError("Your weights are all zero!")
                wp_norm = w_post / np.sum(w_post)
                p_post = list()
                for _ in range(len(self.prior_samples)):
                    cat = np.random.choice(range(len(wp_norm)), p=wp_norm)
                    p_post.append(self.prior_samples[cat])
                p_post = np.array(p_post)

                # Compute precision for current posterior samples
                u = self._precision(p_post)
                us.append(u)

            self.utils = np.array(us)

        else:
            raise NotImplementedError()

        if evaltype == "mean":
            doptimality = self._mean_eval(self.utils)
        elif evaltype == "median":
            doptimality = self._median_eval(self.utils)
        elif evaltype == "robust":
            doptimality = self._robust_eval(self.utils)
        else:
            raise NotImplementedError()

        spread = True
        if spread:
            return doptimality, self.utils
        else:
            return doptimality

        # return doptimality

    def compute_final(
        self, d_opt, y_obs, M=1000, num_cores=1, epsilon=0.1, summary=True
    ):

        """
        Final likelihood-free inference for the optimal design. Produces
        particle set of model parameters and posterior weights.

        Parameters
        ----------
        d_opt: np.ndarray
            Optimal design after optimisation.
        y_obs: np.darray
            Real-world observation at 'd_opt'.
        M: int
            Number of prior samples in ABC.
            (default is 1000)
        num_cores: int
            Number of CPU cores to be used to evaluate the ratios.
            (default is 1)
        epsilon: float >= 0
            ABC threshold for acceptance.
            (default is 0.1)
        summary: Boolean
            Compute the ABC likelihood using summary statistics and not just
            raw data. Useful when dimensions of y are high.
            (default is True)
        """

        if self.evalmethod == "abc":

            w_post = list()
            for i in range(len(self.weights)):

                # Simulate data from prior at theta_i
                y_i = self.simobj.sample_data(
                    d_opt, self.prior_samples[i], num=M)

                # compute ABC likelihood
                abc_like = self.abc_likelihood(
                    y_i, y_obs, epsilon=epsilon, summary=summary
                )
                # Compute posterior new weights
                wi_new = self.weights[i] * abc_like
                w_post.append(wi_new)
            w_post = np.array(w_post)

            # Normalise weights
            w_post_norm = w_post / np.sum(w_post)

        else:
            raise NotImplementedError()

        return self.prior_samples, w_post_norm
