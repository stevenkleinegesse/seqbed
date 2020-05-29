import numpy as np

import sys
import os

# own libraries
import seqbed.utility as utility
import seqbed.methods as methods
import seqbed.robust as robust

# for Bayesian Optimization
from GPyOpt.methods import BayesianOptimization


class SequentialBED:

    """
    Class that performs sequential Bayesian experimental design for a given
    simulator object and and a set of prior samples.

    Attributes
    ----------
    prior_samples: np.ndarray
        Array of prior parameter samples.
    simobj: simulator object
        Object of the implicit simulator model.
    domain: list of dict
        List that contains a dictionary with information about the domain of
        the design parameter, according to GPyOpt standards.
    constraints: list of dict
        List that contains a dictionary with information about the constraints
        on the domain of the design parameter, according to GPyOpt standards.
    num_cores: int
        Number of cores to be used during the Bayesian Optimisation step and
        the final likelihood-free inference.
    utiltype: str
        Choose which utility to use to solve the sequential BED problem.
        Currently only 'MI' (mutual information) and 'Precision' (BD-Opt) are
        implemented.
    prior_pdf: np.ndarray
        Only relevant for the BD-Opt utility.
        Array of prior densities for the provided 'prior_samples'.
    epsilon: float >= 0
        Only relevant for the BD-Opt utility.
        ABC acceptance threshold.
    summary: Boolean
        Only relevant for the BD-Opt utility.
        Compute the ABC likelihood using summary statistics and not just raw
        data. Useful when dimensions of y are high.
    smc: Boolean
        Only relevant for the MI utility.
        Choose whether or not to use SMC methods to evaluate the utility. If
        'True' a single ratio is computed for each prior sample. If 'False',
        several ratios may be computed for a single prior sample.
    evaltype: str
        Type of Monte-Carlo evaluation; can currently take 'mean', 'median' or
        'robust' as input.
    weights: np.ndarray
        Array of weights corresponding to the prior samples. Before
        optimisation() is called, these are the weights used during the
        computation of the utility. After calling optimisation() the weights
        correspond the ratio of posterior and prior density.
    likenum: int
        Only relevant for the BD-Opt utility.
        Number of prior samples to be used in ABC.
    data_d: list
        List that contains all optimal designs.
    data_y: list
        List that contains all real-world observations.
    iter_start: int
        Initial iteration number; set to 0.
    utilobj: utility object
        Object of the utility class, either utility.MutualInformation or
        utility.DOptimality.
    bo_obj: GPyOpt.methods.BayesianOptimization object
        GPyOpt object that contains relevant information about the BO step of
        the sequential BED algorithm.
    savedata: dict
        Dictionary containing all the data that should be saved.

    Methods
    -------
    optimisation:
        Runs sequential Bayesian experimental design by optimising the provided
        utility function.
    save:
        Saves all relevant containers and results to a file.
    """

    def __init__(
        self,
        prior_samples,
        simobj,
        domain,
        constraints=None,
        num_cores=1,
        utiltype="MI",
        prior_pdf=None,
        epsilon=None,
        summary=True,
        smc=False,
        evaltype="mean",
    ):

        """
        Parameters
        ----------
        prior_samples: np.ndarray
            Array of prior parameter samples.
        simobj: simulator object
            Object of the implicit simulator model.
        domain: list of dict
            List that contains a dictionary with information about the domain
            of the design parameter, according to GPyOpt standards.
        constraints: list of dict
            List that contains a dictionary with information about the
            constraints on the domain of the design parameter, according to
            GPyOpt standards.
        num_cores: int
            Number of cores to be used during the Bayesian Optimisation step
            and the final likelihood-free inference.
            (default is 1)
        utiltype: str
            Choose which utility to use to solve the sequential BED problem.
            Currently only 'MI' (mutual information) and 'Precision' (BD-Opt)
            are implemented.
            (default is 'MI')
        prior_pdf: np.ndarray
            Only relevant for the BD-Opt utility.
            Array of prior densities for the provided 'prior_samples'.
            (default is None)
        epsilon: float >= 0
            Only relevant for the BD-Opt utility.
            ABC acceptance threshold.
            (default is None)
        summary: Boolean
            Only relevant for the BD-Opt utility.
            Compute the ABC likelihood using summary statistics and not just
            raw data. Useful when dimensions of y are high.
            (default is True)
        smc: Boolean
            Only relevant for the MI utility.
            Choose whether or not to use SMC methods to evaluate the utility.
            If 'True' a single ratio is computed for each prior sample. If
            'False', several ratios may be computed for a single prior sample.
            (default is False)
        evaltype: str
            Type of Monte-Carlo evaluation; can currently take 'mean', 'median'
            or 'robust' as input.
            (default is 'mean')
        """

        self.prior_samples = prior_samples
        self.simobj = simobj
        self.domain = domain
        self.constraints = constraints
        self.num_cores = num_cores
        self.evaltype = evaltype  # mean, median or robust

        # choose the utility used in the optimisation
        self.utiltype = utiltype
        if self.utiltype == "MI":
            # set uniform weights
            self.weights = np.ones(len(self.prior_samples))
            self.smc = smc
        elif self.utiltype == "Precision":
            # set uniform weights
            self.weights = np.ones(len(self.prior_samples))
            # Define prior density function
            self.prior_pdf = prior_pdf
            # set ABC params
            self.epsilon = epsilon
            self.likenum = len(self.prior_samples)
            self.summary = summary
            # Define data containers
            self.data_d = list()
            self.data_y = list()
        else:
            raise NotImplementedError()

        # for iterations if checkpointed
        self.iter_start = 0

    def _ESS(self):

        """
        Compute the effective sample size for the current set of weights.
        """

        mom1 = np.sum(np.array(self.weights))
        mom2 = np.sum(np.array(self.weights) ** 2)

        return mom1 ** 2 / mom2

    def _objective(self, d):

        """
        Evalute the utility at the current design 'd'.
        """

        if self.utiltype == "MI":
            u = -self.utilobj.compute(
                d, numsamp=10 * len(self.prior_samples), evaltype=self.evaltype
            )
        elif self.utiltype == "Precision":
            u = -self.utilobj.compute(
                d[0],
                epsilon=self.epsilon,
                M=self.likenum,
                evaltype=self.evaltype,
                summary=self.summary,
            )
        return u

    def optimisation(
        self,
        n_iter=1,
        BO_init_num=5,
        BO_max_iter=10,
        filn="./data",
        obs_file=None,
        bounds=None,
    ):

        """
        Runs sequential Bayesian experimental design by optimising the provided
        utility function. When finished, saves all the relevant data.

        Parameters
        ----------
        n_iter: int
            The current iteration of the sequential BED process.
            (default is 1)
        BO_init_num: int
            Initial number of BO evaluations.
            (default is 5)
        BO_max_iter: int
            Maximum number of BO evaluations, in addition to 'BO_init_num'.
            (default is 10)
        filn: str
            Filename, indicating where to save the relevant data.
            (default is ./data)
        obs_file: None or str
            Option to use a file with previous observations at all designs.
            (default is None)
        bounds: None or np.ndarray
            Bounds for the resampling algorithm. If None, the support of the
            prior distribution is infinite.
            (default is None)
        """

        for n in range(self.iter_start + 1, self.iter_start + n_iter + 1):

            print("Iteration {0}/{1}".format(n, self.iter_start + n_iter))
            print("")

            # define the utility object and resample
            if self.utiltype == "MI":

                self.utilobj = utility.MutualInformation(
                    self.prior_samples,
                    self.weights,
                    self.simobj,
                    evalmethod="lfire",
                    smc=self.smc,
                )

                # Check if you need to resample
                ESS = self._ESS()
                if ESS < 0.50 * len(self.prior_samples):
                    print("Resampling!")
                    pp_new, ww_new = methods.resampling(
                        self.prior_samples,
                        self.weights,
                        nsamples=len(self.prior_samples),
                        bounds=bounds,
                    )
                    self.prior_samples = pp_new
                    self.utilobj.prior_samples = pp_new
                    self.weights = ww_new
                else:
                    pass

            elif self.utiltype == "Precision":

                self.utilobj = utility.DOptimality(
                    self.prior_samples, self.weights, self.simobj
                )

                # Check if you need to resample
                ESS = self._ESS()
                if ESS < 0.50 * len(self.prior_samples):
                    print("Resampling!")
                    pp_new, ww_new = methods.ABC_resampling(
                        self.data_d,
                        self.data_y,
                        self.simobj,
                        self.prior_samples,
                        self.weights,
                        self.prior_pdf,
                        e0=self.epsilon,
                        n_iter=n,
                    )
                    self.prior_samples = pp_new
                    self.utilobj.prior_samples = pp_new
                    self.weights = ww_new
                else:
                    pass

            # Define GPyOpt Bayesian Optimization object
            dom = self.domain
            con = self.constraints
            myBopt = BayesianOptimization(
                f=self._objective,
                domain=self.domain,
                constraints=self.constraints,
                acquisition_type="EI",
                normalize_Y=True,
                initial_design_numdata=BO_init_num,
                evaluator_type="local_penalization",
                batch_size=int(self.num_cores),
                num_cores=int(self.num_cores),
                acquisition_jitter=0.01,
            )

            # run the bayesian optimisation
            myBopt.run_optimization(max_iter=BO_max_iter)
            self.bo_obj = myBopt

            # Select method to get optimum
            # optmethod='point' # take optimum from BO evaluations
            optmethod = "interpol"  # use posterior predictive to find optimum
            if optmethod == "point":
                d_opt = self.bo_obj.x_opt
            elif optmethod == "interpol":
                d_opt = methods.get_GP_optimum(self.bo_obj)
            else:
                raise NotImplementedError()

            # Take some real-world data at optimum
            if obs_file is None:
                y_obs = self.simobj.observe(d_opt)[0]
            else:  # ONLY WORKS FOR SEQUENTIAL DESIGN FOR NOW - NOT NON-MYOPIC
                obs_data = np.load("{}_iter{}.npz".format(obs_file, n))
                dd, yy = obs_data.f.dd, obs_data.f.yy
                tmin = np.abs(dd - d_opt)
                idx = np.where(tmin == np.min(tmin))[0][0]
                y_obs = yy[idx]

            if self.utiltype == "MI":

                # Compute ratios r_obs and coefficients b_obs for final obs
                r_obs, b_obs = self.utilobj.compute_final(
                    d_opt, y_obs, num_cores=self.num_cores
                )

                # Compute final utility value
                u_final = robust.MEstimator(np.log(r_obs))

                self.savedata = {
                    "d_opt": d_opt,
                    "y_obs": y_obs,
                    "r_obs": r_obs,
                    "b_obs": b_obs,
                    "u_final": u_final,
                    "neff": self._ESS(),
                    "weights": self.weights,
                }
                self.save("{}_iter{}".format(filn, n))

                # calculate new weights
                self.weights = self.weights * r_obs

            elif self.utiltype == "Precision":

                # Compute ABC posterior samples and according weights
                _, ww_post = self.utilobj.compute_final(
                    d_opt,
                    y_obs,
                    epsilon=self.epsilon,
                    M=self.likenum,
                    summary=self.summary,
                )

                # Generate posterior samples
                ww_post_norm = ww_post / np.sum(ww_post)
                pp_post = list()
                for _ in range(len(self.prior_samples)):
                    cat = np.random.choice(
                        range(len(ww_post_norm)), p=ww_post_norm)
                    pp_post.append(self.prior_samples[cat])
                pp_post = np.array(pp_post)

                # Compute precision for current posterior samples
                u_final = self.utilobj._precision(pp_post)

                # Add data to containers
                self.data_d.append(d_opt)
                self.data_y.append(y_obs)

                # Save data
                self.savedata = {
                    "d_opt": d_opt,
                    "y_obs": y_obs,
                    "w_post": ww_post,
                    "u_final": u_final,
                    "neff": self._ESS(),
                    "weights": self.weights,
                    "data_d": self.data_d,
                    "data_y": self.data_y,
                }
                self.save("{}_iter{}".format(filn, n))

                # Update weights
                self.weights = ww_post

            else:
                raise NotImplementedError()

        print("")

    def save(self, fn):

        """
        Saves all relevant containers and results to a file.

        Parameters
        ----------
        fn: str
            Save file location and name.
        """

        # Save BayesOpt evals
        X_eval = self.bo_obj.X
        Y_eval = self.bo_obj.Y
        # If design dim == 1, save mean and variance
        if self.domain[0]["dimensionality"] == 1:
            if self.domain[0]["type"] == "continuous":
                bounds = self.bo_obj.space.get_bounds()
                x_grid = np.linspace(bounds[0][0], bounds[0][1], 1000)
            elif self.domain[0]["type"] == "discrete":
                x_grid = np.array(self.domain[0]["domain"])
            #                x_grid = x_grid.reshape(len(x_grid),1)
            x_grid = x_grid.reshape(len(x_grid), 1)
            m, v = self.bo_obj.model.predict(x_grid)
        # Store data
        gpyopt_data = [X_eval, Y_eval, x_grid, m, v]

        np.savez(
            "{}.npz".format(fn),
            **self.savedata,
            utilobj=self.utilobj,
            prior_samples=self.prior_samples,
            gpyopt_data=gpyopt_data
        )


class SequentialBED_checkpoint(SequentialBED):

    """
    Child class that performs sequential Bayesian experimental design for a
    given simulator object and and a set of prior samples. Uses checkpointed
    data from previous sequential design evaluations given the appropriate
    files.

    Attributes
    ----------
    fname: str
        Filename of checkpointed file.
    simobj: simulator object
        Object of the implicit simulator model.
    domain: list of dict
        List that contains a dictionary with information about the domain of
        the design parameter, according to GPyOpt standards.
    constraints: list of dict
        List that contains a dictionary with information about the constraints
        on the domain of the design parameter, according to GPyOpt standards.
    num_cores: int
        Number of cores to be used during the Bayesian Optimisation step and
        the final likelihood-free inference.
    utiltype: str
        Choose which utility to use to solve the sequential BED problem.
        Currently only 'MI' (mutual information) and 'Precision' (BD-Opt) are
        implemented.
    prior_pdf: np.ndarray
        Only relevant for the BD-Opt utility.
        Array of prior densities for the provided 'prior_samples'.
    epsilon: float >= 0
        Only relevant for the BD-Opt utility.
        ABC acceptance threshold.
    summary: Boolean
        Only relevant for the BD-Opt utility.
        Compute the ABC likelihood using summary statistics and not just raw
        data. Useful when dimensions of y are high.
    smc: Boolean
        Only relevant for the MI utility.
        Choose whether or not to use SMC methods to evaluate the utility. If
        'True' a single ratio is computed for each prior sample. If 'False',
        several ratios may be computed for a single prior sample.
    evaltype: str
        Type of Monte-Carlo evaluation; can currently take 'mean', 'median' or
        'robust' as input.
    weights: np.ndarray
        Array of weights corresponding to the prior samples. Before
        optimisation() is called, these are the weights used during the
        computation of the utility. After calling optimisation() the weights
        correspond the ratio of posterior and prior density.
    likenum: int
        Only relevant for the BD-Opt utility.
        Number of prior samples to be used in ABC.
    data_d: list
        List that contains all optimal designs.
    data_y: list
        List that contains all real-world observations.
    iter_start: int
        Initial iteration number; set to 0.
    """

    def __init__(
        self,
        fname,
        simobj,
        domain,
        constraints=None,
        num_cores=1,
        utiltype="MI",
        prior_pdf=None,
        epsilon=None,
        summary=True,
        smc=False,
        evaltype="mean",
    ):

        """
        Parameters
        ----------
        fname: str
            Filename of checkpointed file.
        simobj: simulator object
            Object of the implicit simulator model.
        domain: list of dict
            List that contains a dictionary with information about the domain
            of the design parameter, according to GPyOpt standards.
        constraints: list of dict
            List that contains a dictionary with information about the
            constraints on the domain of the design parameter, according to
            GPyOpt standards.
        num_cores: int
            Number of cores to be used during the Bayesian Optimisation step
            and the final likelihood-free inference.
            (default is 1)
        utiltype: str
            Choose which utility to use to solve the sequential BED problem.
            Currently only 'MI' (mutual information) and 'Precision' (BD-Opt)
            are implemented.
            (default is 'MI')
        prior_pdf: np.ndarray
            Only relevant for the BD-Opt utility.
            Array of prior densities for the provided 'prior_samples'.
            (default is None)
        epsilon: float >= 0
            Only relevant for the BD-Opt utility.
            ABC acceptance threshold.
            (default is None)
        summary: Boolean
            Only relevant for the BD-Opt utility.
            Compute the ABC likelihood using summary statistics and not just
            raw data. Useful when dimensions of y are high.
            (default is True)
        smc: Boolean
            Only relevant for the MI utility.
            Choose whether or not to use SMC methods to evaluate the utility.
            If 'True' a single ratio is computed for each prior sample. If
            'False', several ratios may be computed for a single prior sample.
            (default is False)
        evaltype: str
            Type of Monte-Carlo evaluation; can currently take 'mean', 'median'
            or 'robust' as input.
            (default is 'mean')
        """

        # Load data from filename
        self.fname = fname
        self.data = np.load("{}.npz".format(self.fname))
        # Assign prior samples
        self.prior_samples = self.data["prior_samples"]

        super().__init__(
            self.prior_samples,
            simobj,
            domain,
            constraints,
            num_cores,
            utiltype,
            prior_pdf,
            epsilon,
            evaltype=evaltype,
        )

        # Do stuff with the utility type
        if self.utiltype == "MI":
            # Assign weights from data
            self.weights = self.data["weights"] * self.data["r_obs"]
            self.smc = smc
        elif self.utiltype == "Precision":
            # Assign weights from data
            self.weights = self.data["w_post"]
            # Define prior density function
            self.prior_pdf = prior_pdf
            # set ABC params
            self.epsilon = epsilon
            self.likenum = len(self.prior_samples)
            self.summary = summary
            # Assign container data
            self.data_d = list(self.data["data_d"])
            self.data_y = list(self.data["data_y"])

        else:
            raise NotImplementedError()

        # Select which iteration we are at
        self.iter_start = int(self.fname[-1])
