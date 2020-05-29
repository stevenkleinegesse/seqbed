import numpy as np
import warnings


class Simulator:

    """
    Simulator base class for simulating data from different models.

    Attributes
    ----------
    truth: np.ndarray
        Ground truth of the model parameters used in the observe() method.

    Methods
    -------
    summary:
        Computes the summary statistics of simulated data.
    generate_data:
        The forward simulation of the implicit simulator model.
    sample_data:
        Used to obtain samples from the likelihood or marginal.
    observe:
        Make a real-world observation based on the ground 'truth'.
    """

    def __init__(self, truth):

        """
        Parameters
        ----------
        truth: np.ndarray
            Ground truth of the model parameters used in the observe() method.
        """

        self.truth = np.array(truth)

    def summary(self, Y):

        """
        Computes the summary statistics of the provided data. Default is
        simply powers from 1 to 3 of the data values; this is only applicable
        to scalars.

        Parameters
        ----------
        Y: np.ndarray
            Data simulated from the model.
        """

        Y_psi = list()
        for y in Y:
            Y_psi.append([y ** i for i in range(1, 4)])
        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        The forward simulation of the implicit simulator model. Needs to be
        specified.

        Parameters
        ----------
        d: np.ndarray
            The design at which to simulate data.
        p: np.ndarray
            Array of model parameters for which to simulate data.
        """

        pass

    def sample_data(self, d, p, num=None):

        """
        Sample data from the simulator model, based on the generate_data()
        method. The point of this method is to select if to sample from the
        likelihood (if num!=None and len(p)==1) or marginal
        (if num==None and len(p) > 1)

        Parameters
        ----------
        d: np.ndarray
            The design at which to simulate data.
        p: np.ndarray
            Array of model parameters for which to simulate data.
        num: Boolean or int
           Number of samples to produce. If num==None, samples from the
           likelihood and samples from the marginal if otherwise.
           (default is None)
        """

        # sample from an array of params
        if num is None:
            y = np.array([self.generate_data(d, pi) for pi in p])
        # sample several times using the same params:
        else:
            y = np.array([self.generate_data(d, p) for i in range(num)])
        return y

    def observe(self, d, num=1):

        """
        Observed some data according to a ground truth.

        Parameters
        ----------
        d: np.ndarray
            Optimal design at which to make measurements.
        num: int
            Number of observations to make at optimal design.
        """

        y = np.array([self.generate_data(d, self.truth) for i in range(num)])
        return y


class DeathModel(Simulator):

    """
    Class to simulate data according to the Death Model.

    Attributes
    ----------
    truth: np.ndarray
        Ground truth of the model parameters used in the observe() method.
    S0: int
        Initial number of susceptibles in the population.

    Methods
    -------
    summary:
        Computes the summary statistics of simulated data.
    generate_data:
        The forward simulation of the implicit simulator model.
    """

    def __init__(self, truth, S0):

        """
        Parameters
        ----------
        truth: np.ndarray
            Ground truth of the model parameters used in the observe() method.
        S0: int
            Initial number of susceptibles in the population.
        """

        super(DeathModel, self).__init__(truth)
        self.S0 = S0

    def summary(self, Y):

        """
        Computes the summary statistics of the provided data.

        Parameters
        ----------
        Y: np.ndarray
            Data simulated from the model.
        """

        Y_psi = list()
        for arr in Y:
            if np.array(arr).shape == ():
                tmp = [arr ** i for i in range(1, 4)]
            else:
                tmp = [arr[0] ** i for i in range(1, 4)]
            Y_psi.append(tmp)
        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        The forward simulation of the implicit simulator model.

        Parameters
        ----------
        d: np.ndarray
            The design at which to simulate data.
        p: np.ndarray
            Model parameters for which to simulate data.
        """

        inf_num = np.random.binomial(self.S0, 1 - np.exp(-p * d))
        return inf_num


class DeathModelMultiple(Simulator):

    """
    Class to simulate data according to the Death Model. Used in (non-myopic)
    cases where population observations are needed at several design times.

    Attributes
    ----------
    truth: np.ndarray
        Ground truth of the model parameters used in the observe() method.
    S0: int
        Initial number of susceptibles in the population.

    Methods
    -------
    summary:
        Computes the summary statistics of simulated data.
    generate_data:
        The forward simulation of the implicit simulator model.
    """

    def __init__(self, truth, S0):

        """
        Parameters
        ----------
        truth: np.ndarray
            Ground truth of the model parameters used in the observe() method.
        S0: int
            Initial number of susceptibles in the population.
        """

        super(DeathModelMultiple, self).__init__(truth)
        self.S0 = S0

    def summary(self, Y):

        """
        Computes the summary statistics of the provided data.

        Parameters
        ----------
        Y: np.ndarray
            Data simulated from the model.
        """

        Y_psi = list()
        ind = 0
        for arr in Y:
            if len(arr) == 1:
                tmp = [arr[0] ** i for i in range(1, 4)]
            else:
                tmp = list()
                for i in range(1, 4):
                    tmp.append(arr ** i)
                tmp = np.array(tmp).flatten()
            Y_psi.append(tmp)
            ind += 1
        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        The forward simulation of the implicit simulator model.

        Parameters
        ----------
        d: np.ndarray
            Array of the designs at which to simulate data.
        p: np.ndarray
            Model parameters for which to simulate data.
        """

        infected = list()
        d0 = 0
        I0 = 0
        if isinstance(d, float):
            inf_num = np.random.binomial(
                self.S0 - I0, 1 - np.exp(-p * d))
            infected.append(inf_num)
        else:
            for idx in range(len(d)):
                if d[idx] < d0:
                    raise ValueError("You can't go backwards in time!")
                inf_num = I0 + np.random.binomial(
                    self.S0 - I0, 1 - np.exp(-p * (d[idx] - d0))
                )
                infected.append(inf_num)
                d0 = d[idx]
                I0 = inf_num

        return np.array(infected)


class SIRModel(Simulator):

    """
    Class to simulate data according to the Susceptible-Infected-Recovered
    (SIR) model.

    Attributes
    ----------
    truth: np.ndarray
        Ground truth of the model parameters used in the observe() method.
    N: int
        Total number of individuals in the population.
    S0: int
        Initial number of susceptibles in the population.
    I0: int
        Initial number of infected in the population.
    R0: int
        Initial number of recovered in the population.
    sumtype: str
        What type of summary statistics are used.

    Methods
    -------
    summary:
        Computes the summary statistics of simulated data.
    generate_data:
        The forward simulation of the implicit simulator model.
    """

    def __init__(self, truth, N, sumtype="linear"):

        """
        Parameters
        ----------
        truth: np.ndarray
            Ground truth of the model parameters used in the observe() method.
        N: int
            Total number of individuals in the population.
        sumtype: str
            The type of summary statistics that are used. Currently, only
            'linear' (S, I, R) and 'all' are supported (all combinations of
            S, I and R up to power of 3 --> recommend).
            (default is 'linear')
        """

        super(SIRModel, self).__init__(truth)
        self.N = N
        self.S0 = N - 1
        self.I0 = 1
        self.R0 = 0
        self.sumtype = sumtype

    def summary(self, Y):

        """
        Computes the summary statistics of the provided data.

        Parameters
        ----------
        Y: np.ndarray
            Data simulated from the model.
        """

        Y_psi = list()
        for arr in Y:
            S = arr[0]
            I = arr[1]
            if self.sumtype == "linear":
                Y_psi.append([S, I, arr[2]])
            else:
                Y_psi.append(
                    [
                        S,
                        I,
                        S * I,
                        S ** 2,
                        I ** 2,
                        S ** 2 * I,
                        S * I ** 2,
                        S ** 3,
                        I ** 3,
                    ]
                )

        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        The forward simulation of the implicit simulator model.

        Parameters
        ----------
        d: np.ndarray
            The design at which to simulate data.
        p: np.ndarray
            Model parameters for which to simulate data.
        """

        dt = 0.01
        times = np.arange(0 + dt, d + dt, dt)

        S = self.S0
        I = self.I0
        R = self.R0

        for _ in times:

            pinf = p[0] * I / self.N
            dI = np.random.binomial(S, pinf)

            precov = p[1]
            dR = np.random.binomial(I, precov)

            S = S - dI
            I = I + dI - dR
            R = R + dR

        return np.array([S, I, R])


class SIRModelMultiple(Simulator):

    """
    Class to simulate data according to the Susceptible-Infected-Recovered
    (SIR) model. Used in (non-myopic) cases where population observations are
    needed at several design times.

    Attributes
    ----------
    truth: np.ndarray
        Ground truth of the model parameters used in the observe() method.
    N: int
        Total number of individuals in the population.
    S0: int
        Initial number of susceptibles in the population.
    I0: int
        Initial number of infected in the population.
    R0: int
        Initial number of recovered in the population.
    sumtype: str
        What type of summary statistics are used.

    Methods
    -------
    summary:
        Computes the summary statistics of simulated data.
    generate_data:
        The forward simulation of the implicit simulator model.
    """

    def __init__(self, truth, N):

        """
        Parameters
        ----------
        truth: np.ndarray
            Ground truth of the model parameters used in the observe() method.
        N: int
            Total number of individuals in the population.
        """

        super(SIRModelMultiple, self).__init__(truth)
        self.N = N
        self.S0 = N - 1
        self.I0 = 1
        self.R0 = 0

    def summary(self, Y):

        """
        Computes the summary statistics of the provided data.

        Parameters
        ----------
        Y: np.ndarray
            Data simulated from the model.
        """

        Y_psi = list()
        for arr in Y:

            # Y_psi.append(arr)

            flat = arr.flatten()
            Y_psi.append(flat)

        return np.array(Y_psi)

    def generate_data(self, d, p):

        """
        The forward simulation of the implicit simulator model.

        Parameters
        ----------
        d: np.ndarray
            Array of the designs at which to simulate data.
        p: np.ndarray
            Model parameters for which to simulate data.
        """

        dt = 0.01

        data = list()

        St = self.S0
        It = self.I0
        Rt = self.R0

        for tau in d:

            times = np.arange(0 + dt, tau + dt, dt)

            S = St
            I = It
            R = Rt

            for _ in times:

                pinf = p[0] * I / self.N
                # print(pinf)
                dI = np.random.binomial(S, pinf)

                precov = p[1]
                dR = np.random.binomial(I, precov)

                S = S - dI
                I = I + dI - dR
                R = R + dR

            y = [S, I, R]
            data.append(y)

            St = S
            It = I
            Rt = R

        return np.array(data)


class CellModel(Simulator):

    """
    Class to simulate data according to the Cell model of Vo et al. (2015).

    Attributes
    ----------
    truth: np.ndarray
        Ground truth of the model parameters used in the observe() method.
    N0: int
        Initial number of cells on the scratch assay
    num_total: int
        Number of images, i.e. discrete time steps that are taken.

    Methods
    -------
    summary:
        Computes the summary statistics of simulated data.
    generate_data:
        The forward simulation of the implicit simulator model.
    """

    def __init__(self, truth, N0, num_total):

        """
        Parameters
        ----------
        truth: np.ndarray
            Ground truth of the model parameters used in the observe() method.
        N0: int
            Initial number of cells on the scratch assay
        num_total: int
            Number of images, i.e. discrete time steps that are taken.
        """

        super(CellModel, self).__init__(truth)

        self.N0 = N0
        self.num_total = num_total

    def summary(self, Y):

        """
        Computes the summary statistics of the provided data.

        Parameters
        ----------
        Y: np.ndarray
            Data simulated from the model.
        """

        # For a single sample
        if len(Y.shape) == 3:

            # Hamming Distance
            Ydiff = Y[1:] - Y[:-1]
            s = np.sum(np.abs(Ydiff), axis=(1, 2))

            # Total number at end
            K = np.count_nonzero(Y[-1])

            summ = np.array([*s, K])

        # For several samples
        else:

            summ = list()
            for y in Y:

                # Hamming Distance
                Ydiff = y[1:] - y[:-1]
                s = np.sum(np.abs(Ydiff), axis=(1, 2))

                # Total number at end
                K = np.count_nonzero(y[-1])

                s = np.array([*s, K])

                summ.append(s)
            summ = np.array(summ)

        return summ

    def generate_data(self, d, p):

        """
        The forward simulation of the implicit simulator model.

        Parameters
        ----------
        d: np.ndarray
            The design at which to simulate data.
        p: np.ndarray
            Model parameter for which to simulate data.
        """

        Yinit = np.zeros((27, 36))

        Nfill = 0
        not_filled = True
        while Nfill < self.N0:
            r = np.random.choice(range(13))
            c = np.random.choice(range(36))
            if Yinit[r, c] == 0:
                Yinit[r, c] = 1
                Nfill += 1

        self.Yinit = Yinit

        N = np.count_nonzero(self.Yinit)
        num_motility, num_prolif = 0, 0

        Yobs = [self.Yinit]
        Y = np.array(self.Yinit)

        # simulate over discrete time steps
        for _ in range(self.num_total):

            rows, cols = np.where(Y == 1)

            Nevent = int(N)
            # potential motility event for each active cell
            for n in range(Nevent):

                # probability that motility event is happening
                if np.random.random_sample() < p[0]:

                    # select which cell to move
                    k = np.random.choice(range(rows.shape[0]))
                    rowk, colk = rows[k], cols[k]

                    # Random walk into one of the four directions
                    u = np.random.choice(range(4))
                    if u == 0:
                        rowk_prop = rowk - 1
                        colk_prop = colk
                    elif u == 1:
                        rowk_prop = rowk + 1
                        colk_prop = colk
                    elif u == 2:
                        rowk_prop = rowk
                        colk_prop = colk - 1
                    else:
                        rowk_prop = rowk
                        colk_prop = colk + 1

                    # Check if target position is within bounds
                    if (rowk_prop >= 0 and rowk_prop <= Yinit.shape[0] - 1) and (
                        colk_prop >= 0 and colk_prop <= Yinit.shape[1] - 1
                    ):

                        # Check if target position is empty and change values
                        if Y[rowk_prop, colk_prop] == 0:

                            num_motility += 1

                            Y[rowk, colk] = 0
                            Y[rowk_prop, colk_prop] = 1
                            rows[k] = rowk_prop
                            cols[k] = colk_prop

            # potential proliferation event for each active cell
            for n in range(Nevent):

                # probability that motility event is happening
                if np.random.random_sample() < p[1]:

                    # select which cell to move
                    k = np.random.choice(range(rows.shape[0]))
                    rowk, colk = rows[k], cols[k]

                    # Random walk into one of the four directions
                    u = np.random.choice(range(4))
                    if u == 0:
                        rowk_prop = rowk - 1
                        colk_prop = colk
                    elif u == 1:
                        rowk_prop = rowk + 1
                        colk_prop = colk
                    elif u == 2:
                        rowk_prop = rowk
                        colk_prop = colk - 1
                    else:
                        rowk_prop = rowk
                        colk_prop = colk + 1

                    # Check if target position is within bounds
                    if (rowk_prop >= 0 and rowk_prop <= Yinit.shape[0] - 1) and (
                        colk_prop >= 0 and colk_prop <= Yinit.shape[1] - 1
                    ):

                        # Check if target position is empty and change values
                        if Y[rowk_prop, colk_prop] == 0:

                            num_prolif += 1
                            N += 1

                            Y[rowk_prop, colk_prop] = 1
                            rows = np.append(rows, rowk_prop)
                            cols = np.append(cols, colk_prop)

            y = np.array(Y)
            Yobs.append(y)

        Yobs = np.array(Yobs)
        Yobs = Yobs.astype(np.int16)

        Yd = np.array([self.Yinit] + [Yobs[int(didx)] for didx in d])
        return Yd
