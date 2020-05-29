from scipy.stats import truncnorm

import seqbed.simulator as simulator
import seqbed.sequentialdesign as sequentialdesign

# ---- GLOBAL PARAMS ------ #

# Number of CPU cores
num_cores = 2
# Set dimensions of design variable
DIMS = 1  # Keep at 1 for myopic BED

# Number of prior samples
NS = 1000
# Max number of utility evaluations in B.O. (per core)
MAX_ITER = 5

# number of initial data points for B.O.
if num_cores > 5:
    INIT_NUM = num_cores
else:
    INIT_NUM = 5

# ----- SPECIFY MODEL ----- #

# Obtain Death model prior samples
mu, sigma = 1, 1
lower, upper = 0, 50
trunc = truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
prior_death = trunc.rvs(size=NS)

# Define the domain for BO, according to GPyOpt standards
domain_death = [
    {
        "name": "var_1",
        "type": "continuous",
        "domain": (0.01, 4.00),
        "dimensionality": int(DIMS),
    }
]

# Define the constraints for BO
# Time cannot go backwards
if DIMS == 1:
    constraints_death = None
elif DIMS > 1:
    constraints_death = list()
    for i in range(1, DIMS):
        dic = {
            "name": "constr_{}".format(i),
            "constraint": "x[:,{}]-x[:,{}]".format(i - 1, i),
        }
        constraints_death.append(dic)
else:
    raise ValueError()

# ----- RUN MODEL ----- #

# Define the simulator model
truth_death = 1.5
if DIMS == 1:
    model_death = simulator.DeathModel(truth_death, 50)
else:
    model_death = simulator.DeathModelMultiple(truth_death, 50)
bounds_death = [[0.0, 50]]

# Define the SequentialBED object
foo = "../data/deathmodel_seq"
BED_death = sequentialdesign.SequentialBED(
    prior_death,
    model_death,
    domain=domain_death,
    constraints=constraints_death,
    num_cores=num_cores,
    utiltype="MI",
)

# Run the actual optimisation and save data
# Data is saved as '../data/deathmodel_seq_iter1.npz'
BED_death.optimisation(
    n_iter=1,
    BO_init_num=INIT_NUM,
    BO_max_iter=MAX_ITER,
    filn=foo,
    obs_file=None,
    bounds=bounds_death
)
