import numpy as np

import seqbed.simulator as simulator
import seqbed.sequentialdesign as sequentialdesign

# ---- GLOBAL PARAMS ------ #

# Number of CPU cores
num_cores = 2

# Set dimensions of design variable
DIMS = 1  # Keep at 1 for myopic BED

# Number of prior samples (higher => more accurate posterior)
NS = 1000
# Max number of utility evaluations in B.O. (per core)
MAX_ITER = 5

# number of initial data points for B.O.
if num_cores > 5:
    INIT_NUM = num_cores
else:
    INIT_NUM = 5

# ----- SPECIFY MODEL ----- #

# Obtain model prior samples
param_0 = np.random.uniform(0, 0.5, NS).reshape(-1, 1)
param_1 = np.random.uniform(0, 0.5, NS).reshape(-1, 1)
params = np.hstack((param_0, param_1))

# Define the domain for BO
domain = [
    {
        "name": "var_1",
        "type": "continuous",
        "domain": (0.01, 3.00),
        "dimensionality": int(DIMS),
    }
]

# Define the constraints for BO
# Time cannot go backwards
if DIMS == 1:
    constraints = None
elif DIMS > 1:
    constraints = list()
    for i in range(1, DIMS):
        dic = {
            "name": "constr_{}".format(i),
            "constraint": "x[:,{}]-x[:,{}]".format(i - 1, i),
        }
        constraints.append(dic)
else:
    raise ValueError()

# ----- RUN MODEL ----- #

# Define the simulator model
truth = np.array([0.15, 0.05])
sumtype = "all"
if DIMS == 1:
    model = simulator.SIRModel(truth, N=50, sumtype=sumtype)
else:
    model = simulator.SIRModelMultiple(truth, N=50)
bounds_sir = [[0, 0.5], [0, 0.5]]

# Define the SequentialBED object
foo = "../data/sirmodel_seq"
SIR_death = sequentialdesign.SequentialBED(
    params,
    model,
    domain=domain,
    constraints=constraints,
    num_cores=num_cores,
    utiltype="MI",
)

# Run the actual optimisation and save data
# Data is saved as '../data/sirmodel_seq_iter1.npz'
SIR_death.optimisation(
    n_iter=1,
    BO_init_num=INIT_NUM,
    BO_max_iter=MAX_ITER,
    filn=foo,
    obs_file=None,
    bounds=bounds_sir
)
