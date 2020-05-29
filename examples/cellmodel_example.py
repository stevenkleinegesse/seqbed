import numpy as np

import seqbed.simulator as simulator
import seqbed.sequentialdesign as sequentialdesign

# ---- GLOBAL PARAMS ------ #

# Set these to change performance of experimental design

# Number of CPU cores
num_cores = 2
# Set dimensions of design variable
DIMS = 1  # Keep at 1 for myopic BED

# Number of prior samples (higher => more accurate posterior)
NS = 300
# Max number of utility evaluations in B.O. (per core)
MAX_ITER = 5

# number of initial data points for B.O.
if num_cores > 5:
    INIT_NUM = num_cores
else:
    INIT_NUM = 5

# ----- SPECIFY MODEL ----- #

# Obtain Death model prior samples
param_0 = np.random.uniform(0, 1, NS).reshape(-1, 1)
param_1 = np.random.uniform(0, 0.005, NS).reshape(-1, 1)
prior_cell = np.hstack((param_0, param_1))

# Define the domain for BO
# Dimensions
DIMS = 1

# Define the domain for BO
domain_cell = [
    {
        "name": "var_1",
        "type": "discrete",
        "domain": tuple(range(1, 145)),
        "dimensionality": DIMS,
    }
]

# Define the constraints for BO
# Time cannot go backwards
if DIMS == 1:
    constraints_cell = None
elif DIMS > 1:
    constraints_cell = list()
    for i in range(1, DIMS):
        dic = {
            "name": "constr_{}".format(i),
            "constraint": "x[:,{}]-x[:,{}]".format(i - 1, i),
        }
        constraints_cell.append(dic)
else:
    raise ValueError()

# ----- RUN MODEL ----- #

# Define the simulator model
truth_cell = np.array([0.35, 0.001])
model_cell = simulator.CellModel(truth_cell, 110, 144)
bounds_cell = [[0.0, 1.0], [0.0, 0.005]]

# Define the SequentialBED object
foo = "../data/cellmodel_seq"
BED_cell = sequentialdesign.SequentialBED(
    prior_cell,
    model_cell,
    domain=domain_cell,
    constraints=constraints_cell,
    num_cores=num_cores,
    utiltype="MI",
)

# Run the actual optimisation and save data
# Data is saved as '../data/cellmodel_seq_iter1.npz'
BED_cell.optimisation(
    n_iter=1,
    BO_init_num=INIT_NUM,
    BO_max_iter=MAX_ITER,
    filn=foo,
    obs_file=None,
    bounds=bounds_cell
)
