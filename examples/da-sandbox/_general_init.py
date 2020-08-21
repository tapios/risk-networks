import os
from numba import set_num_threads

from epiforecast.utilities import seed_three_random_states

from _constants import OUTPUT_PATH, SAVE_PATH
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
# numba ########################################################################
set_num_threads(1)

# random seeds for reproducibility #############################################
seed = 942395
seed_three_random_states(seed)

# create directories ###########################################################
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

################################################################################
print_end_of(__name__)

