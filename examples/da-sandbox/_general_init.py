import os
from numba import set_num_threads

from epiforecast.utilities import seed_three_random_states

from _constants import OUTPUT_PATH
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
# numba ########################################################################
set_num_threads(1)

# seeding ######################################################################
seed_three_random_states(942395)

# create an output directory ###################################################
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

################################################################################
print_end_of(__name__)

