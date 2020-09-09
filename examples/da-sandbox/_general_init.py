import os
from numba import set_num_threads
import ray

from epiforecast.utilities import seed_three_random_states

from _argparse_init import arguments
from _constants import OUTPUT_PATH, SAVE_PATH, SEED_GENERAL_INIT
from _utilities import print_start_of, print_end_of


print_start_of(__name__)
################################################################################
# parallel #####################################################################
if arguments.parallel_flag:
    object_store_memory = min(arguments.parallel_memory // 2, 2_000_000_000)
    memory = arguments.parallel_memory - object_store_memory

    if len(arguments.parallel_temp_dir) > 0:
        temp_dir = arguments.parallel_temp_dir
    else:
        temp_dir = None # this is the default in ray's method

    ray.init(num_cpus=arguments.parallel_num_cpus,
             memory=memory,
             object_store_memory=object_store_memory,
             temp_dir=temp_dir
    )

# numba ########################################################################
set_num_threads(1)

# seeding ######################################################################
seed_three_random_states(SEED_GENERAL_INIT)

# create directories ###########################################################
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

################################################################################
print_end_of(__name__)

