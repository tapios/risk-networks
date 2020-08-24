import os

from _argparse_init import arguments
from _constants import SAVE_PATH
from _utilities import print_start_of, print_end_of, print_info_module


print_start_of(__name__)
################################################################################
storage_path = os.path.join(SAVE_PATH, arguments.epidemic_storage_name)
kinetic_path = os.path.join(SAVE_PATH, arguments.epidemic_kinetic_states_name)

if arguments.epidemic_load_data:
    import pickle

    print_info_module(__name__, "Loading epidemic data...")
    load_success = True
    try:
        with open(storage_path, 'rb') as storage_pickle:
            epidemic_data_storage = pickle.load(storage_pickle)

        with open(kinetic_path, 'rb') as kinetic_pickle:
            kinetic_states_timeseries = pickle.load(kinetic_pickle)
    except:
        print_info_module(__name__, "Loading failed, will compute")
        load_success = False

if (not arguments.epidemic_load_data) or (not load_success):
    from _run_and_store_epidemic import (epidemic_data_storage,
                                         kinetic_states_timeseries)

    if arguments.epidemic_save_data:
        with open(storage_path, 'wb') as storage_pickle:
            pickle.dump(epidemic_data_storage, storage_pickle, -1)

        with open(kinetic_path, 'wb') as kinetic_pickle:
            pickle.dump(kinetic_states_timeseries, kinetic_pickle, -1)

################################################################################
print_end_of(__name__)

