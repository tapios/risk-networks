import os, sys; sys.path.append(os.path.join("../.."))
from timeit import default_timer as timer
import pickle

from _constants import (start_time,
                        total_time,
                        static_contact_interval,
                        ENSEMBLE_PATH)
from _network_init import network
from _stochastic_init import (epidemic_simulator,
                              epidemic_data_storage,
                              kinetic_ic)
from _utilities import print_start_of, print_end_of, print_info_module

from _argparse_init import arguments


print_start_of(__name__)
################################################################################
print_info_module(__name__,
                  "Running epidemic_simulator for",
                  total_time,
                  "days")

time = start_time          # float
kinetic_state = kinetic_ic # dict { node : compartment }

kinetic_states_timeseries = []
kinetic_states_timeseries.append(kinetic_state) # storing ic

walltime_epidemic_simulator = 0.0
walltime_data_storage = 0.0
timer_run_and_store = timer()
for i in range(int(total_time/static_contact_interval)):
    timer_epidemic_simulator = timer()
    network = epidemic_simulator.run(
            stop_time=epidemic_simulator.time + static_contact_interval,
            current_network=network)
    walltime_epidemic_simulator += timer() - timer_epidemic_simulator

    # store for further usage (master equations etc)
    timer_data_storage = timer()
    epidemic_data_storage.save_network_by_start_time(
            start_time=time,
            contact_network=network)
    epidemic_data_storage.save_start_statuses_to_network(
            start_time=time,
            start_statuses=kinetic_state)

    time          = epidemic_simulator.time
    kinetic_state = epidemic_simulator.kinetic_model.current_statuses

    epidemic_data_storage.save_end_statuses_to_network(
            end_time=time,
            end_statuses=kinetic_state)
    walltime_data_storage += timer() - timer_data_storage

    # store for plotting
    kinetic_states_timeseries.append(kinetic_state)

# store epidemic data to file
pickle.dump(epidemic_data_storage,
            open(os.path.join(ENSEMBLE_PATH, arguments.epidemic_storage_name), "wb" ),
            protocol=pickle.HIGHEST_PROTOCOL)
            
print_info_module(__name__,
                  "You can find the kinetic ensemble at:",
                  ENSEMBLE_PATH)

print_info_module(__name__,
                  "You can find the kinetic timeseries as:",
                  arguments.epidemic_storage_name)


print_info_module(__name__,
                  "Simulation done; elapsed:",
                  timer() - timer_run_and_store)
print_info_module(__name__,
                  "Simulation done; epidemic simulator walltime:",
                  walltime_epidemic_simulator)
print_info_module(__name__,
                  "Simulation done; data storage walltime:",
                  walltime_data_storage,
                  end='\n\n')

################################################################################
print_end_of(__name__)
