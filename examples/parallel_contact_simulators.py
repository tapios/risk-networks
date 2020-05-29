import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

from epiforecast.contact_simulator import ContactSimulator, DiurnalMeanContactRate

np.random.seed(1234)

total_contacts = 20000 # total number of contacts / edges
block_size     = 2000  # size of each block

n_blocks = int(total_contacts / block_size)

# List of maximum mean contacts for each block
max_mean_contacts = np.linspace(start=4, stop=20, num=n_blocks)

# Start and stop time for the simulation
start_time = 1/2       # noon
stop_time  = 1/2 + 1/8 # 1/8 of a day after noon

#
# Simulate all the contacts in one bulk simulation with the mean max contacts...
#

bulk_mean_contacts = max_mean_contacts.mean()

bulk_simulator = ContactSimulator(n_contacts=total_contacts, initial_fraction_active_contacts=0.01,
                                  start_time=start_time)

bulk_contact_rate = DiurnalMeanContactRate(maximum=bulk_mean_contacts, minimum=3)

# Simulate
start = timer()
bulk_simulator.simulate_contact_duration(stop_time=stop_time, mean_contact_rate=bulk_contact_rate)
end = timer()
print("\n The bulk simulation took", end - start, "seconds of wall time.\n")
print(   "The bulk simulation took", bulk_simulator.interval_steps, "Gillespie steps.")
 
#
# Break up the population into "blocks" and simulate independently
#

simulators = [ContactSimulator(n_contacts=block_size, initial_fraction_active_contacts=0.01,
                               start_time=start_time) for lam in max_mean_contacts]
                                        
block_contact_rates = [DiurnalMeanContactRate(maximum=lam, minimum=3) for lam in max_mean_contacts]
                                 
# Run forward all the blocks for 1/8 of a day starting at noon
start = timer()

for i, sim in enumerate(simulators):
    block_rate = block_contact_rates[i]

    sim.simulate_contact_duration(stop_time=stop_time, mean_contact_rate=block_rate) 

    print("Block", i,
          "with max(λ) = {:.1f}".format(block_rate.maximum_mean_contacts),
          "took", sim.interval_steps, "Gillespie steps.")

end = timer()

print("\n A loop over", n_blocks, "blocks took", end - start, "seconds of wall time.\n")

#
# Print info about the mean contact rate in bulk and block simulations
#

minute = 1 / 60 / 24 # in units of days

print("The network-averaged mean contact duration in the bulk simulation",
      "is {:.3f} minutes.".format(bulk_simulator.contact_duration.mean() / minute))

super_average = 0.0

for i, sim in enumerate(simulators):
    print("The network-averaged mean contact duration for block", i,
          "is {:.3f} minutes".format(sim.contact_duration.mean() / minute))

    super_average += sim.contact_duration.mean() / n_blocks

print("\n The super-average contact duration over all contacts in all blocks",
      "is {:3f} minutes.".format(super_average / minute))
