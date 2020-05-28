import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

from epiforecast.contact_simulator import ContactSimulator, DiurnalContactModulation

np.random.seed(1234)

total_contacts = 20000
block_size = 2000
n_blocks = int(total_contacts / block_size)
start_time = 0.5
stop_time = 0.5

max_mean_contacts = np.linspace(start=4, stop=20, num=n_blocks)

#####
##### Simulate all the contacts in one bulk simulation with the mean max contacts...
#####

bulk_mean_contacts = max_mean_contacts.mean()

bulk_simulator = ContactSimulator(n_contacts=total_contacts, initial_fraction_active_contacts=0.01, start_time=0.5)
bulk_modulation = DiurnalContactModulation(peak=bulk_mean_contacts, minimum=3)

start = timer()
bulk_simulator.simulate_contact_duration(stop_time = 1/8 + 1/2, modulation = bulk_modulation)
end = timer()

print("\nBulk simulation took:", end - start, "\n")
 
#####
##### Break up the population into "blocks" and simulate independently
#####

simulators = [
              ContactSimulator(n_contacts=block_size, initial_fraction_active_contacts=0.01, start_time=0.5)
              for lam in max_mean_contacts
             ]
                                        
modulations = [DiurnalContactModulation(peak=lam, minimum=3) for lam in max_mean_contacts]
                                 
# Run forward all the blocks for 1/8 of a day starting at noon
start = timer()

for i, sim in enumerate(simulators):
    block_modulation = modulations[i]

    sim.simulate_contact_duration(stop_time = 1/8 + 1/2, modulation = block_modulation) 

    print("Block", i,
          "with max(λ) = {:.1f}".format(block_modulation.peak_mean_contacts),
          "took", sim.interval_steps, "Gillespie steps")

end = timer()

print("\nA loop over", n_blocks, "blocks took:", end - start, "\n")

#####
##### Print some informative messages
#####

minute = 1 / 60 / 24

print("Network-averaged mean contact duration in the bulk simulation",
      "is {:.3f} minutes".format(bulk_simulator.contact_duration.mean() / minute))

super_average = 0.0

for i, sim in enumerate(simulators):
    print("Network-averaged mean contact duration for block", i,
          "is {:.3f} minutes".format(sim.contact_duration.mean() / minute))

    super_average += sim.contact_duration.mean() / n_blocks

print("\nSuper average over all contacts in all blocks",
      "is {:3f} minutes".format(super_average / minute))
