import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random 
import matplotlib.pyplot as plt

# Utilities for generating random populations
from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler

from epiforecast.contact_simulator import ContactSimulator, DiurnalMeanContactRate
from epiforecast.kinetic_model_simulator import KineticModel, print_statuses
from epiforecast.scenarios import random_epidemic

#
# Set random seeds for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
np.random.seed(12223)
random.seed(123)

#
# Create an example network
#

population_network = nx.OrderedGraph(nx.barabasi_albert_graph(2000, 2))
population = len(population_network)

#
# Clinical parameters of an age-distributed population
#

assign_ages(population_network, distribution=[0.21, 0.4, 0.25, 0.08, 0.06])

# We process the clinical data to determine transition rates between each epidemiological state,
transition_rates = TransitionRates(population_network,

                  latent_periods = 3.7,
     community_infection_periods = 3.2,
      hospital_infection_periods = 5.0,
        hospitalization_fraction = AgeDependentBetaSampler(mean=[0.002,  0.01,   0.04, 0.075,  0.16], b=4),
    community_mortality_fraction = AgeDependentBetaSampler(mean=[ 1e-4,  1e-3,  0.003,  0.01,  0.02], b=4),
     hospital_mortality_fraction = AgeDependentBetaSampler(mean=[0.019, 0.075,  0.195, 0.328, 0.514], b=4)

)

transmission_rate = 12.0
hospital_transmission_reduction = 0.1

#
# Build the contact simulator
#

minute = 1 / 60 / 24
hour = 60 * minute

contact_simulator = ContactSimulator(n_contacts = nx.number_of_edges(population_network),
                                     mean_event_lifetime = 1 * minute,
                                     inception_rate = DiurnalMeanContactRate(maximum=34, minimum=2))

#
# Build the kinetic model
#

kinetic_model = KineticModel(contact_network = population_network,
                             transition_rates = transition_rates,
                             community_transmission_rate = transmission_rate,
                             hospital_transmission_reduction = hospital_transmission_reduction)

# 
# Seed an infection
#

statuses = random_epidemic(population_network, fraction_infected=0.01)
kinetic_model.set_statuses(statuses)

# 
# Simulate the growth and equilibration of an epidemic
#

minute = 1 / 60 / 24
hour = 60 * minute

static_contact_interval = 3 * hour
interval = 21 # days

steps = int(interval / static_contact_interval)
stop_times = static_contact_interval * np.arange(start=1, stop=steps+1)

mean_contact_duration = []

# Run the simulation
for i in range(steps):

    start = timer() 
    contacts = contact_simulator.mean_contact_duration(stop_time = stop_times[i])
    end = timer() 

    mean_contact_duration.append(np.mean(contacts))

    kinetic_model.set_mean_contact_duration(contacts)
    kinetic_model.simulate(static_contact_interval)

    print("Epidemic day: {: 7.3f}, wall_time: {: 6.3f} s,".format(kinetic_model.times[-1], end - start), 
          "mean(w_ji): {: 3.0f} min,".format(mean_contact_duration[-1] / minute),
          "statuses: ",
          "S {: 4d} |".format(kinetic_model.statuses['S'][-1]), 
          "E {: 4d} |".format(kinetic_model.statuses['E'][-1]), 
          "I {: 4d} |".format(kinetic_model.statuses['I'][-1]), 
          "H {: 4d} |".format(kinetic_model.statuses['H'][-1]), 
          "R {: 4d} |".format(kinetic_model.statuses['R'][-1]), 
          "D {: 4d} |".format(kinetic_model.statuses['D'][-1]))

#
# Plot the results.
#

fig, axs = plt.subplots(nrows=3, sharex=True)

plt.sca(axs[0])
plt.plot(stop_times, np.array(mean_contact_duration) / minute)
plt.ylabel("Mean $ w_{ji} $")

plt.sca(axs[1])
plt.plot(kinetic_model.times, kinetic_model.statuses['S'])
plt.ylabel("Total susceptible, $S$")

plt.sca(axs[2])
plt.plot(kinetic_model.times, kinetic_model.statuses['E'], label='Exposed')
plt.plot(kinetic_model.times, kinetic_model.statuses['I'], label='Infected')
plt.plot(kinetic_model.times, kinetic_model.statuses['H'], label='Hospitalized')
plt.plot(kinetic_model.times, kinetic_model.statuses['R'], label='Resistant')
plt.plot(kinetic_model.times, kinetic_model.statuses['D'], label='Deceased')

plt.xlabel("Time (days)")
plt.ylabel("Total $E, I, H, R, D$")
plt.legend()

image_path = "simple_epidemic.png"
print("Saving a visualization of results at", image_path)
plt.savefig(image_path, dpi=480)
