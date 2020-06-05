import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random 
import matplotlib.pyplot as plt

# Utilities for generating random populations
from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler

from epiforecast.health_service import HealthService
from epiforecast.contact_simulator import DiurnalContactInceptionRate
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.scenarios import random_epidemic

#
# Set random seeds for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
np.random.seed(123)
random.seed(123)

#
# Create an example network
#

population_network = nx.barabasi_albert_graph(10000, 2)
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

#
# Build the epidemic simulator
#

minute = 1 / 60 / 24
hour = 60 * minute

health_service = HealthService(patient_capacity = 5,
                               health_worker_population = 10, # sets the first 10 nodes as health workers
                               static_population_network = population_network)

epidemic_simulator = EpidemicSimulator(population_network,            
                                                 mean_contact_lifetime = 1 * minute,
                                                contact_inception_rate = DiurnalContactInceptionRate(maximum=34, minimum=2),
                                                      transition_rates = transition_rates,
                                               static_contact_interval = 3 * hour,
                                           community_transmission_rate = 12.0,
                                                        health_service = None, # health_service,
                                       hospital_transmission_reduction = 0.1)

statuses = random_epidemic(population_network, fraction_infected=0.01)
epidemic_simulator.set_statuses(statuses)

epidemic_simulator.run(stop_time = 21) # days

#
# Plot the results.
#

kinetic_model = epidemic_simulator.kinetic_model

fig, axs = plt.subplots(nrows=2, sharex=True)

plt.sca(axs[0])
plt.plot(kinetic_model.times, kinetic_model.statuses['S'])
plt.ylabel("Total susceptible, $S$")

plt.sca(axs[1])
plt.plot(kinetic_model.times, kinetic_model.statuses['E'], label='Exposed')
plt.plot(kinetic_model.times, kinetic_model.statuses['I'], label='Infected')
plt.plot(kinetic_model.times, kinetic_model.statuses['H'], label='Hospitalized')
plt.plot(kinetic_model.times, kinetic_model.statuses['R'], label='Resistant')
plt.plot(kinetic_model.times, kinetic_model.statuses['D'], label='Deceased')

plt.xlabel("Time (days)")
plt.ylabel("Total $E, I, H, R, D$")
plt.legend()

image_path = "super_simple_epidemic.png"
print("Saving a visualization of results at", image_path)
plt.savefig(image_path, dpi=480)
