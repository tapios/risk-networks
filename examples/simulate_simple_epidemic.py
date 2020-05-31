import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random 
import matplotlib.pyplot as plt

# Utilities for generating random populations
from epiforecast.populations import populate_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler, ConstantSampler

from epiforecast.contact_simulator import ContactSimulator, DiurnalMeanContactRate
from epiforecast.kinetic_model_simulator import KineticModel, print_statuses
from epiforecast.scenarios import load_edges

from epiforecast.node_identifier_helper import load_node_identifiers

def random_epidemic(population, initial_infected, initial_exposed):
    """
    Returns a status dictionary associated with a random infection
    within a population associated with node_identifiers.
    """

    statuses = {node: 'S' for node in range(population)}

    initial_infected_nodes = np.random.choice(population, size=initial_infected, replace=False)
    initial_exposed_nodes = np.random.choice(population, size=initial_exposed, replace=False)

    for i in initial_infected_nodes:
        statuses[i] = 'I'

    for i in initial_exposed_nodes:
        if statuses[i] is not 'I':
            statuses[i] = 'E'

    return statuses

#
# Set random seeds for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
np.random.seed(123)
random.seed(123)

#
# Load an example network
#

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt')) 

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
population = len(contact_network)

#
# Build the contact simulator
#

contact_simulator = ContactSimulator(

             n_contacts = nx.number_of_edges(contact_network),
    mean_event_duration = 1 / 60 / 24, # 1 minute in units of days
      mean_contact_rate = DiurnalMeanContactRate(maximum=40, minimum=2),
             start_time = -3 / 24, # negative start time allows short 'spinup' of contacts

)

#
# Clinical parameters of an age-distributed population
#

# The age category of each community individual,

age_distribution = [ 0.21,  # 0-17 years
                     0.40,  # 18-44 years
                     0.25,  # 45-64 years
                     0.08   # 65-75 years
                   ]

## 75 onwards
age_distribution.append(1 - sum(age_distribution))
                       
ages = populate_ages(population, distribution=age_distribution)

# Next, we randomly generate clinical properties for our example population.
# Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
latent_periods              = sample_distribution(ConstantSampler(3.7), population=population, minimum=0)
community_infection_periods = sample_distribution(ConstantSampler(3.2), population=population, minimum=0)
hospital_infection_periods  = sample_distribution(ConstantSampler(5), population=population, minimum=0)

hospitalization_fraction     = sample_distribution(AgeAwareBetaSampler(mean=[ 0.002,  0.01,  0.04, 0.075, 0.16], b=4), ages=ages)
community_mortality_fraction = sample_distribution(AgeAwareBetaSampler(mean=[1e-4, 1e-3, 0.003, 0.01, 0.02], b=4), ages=ages)
hospital_mortality_fraction  = sample_distribution(AgeAwareBetaSampler(mean=[0.019, 0.075,  0.195, 0.328,  0.514], b=4), ages=ages)

# We process the clinical data to determine transition rates between each epidemiological state,
transition_rates = TransitionRates(population,
                                   latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

transmission_rate = 12.0
hospital_transmission_reduction = 0.1

#
# Build the kinetic model
#

kinetic_model = KineticModel(contact_network = contact_network,
                             transition_rates = transition_rates,
                             community_transmission_rate = transmission_rate,
                             hospital_transmission_reduction = hospital_transmission_reduction)

# 
# Seed an infection
#

statuses = random_epidemic(population, initial_infected=2, initial_exposed=0)
kinetic_model.set_statuses(statuses)

# 
# Simulate the growth and equilibration of an epidemic
#

static_contact_interval = 3/24 # days
interval = 21                  # days

steps = int(interval / static_contact_interval)
start_times = static_contact_interval * np.arange(steps)

mean_contact_duration = []

# Run the simulation
for i in range(steps):

    start = timer() 
    contacts = contact_simulator.mean_contact_duration(stop_time = start_times[i])
    end = timer() 

    mean_contact_duration.append(np.mean(contacts))

    kinetic_model.set_mean_contact_duration(contacts)
    kinetic_model.simulate(static_contact_interval)

    print("Epidemic day: {: 7.3f}, wall_time: {: 6.3f} s,".format(kinetic_model.times[-1], end - start), 
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

second = 1 / 60 / 60 / 24

fig, axs = plt.subplots(nrows=3, sharex=True)

plt.sca(axs[0])
plt.plot(start_times, np.array(mean_contact_duration) / second)
plt.ylabel("Network-averaged contact duration (seconds)")

plt.sca(axs[1])
plt.plot(kinetic_model.times, kinetic_model.statuses['S'])
plt.ylabel("Number of Susceptible, S")

plt.sca(axs[2])
plt.plot(kinetic_model.times, kinetic_model.statuses['E'], label='Exposed')
plt.plot(kinetic_model.times, kinetic_model.statuses['I'], label='Infected')
plt.plot(kinetic_model.times, kinetic_model.statuses['H'], label='Hospitalized')
plt.plot(kinetic_model.times, kinetic_model.statuses['R'], label='Resistant')
plt.plot(kinetic_model.times, kinetic_model.statuses['D'], label='Deceased')

plt.xlabel("Time (days)")
plt.ylabel("Number of E, I, H, R, D")
plt.legend()

image_path = "simple_epidemic.png"
print("Saving a visualization of results at", image_path)
plt.savefig(image_path, dpi=480)
