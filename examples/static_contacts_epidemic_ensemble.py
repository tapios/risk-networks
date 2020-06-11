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
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.scenarios import random_epidemic
from epiforecast.utilities import seed_three_random_states

#
# Set random states for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
seed_three_random_states(123)

#
# Create an example network
#

contact_network = nx.barabasi_albert_graph(3000, 10)
population = len(contact_network)

#
# Clinical parameters of an age-distributed population
#

assign_ages(contact_network, distribution=[0.21, 0.4, 0.25, 0.08, 0.06])

# We process the clinical data to determine transition rates between each epidemiological state,
transition_rates = TransitionRates(contact_network,

                  latent_periods = 3.7,
     community_infection_periods = 3.2,
      hospital_infection_periods = 5.0,
        hospitalization_fraction = AgeDependentBetaSampler(mean=[0.002,  0.01,   0.04, 0.075,  0.16], b=4),
    community_mortality_fraction = AgeDependentBetaSampler(mean=[ 1e-4,  1e-3,  0.003,  0.01,  0.02], b=4),
     hospital_mortality_fraction = AgeDependentBetaSampler(mean=[0.019, 0.075,  0.195, 0.328, 0.514], b=4)

)

#
# Build the epidemic simulator and determine initial condition
#

epidemic_simulator = EpidemicSimulator(contact_network,            
                                       contact_simulator = False,
                                       transition_rates = transition_rates,
                                       community_transmission_rate = 12.0,
                                       hospital_transmission_reduction = 0.1)

# Random initial condition
statuses = random_epidemic(contact_network, fraction_infected=0.01)

#
# Create an ensemble of epidemics starting from identical initial conditions
#

n_ensemble = 10
ensemble_times = []
ensemble_statuses = []

kinetic_model = epidemic_simulator.kinetic_model

minute = 1 / 60 / 24

for m in range(n_ensemble):
    # Resets kinetic_model.times and kinetic_model.statuses with time=0.0
    epidemic_simulator.set_statuses(statuses, time=0.0)

    # Run forward. mean_contact_duration is passed to nx.set_edge_attributes.
    epidemic_simulator.run_with_static_contacts(stop_time = 35,
                                                mean_contact_duration = 10 * minute)

    ensemble_times.append(kinetic_model.times)
    ensemble_statuses.append(kinetic_model.statuses)

#
# Plot the results.
#

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
alpha = 0.4
width = 1

fig, axs = plt.subplots(nrows=3, sharex=True)

plt.sca(axs[0])
plt.ylabel("Susceptible, $S$")

plt.sca(axs[1])
plt.ylabel("Resistant, $R$")

plt.sca(axs[2])
plt.xlabel("Time (days)")
plt.ylabel("Total $E, I, H, D$")

for m in range(n_ensemble):
    times = ensemble_times[m]
    statuses = ensemble_statuses[m]
    color = default_colors[m]

    plt.sca(axs[0])
    plt.plot(times, statuses['S'], linewidth=width, color=color, alpha=alpha)

    plt.sca(axs[1])
    plt.plot(times, statuses['R'], linewidth=width, color=color, alpha=alpha)

    plt.sca(axs[2])

    if m == 0:
        exposed_label = "Exposed"
        infected_label = "Infected"
        hospitalized_label = "Hospitalized"
        deceased_label = "Deceased"
    else:
        exposed_label = ""
        infected_label = ""
        hospitalized_label = ""
        deceased_label = ""

    plt.plot(times, statuses['E'], "-",  label=exposed_label,      linewidth=width, color=color, alpha=alpha)
    plt.plot(times, statuses['I'], "--", label=infected_label,     linewidth=width, color=color, alpha=alpha)
    plt.plot(times, statuses['H'], "-.", label=hospitalized_label, linewidth=width, color=color, alpha=alpha)
    plt.plot(times, statuses['D'], ":",  label=deceased_label,     linewidth=width, color=color, alpha=alpha)

plt.sca(axs[2])
plt.legend()

image_path = "epidemic_ensemble_on_static_network.png"
print("Saving a visualization of results at", image_path)
plt.savefig(image_path, dpi=480)
