import os, sys; sys.path.append(os.path.join("..", ".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import random 
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from numba import set_num_threads

set_num_threads(1)

from matplotlib import rcParams

# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Latin Modern Roman',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True}
# tell matplotlib about your params
rcParams.update(params)

# set nice figure sizes
fig_width_pt = 368    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.75*fig_width*ratio  # height in inches
fig_size = [fig_width, fig_height]
rcParams.update({'figure.figsize': fig_size})

# Utilities for generating random populations
from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant

from epiforecast.kinetic_model_simulator import KineticModel, print_statuses
from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.node_identifier_helper import load_node_identifiers

from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService

from epiforecast.utilities import seed_numba_random_state, seed_three_random_states

def simulation_average(model_data, sampling_time = 1):
    """
    Returns daily averages of simulation data.
    """
    
    simulation_data_average = {}
    daily_average = {}

    for key in model_data.statuses.keys():
        simulation_data_average[key] = []
        daily_average[key] = []
    
    tav = 0

    for i in range(len(model_data.times)):
        for key in model_data.statuses.keys():
            simulation_data_average[key].append(model_data.statuses[key][i])

        if model_data.times[i] >= tav:
            for key in model_data.statuses.keys():
                daily_average[key].append(np.mean(simulation_data_average[key]))
                simulation_data_average[key] = []

            tav += sampling_time

    return daily_average
        
#
# Set random seeds for reproducibility
#

seed = 2132

seed_three_random_states(seed)

#
# Load an example network
#

edges = load_edges(os.path.join('..', '..', 'data', 'networks', 'edge_list_SBM_1e4_nobeds.txt')) 
node_identifiers = load_node_identifiers(os.path.join('..', '..', 'data', 'networks', 'node_identifier_SBM_1e4_nobeds.txt'))

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
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
        hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16]),
    community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07,  0.015]),
     hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512])

)

transmission_rate = 12.0
hospital_transmission_reduction = 0.1

# 
# Simulate the growth and equilibration of an epidemic
#

minute = 1 / 60 / 24
hour = 60 * minute

# Run the simulation

health_service = HealthService(patient_capacity = int(0.05 * len(contact_network)),
                               health_worker_population = len(node_identifiers['health_workers']),
                               static_population_network = contact_network)

epidemic_simulator = EpidemicSimulator(contact_network,            
                                                 mean_contact_lifetime = 0.5 * minute,
                                                  day_inception_rate = 22,
                                                  night_inception_rate = 2,
                                                      transition_rates = transition_rates,
                                               static_contact_interval = 3 * hour,
                                           community_transmission_rate = 12.0,
                                                        health_service = health_service,
                                       hospital_transmission_reduction = 0.1)

statuses = random_epidemic(contact_network, fraction_infected=0.005)

epidemic_simulator.set_statuses(statuses)

epidemic_simulator.run(stop_time = 14)

kinetic_model = epidemic_simulator.kinetic_model

# SD intervention

health_service = epidemic_simulator.health_service

statuses = epidemic_simulator.kinetic_model.current_statuses

epidemic_simulator = EpidemicSimulator(contact_network,            
                                                 mean_contact_lifetime = 0.5 * minute,
                                                  day_inception_rate = 8,
                                                  night_inception_rate = 2,
                                                      transition_rates = transition_rates,
                                               static_contact_interval = 3 * hour,
                                           community_transmission_rate = 12.0,
                                                        health_service = health_service,
                                       hospital_transmission_reduction = 0.1)

epidemic_simulator.set_statuses(statuses)

epidemic_simulator.run(stop_time = 101) # days

for key in kinetic_model.statuses.keys():
   kinetic_model.statuses[key].extend(epidemic_simulator.kinetic_model.statuses[key])


kinetic_model.times.extend(kinetic_model.times[-1]+epidemic_simulator.kinetic_model.times)

# loosening SD intervention

health_service = epidemic_simulator.health_service

statuses = epidemic_simulator.kinetic_model.current_statuses

inf_cnt = 50
cnt = 0

for i in range(len(statuses)):
    if statuses[i] == 'S' and cnt <= inf_cnt:
        statuses[i] = 'I'
        cnt += 1

epidemic_simulator = EpidemicSimulator(contact_network,            
                                                 mean_contact_lifetime = 0.5 * minute,
                                                  day_inception_rate = 22,
                                                  night_inception_rate = 2,
                                                      transition_rates = transition_rates,
                                               static_contact_interval = 3 * hour,
                                           community_transmission_rate = 12.0,
                                                        health_service = health_service,
                                       hospital_transmission_reduction = 0.1)

epidemic_simulator.set_statuses(statuses)

epidemic_simulator.run(stop_time = 70) # days

for key in kinetic_model.statuses.keys():
   kinetic_model.statuses[key].extend(epidemic_simulator.kinetic_model.statuses[key])


kinetic_model.times.extend(kinetic_model.times[-1]+epidemic_simulator.kinetic_model.times)

#
# Plot the results and compare with NYC data.
#

np.savetxt("../../data/simulation_data/simulation_data_NYC_1e4_6.txt", np.c_[kinetic_model.times, kinetic_model.statuses['S'], kinetic_model.statuses['E'], kinetic_model.statuses['I'], kinetic_model.statuses['H'], kinetic_model.statuses['R'],kinetic_model.statuses['D']], header = 'S E I H R D seed: %d'%seed)

NYC_data = pd.read_csv(os.path.join('..', '..', 'data', 'NYC_COVID_CASES', 'data_new_york.csv'))
NYC_cases = np.asarray([float(x) for x in NYC_data['Cases'].tolist()])
NYC_deaths =  np.asarray([float(x) for x in NYC_data['Deaths'].tolist()])
NYC_date_of_interest = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_data['DATE_OF_INTEREST'].tolist()])

# population of NYC
NYC_population = 8.399e6

# fraction reported cases
fraction_reported = 0.15

# cumulative cases NYC
cumulative_reported_cases_NYC = 1/fraction_reported*np.cumsum(NYC_cases)/NYC_population
cumulative_deaths_NYC = np.cumsum(NYC_deaths)/NYC_population*1e5

# daily averages of simulation data
# sampling_time = 1 means that we average over 1-day intervals
daily_average = simulation_average(kinetic_model, sampling_time=1)
cumulative_cases_simulation = 1-np.asarray(daily_average['S'])/population
cumulative_deaths_simulation = np.asarray(daily_average['D'])/population*1e5

simulation_dates = np.asarray([NYC_date_of_interest[0] + i*dt.timedelta(days=1) for i in range(len(cumulative_cases_simulation))])

fig, ax = plt.subplots()

ax2 = ax.twinx()

ax.plot(NYC_date_of_interest[::3], cumulative_reported_cases_NYC[::3], marker = 'o', markersize = 3, color = 'k', ls = 'None', label = r'total cases (NYC)')
ax.plot(simulation_dates+dt.timedelta(days = 17), cumulative_cases_simulation, 'k', label = 'total cases (simulation)')

ax2.plot(NYC_date_of_interest[::3], cumulative_deaths_NYC[::3], marker = 's', markersize = 4, color = 'darkred', markeredgecolor = 'Grey', ls = 'None', label = r'death cases (NYC)')
ax2.plot(simulation_dates+dt.timedelta(days = 17), cumulative_deaths_simulation, 'darkred', label = 'death cases (simulation)')

ax.set_xlim([dt.date(2020, 3, 1), dt.date(2020, 7, 19)])
ax.set_xticklabels(NYC_date_of_interest[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylim(0,0.3)
ax.set_ylabel(r'proportion of total cases', labelpad = 3)

ax2.set_ylim(0,400)
ax2.set_ylabel(r'total deaths/100,000', color = 'darkred')
ax2.tick_params(axis='y', labelcolor = 'darkred')   

ax.legend(frameon = False, loc = 2, fontsize = 6)
ax2.legend(frameon = False, loc = 1, fontsize = 6)

plt.tight_layout()
plt.margins(0,0)
plt.savefig('../../figs/new_york_cases.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)

# plot reproduction number estimate
fig, ax = plt.subplots()

ax.set_xlim([dt.date(2020, 3, 1), dt.date(2020, 7, 19)])

plt.plot(simulation_dates+dt.timedelta(days = 17), np.asarray(daily_average['E'])/np.asarray(daily_average['I']), 'k')

plt.plot([dt.date(2020, 3, 1), dt.date(2020, 6, 21)], [1,1], linestyle = '--', color = 'Grey')

ax.set_xticklabels(NYC_date_of_interest[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylabel(r'$R(t)$')

ax.set_ylim(0,4)

plt.tight_layout()
plt.margins(0,0)
plt.savefig('../../figs/reproduction_number.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)

# plot all model compartments
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

image_path = ("../../figs/simple_epidemic_with_slow_contact_simulator_" + 
              "maxlambda_{:d}.png".format(contact_simulator.mean_contact_rate.maximum_i))

print("Saving a visualization of results at", image_path)
plt.savefig(image_path, dpi=480)
