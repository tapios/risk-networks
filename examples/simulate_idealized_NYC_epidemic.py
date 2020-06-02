import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import random 
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# Utilities for generating random populations
from epiforecast.populations import populate_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler, ConstantSampler

from epiforecast.fast_contact_simulator import FastContactSimulator, DiurnalMeanContactRate
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

def repeated_contact_simulator(contact_simulator, steps_contact_simulator, start_times):
    """
    Returns a contact simulation for "steps_contact_simulator" steps.
    """

    mean_contact_duration = []
    contact_duration = []

    print("...contact simulation started...")

    for i in range(steps_contact_simulator):
    
        start = timer() 
        contacts = contact_simulator.mean_contact_duration(stop_time = start_times[i])
        end = timer() 
    
        print("...step {:d}/{:d} completed in {: 6.3f} s".format(i, steps_contact_simulator, end - start))
        contact_duration.append(contacts)
        mean_contact_duration.append(np.mean(contacts))

    print("...contact simulation completed...")

    return mean_contact_duration, contact_duration
        
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
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

#
# Build the contact simulator
#

contact_simulator = FastContactSimulator(

             n_contacts = nx.number_of_edges(contact_network),
    mean_event_duration = 0.6 / 60 / 24, # 1 minute in units of days
      mean_contact_rate = DiurnalMeanContactRate(minimum_i = 2, maximum_i = 22, minimum_j = 2, maximum_j = 22),
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
hospital_infection_periods  = sample_distribution(ConstantSampler(5),   population=population, minimum=0)

#                                                                      ages:  0-17  18-44   45-64  65-75    75+
hospitalization_fraction     = sample_distribution(AgeAwareBetaSampler(mean=[0.002,  0.01,   0.04, 0.075,  0.16], b=4), ages=ages)
community_mortality_fraction = sample_distribution(AgeAwareBetaSampler(mean=[ 1e-4,  1e-3,  0.003,  0.01,  0.02], b=4), ages=ages)
hospital_mortality_fraction  = sample_distribution(AgeAwareBetaSampler(mean=[0.019, 0.075,  0.195, 0.328, 0.514], b=4), ages=ages)

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

statuses = random_epidemic(population, initial_infected=10, initial_exposed=0)
kinetic_model.set_statuses(statuses)

# 
# Simulate the growth and equilibration of an epidemic
#

minute = 1 / 60 / 24
hour = 60 * minute

static_contact_interval = 3 * hour
interval = 100 # days

steps = int(interval / static_contact_interval)
steps_contact_simulator = int(1 / static_contact_interval)
start_times = static_contact_interval * np.arange(steps_contact_simulator)

#
# Determine contacts for a one day period
#

mean_contact_duration, contact_duration = repeated_contact_simulator(contact_simulator, steps_contact_simulator, start_times)

# Social distancing time
time_SD = 13 # days
λmax_SD = 3
λmin_SD = 3
SD_flag = 0

# Run the simulation
for i in range(steps):

    kinetic_model.set_mean_contact_duration(contact_duration[i%steps_contact_simulator])
    start = timer()
    kinetic_model.simulate(static_contact_interval)
    end = timer() 

    print("Epidemic day: {: 7.3f}, wall_time: {: 6.3f} s,".format(kinetic_model.times[-1], end - start), 
          "mean(w_ji): {: 3.0f} min,".format(mean_contact_duration[i%steps_contact_simulator] / minute),
          "statuses: ",
          "S {: 4d} |".format(kinetic_model.statuses['S'][-1]), 
          "E {: 4d} |".format(kinetic_model.statuses['E'][-1]), 
          "I {: 4d} |".format(kinetic_model.statuses['I'][-1]), 
          "H {: 4d} |".format(kinetic_model.statuses['H'][-1]), 
          "R {: 4d} |".format(kinetic_model.statuses['R'][-1]), 
          "D {: 4d} |".format(kinetic_model.statuses['D'][-1]))

    # social distancing intervention
    if kinetic_model.times[-1] > time_SD and SD_flag == 0: 
        print("Social distancing intervention with λmax = %2.1f and λmin = %2.1f"%(λmax_SD, λmin_SD))

        contact_simulator = FastContactSimulator(

             n_contacts = nx.number_of_edges(contact_network),
             mean_event_duration = 0.6 / 60 / 24, # 0.5 minutes in units of days
             mean_contact_rate = DiurnalMeanContactRate(minimum_i=λmin_SD, maximum_i=λmax_SD, minimum_j=λmin_SD, maximum_j=λmax_SD),
             start_time = -3 / 24, # negative start time allows short 'spinup' of contacts

        )

        start_times = static_contact_interval * np.arange(steps_contact_simulator)
        mean_contact_duration, contact_duration = repeated_contact_simulator(contact_simulator, steps_contact_simulator, start_times)
        SD_flag = 1

#
# Plot the results and compare with NYC data.
#

NYC_data = pd.read_csv(os.path.join('..', 'data', 'NYC_COVID_CASES', 'data_new_york.csv'))
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

ax.plot(NYC_date_of_interest[::3], cumulative_reported_cases_NYC[::3], marker = 'o', markersize = 3, color = 'k', ls = 'None')
ax.plot(simulation_dates+dt.timedelta(days = 9), cumulative_cases_simulation, 'k')

ax2.plot(NYC_date_of_interest[::3], cumulative_deaths_NYC[::3], marker = 's', markersize = 4, color = 'darkred', markeredgecolor = 'Grey', ls = 'None')
ax2.plot(simulation_dates+dt.timedelta(days = 9), cumulative_deaths_simulation, 'darkred')

ax.set_xlim([dt.date(2020, 3, 1), dt.date(2020, 6, 7)])
ax.set_xticklabels(NYC_date_of_interest[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylim(0,0.4)
ax.set_ylabel(r'proportion infected', labelpad = 3)

ax2.set_ylim(0,800)
ax2.set_ylabel(r'total deaths/100,000', color = 'darkred')
ax2.tick_params(axis='y', labelcolor = 'darkred')   

plt.legend(frameon = False, loc = 5, fontsize = 6)
plt.tight_layout()
plt.margins(0,0)
plt.savefig('new_york_cases.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)

# plot reproduction number estimate
fig, ax = plt.subplots()

ax.set_xlim([dt.date(2020, 3, 1), dt.date(2020, 6, 7)])

plt.plot(simulation_dates+dt.timedelta(days = 9), np.asarray(daily_average['E'])/np.asarray(daily_average['I']), 'k')

plt.plot([dt.date(2020, 3, 1), dt.date(2020, 6, 7)], [1,1], linestyle = '--', color = 'Grey')

ax.set_xticklabels(NYC_date_of_interest[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylabel(r'$R(t)$')

ax.set_ylim(0,4)

plt.tight_layout()
plt.margins(0,0)
plt.savefig('reproduction_number.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)

# plot all model compartments
fig, axs = plt.subplots(nrows=3, sharex=True)

plt.sca(axs[0])
plt.plot(start_times, np.array(mean_contact_duration) / minute)
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

image_path = ("simple_epidemic_with_slow_contact_simulator_" + 
              "maxlambda_{:d}.png".format(contact_simulator.mean_contact_rate.maximum_i))

print("Saving a visualization of results at", image_path)
plt.savefig(image_path, dpi=480)
