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

from numba import set_num_threads

set_num_threads(1)

from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant

from epiforecast.scenarios import load_edges, random_epidemic

from epiforecast.epiplots import plot_ensemble_states, plot_kinetic_model_data, plot_scalar_parameters

from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.performance_metrics import PerformanceTracker
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.measurements import Observation, DataObservation, DataNodeObservation
from epiforecast.data_assimilator import DataAssimilator

from epiforecast.risk_simulator_initial_conditions import random_risk, uniform_risk, deterministic_risk, prevalence_random_risk, prevalence_deterministic_risk
from epiforecast.utilities import seed_numba_random_state

"""
Set random seeds for reproducibility -----------------------------------------
"""
seed = 212212
np.random.seed(seed)
random.seed(seed)
seed_numba_random_state(seed)

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

mkdir_p('da_obs/performance/')
mkdir_p('da_obs/epidemic/')

"""
Load an example network ------------------------------------------------------
"""

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3_nobeds.txt'))
node_identifiers = load_node_identifiers(os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e3_nobeds.txt'))

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

"""
Build the contact simulator --------------------------------------------------
"""

start_time = -3 / 24
minute = 1 / 60 / 24
hour = 60 * minute

"""
Clinical parameters of an age-distributed population -------------------------
"""

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

community_transmission_rate = 12.0

"""
Define time horizon parameters------------------------------------------------
"""
static_contact_interval = 3 * hour
simulation_length = 3

"""
Initalize health service and epidemic simulator ------------------------------
"""

health_service = HealthService(static_population_network = contact_network,
                               health_workers = node_identifiers['health_workers'],
                               health_workers_per_patient=5)

mean_contact_lifetime=0.5*minute
hospital_transmission_reduction = 0.1


epidemic_simulator = EpidemicSimulator(
                 contact_network = contact_network,
                transition_rates = transition_rates,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
         static_contact_interval = static_contact_interval,
           mean_contact_lifetime = mean_contact_lifetime,
              day_inception_rate = 22,
            night_inception_rate = 2,
                  health_service = health_service,
                      start_time = start_time
                                      )
"""
Initialize ensemble ----------------------------------------------------------
"""

ensemble_size = 20

transition_rates_ensemble = []
community_transmission_rate_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(

        TransitionRates(contact_network,
                        latent_periods = transition_rates.latent_periods, # np.random.normal(3.7,0.37),
                        community_infection_periods = transition_rates.community_infection_periods, # np.random.normal(3.2, 0.32),
                        hospital_infection_periods  = transition_rates.hospital_infection_periods, # np.random.normal(5.0, 0.50),
                        hospitalization_fraction    = transition_rates.hospitalization_fraction,
                        community_mortality_fraction = transition_rates.community_mortality_fraction,
                        hospital_mortality_fraction  = transition_rates.hospital_mortality_fraction
                       )
        )
    community_transmission_rate_ensemble.append(community_transmission_rate)

master_eqn_ensemble = MasterEquationModelEnsemble(contact_network = contact_network,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size,
                                                  start_time=0.0)

"""
Observations objects ---------------------------------------------------------
"""
obs_frac = 0.30

medical_infection_test = Observation(N = population,
                                     obs_frac = obs_frac,
                                     obs_status = 'I',
                                     obs_name = "0.01 < Infected(100%) < 0.25",
                                     min_threshold=0.00,
                                     max_threshold=1.00)
                                     # sensitivity = 0.99)

random_infection_test = Observation(N = population,
                                     obs_frac = obs_frac,
                                     obs_status = 'I',
                                     obs_name = str(int(obs_frac*100)).zfill(3)+"randinf")

positive_hospital_records = DataObservation(N = population,
                                       set_to_one=True,
                                       obs_status = 'H',
                                       obs_name = "hospstate")

positive_death_records = DataObservation(N = population,
                                    set_to_one=True,
                                    obs_status = 'D',
                                    obs_name = "deathstate")

# positive_hospital_records = DataNodeObservation(N = population,
#                                         bool_type = True,
#                                        obs_status = 'H',
#                                          obs_name = "hospnode")
#
# positive_death_records = DataNodeObservation(N = population,
#                                     bool_type  = True,
#                                     obs_status = 'D',
#                                     obs_name   = "deathnode")

negative_hospital_records = DataObservation(N = population,
                                    set_to_one=False,
                                    obs_status = 'H',
                                    obs_name = "nohospstate")

negative_death_records = DataObservation(N = population,
                                    set_to_one=False,
                                    obs_status = 'D',
                                    obs_name = "nodeathstate")

# observations=[positive_death_records,
#               negative_death_records,
#               positive_hospital_records,
#               negative_hospital_records]
#
# observations=[medical_infection_test,
#               random_infection_test,
#               positive_death_records,
#               positive_hospital_records]

observations=[random_infection_test,
              positive_death_records,
              positive_hospital_records]

# observations=[random_infection_test]

"""
Initialize the Data Assimilator ------------------------------------------------
"""
# give the data assimilator which transition rates and transmission rate to assimilate
transition_rates_to_update_str=[]#'latent_periods', 'community_infection_periods', 'hospital_infection_periods']
transmission_rate_to_update_flag=False #True

# create the assimilator
assimilator = DataAssimilator(observations = observations,
                              errors = [],
                              transition_rates_to_update_str= transition_rates_to_update_str,
                              transmission_rate_to_update_flag = transmission_rate_to_update_flag)

"""
Create initial conditions ------------------------------------------------------
"""

time = start_time
statuses = random_epidemic(contact_network,
                           fraction_infected=0.01)

states_ensemble, risk_str = random_risk(contact_network,
                              fraction_infected = 0.01,
                              ensemble_size = ensemble_size)

# states_ensemble, risk_str = uniform_risk(contact_network,
#                               fraction_infected = 0.01,
#                               ensemble_size = ensemble_size)

# states_ensemble, risk_str = deterministic_risk(contact_network,
#                               statuses,
#                               ensemble_size = ensemble_size)

# states_ensemble, risk_str = prevalence_random_risk(contact_network,
#                               fraction_infected = 0.01,
#                               ensemble_size = ensemble_size)

# states_ensemble, risk_str = prevalence_deterministic_risk(contact_network,
#                               statuses,
#                               ensemble_size = ensemble_size)

epidemic_simulator.set_statuses(statuses)
master_eqn_ensemble.set_states_ensemble(states_ensemble)

"""
Setup for output name ----------------------------------------------------------
"""
contact_interval_str = str(round(static_contact_interval/hour)).zfill(2)+'hrs'
simulation_length_str = str(simulation_length).zfill(3)+'days'
obs_str = '_'.join([obs.name for obs in observations])

plot_name_observations = risk_str + '_' + obs_str + '_' + simulation_length_str + '_' + contact_interval_str

"""
Simulation code ----------------------------------------------------------------

As some other functions above use the random seeds, I need to set them again for
reproducibility.

"""

np.random.seed(seed)
random.seed(seed)
seed_numba_random_state(seed)

fig, axes = plt.subplots(1, 3, figsize = (16, 4))

community_transmission_rate_trace = np.copy(community_transmission_rate_ensemble)
latent_periods_trace              = np.copy(np.array([member.latent_periods for member in transition_rates_ensemble]).reshape(-1,1))
community_infection_periods_trace = np.copy(np.array([member.community_infection_periods for member in transition_rates_ensemble]).reshape(-1,1))
hospital_infection_periods_trace  = np.copy(np.array([member.hospital_infection_periods for member in transition_rates_ensemble]).reshape(-1,1))

model_performance = PerformanceTracker()

for i in range(int(simulation_length/static_contact_interval)):

    epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval)
    #Within the epidemic_simulator:
    # health_service discharge and admit patients [changes the contact network]
    # contact_simulator run [changes the mean contact duration on the given network]
    # set the new contact rates on the network! kinetic_model.set_mean_contact_duration(contact_duration)
    # run the kinetic model [kinetic produces the current statuses used as data]

    # as kinetic sets the weights, we do not need to update the contact network.
    # run the master equation model [master eqn produces the current states of the risk model]
    master_eqn_ensemble.set_mean_contact_duration()
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)

    if i % 1 == 0:
    # perform data assimlation [update the master eqn states, the transition rates, and the transmission rate (if supplied)]
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
         ) = assimilator.update(ensemble_state = states_ensemble,
                                data = epidemic_simulator.kinetic_model.current_statuses,
                                full_ensemble_transition_rates = transition_rates_ensemble,
                                full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                user_network = contact_network)

        # update model parameters (transition and transmission rates) of the master eqn model
        master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
        master_eqn_ensemble.update_transmission_rate(community_transmission_rate_ensemble)


        # for tracking and plotting purposes
        community_transmission_rate_trace = np.hstack([community_transmission_rate_trace, community_transmission_rate_ensemble])
        latent_periods_trace              = np.hstack([latent_periods_trace, np.array([member.latent_periods for member in transition_rates_ensemble]).reshape(-1,1)])
        community_infection_periods_trace = np.hstack([community_infection_periods_trace, np.array([member.community_infection_periods for member in transition_rates_ensemble]).reshape(-1,1)])
        hospital_infection_periods_trace  = np.hstack([hospital_infection_periods_trace, np.array([member.hospital_infection_periods for member in transition_rates_ensemble]).reshape(-1,1)])


    model_performance.update(epidemic_simulator.kinetic_model.current_statuses, states_ensemble)
    print(model_performance)

    #update states/statuses/times for next iteration
    master_eqn_ensemble.set_states_ensemble(states_ensemble)

    axes = plot_ensemble_states(master_eqn_ensemble.states_trace,
                                master_eqn_ensemble.simulation_time,
                                axes = axes,
                                xlims = (-0.1, simulation_length),
                                a_min = 0.0)

    axes = plot_kinetic_model_data(epidemic_simulator.kinetic_model,
                                   axes = axes)
    plt.savefig('da_obs/epidemic/'+plot_name_observations+'.png', rasterized=True, dpi=150)

"""
Additional plots: parameters --------------------------------------------------
"""



# parameters = [community_transmission_rate_trace, latent_periods_trace, community_infection_periods_trace, hospital_infection_periods_trace ]
# parameters_names = ['transmission_rates', 'latent_periods', 'community_infection_periods', 'hospital_infection_periods']
#
# axes = plot_scalar_parameters(parameters, time_horizon, parameters_names)
# plt.savefig('da_parameters_tprobs_'+plot_name_observations +'.png', rasterized=True, dpi=150)

"""
Additional plots: Performance scores -------------------------------------------
"""
time_horizon = np.linspace(0.0, simulation_length, int(simulation_length/static_contact_interval))

fig, axes =  plt.subplots(1, 2, figsize = (12, 4))
axes[0].plot(time_horizon, model_performance.performance_track[:,0])
axes[0].plot(time_horizon, 1 - model_performance.prevalence_track, linestyle = '--', color = 'red')

axes[1].plot(time_horizon, model_performance.performance_track[:,1])

axes[0].set_title("Accuracy")
axes[1].set_title("F1-Score")
plt.savefig('da_obs/performance/performance_'+plot_name_observations +'.png', rasterized=True, dpi=150)


print("See figures: " + plot_name_observations)
