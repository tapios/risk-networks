#!/usr/bin/python3 --

import os, sys; sys.path.append(os.path.join("..", ".."))


import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from timeit import default_timer as timer
from numba import set_num_threads

from epiforecast.populations import assign_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant
from epiforecast.scenarios import load_edges, random_epidemic
from epiforecast.epiplots import plot_master_eqns
from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.measurements import Observation
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.intervention import Intervention
from epiforecast.utilities import seed_three_random_states



def random_risk(contact_network, fraction_infected=0.01, ensemble_size=1):
  population = len(contact_network)
  ensemble_states = np.zeros([ensemble_size, 5 * population])
  for mm in range(ensemble_size):
    infected = np.random.choice(population, replace=False, size=int(population * fraction_infected))
    E, I, H, R, D = np.zeros([5, population])
    S = np.ones(population,)
    I[infected] = 1.
    S[infected] = 0.
    ensemble_states[mm, : ] = np.hstack((S, I, H, R, D))

  return ensemble_states

################################################################################
# constants ####################################################################
################################################################################
SAVE_FLAG = False
PLOT_FLAG = False
NETWORKS_PATH = os.path.join('..', '..', 'data', 'networks')
SIMULATION_PATH = os.path.join('..', '..', 'data', 'simulation_data')
FIGURES_PATH = os.path.join('..', '..', 'figs')

minute = 1 / 60 / 24
hour = 60 * minute
day = 1.0

start_time = -3 / 24
simulation_length = 3 # number of days

community_transmission_rate = 12.0
hospital_transmission_reduction = 0.1
static_contact_interval = 3 * hour
mean_contact_lifetime = 0.5 * minute
ensemble_size = 100
compartment_index = { # the layout of states in 'ensemble_states'
    'S' : 0,
    'E' : -1,
    'I' : 1,
    'H' : 2,
    'R' : 3,
    'D' : 4}

# 5 age groups (0-17, 18-44, 45-64, 65-74, >=75) and their respective rates
age_distribution = [0.207, 0.400, 0.245, 0.083, 0.065]
age_dep_h      = [0.002   ,  0.010  ,  0.040,  0.076,  0.160]
age_dep_d      = [0.000001,  0.00001,  0.001,  0.007,  0.015]
age_dep_dprime = [0.019   ,  0.073  ,  0.193,  0.327,  0.512]

assert sum(age_distribution) == 1.0

################################################################################
# initialization ###############################################################
################################################################################
# numba
set_num_threads(1)

# set random seeds for reproducibility
seed = 2132
seed_three_random_states(seed)

# load network topology
edges = load_edges(
    os.path.join(NETWORKS_PATH, 'edge_list_SBM_1e3_nobeds.txt'))
node_identifiers = load_node_identifiers(
    os.path.join(NETWORKS_PATH, 'node_identifier_SBM_1e3_nobeds.txt'))

# global variable to store the network; will be changing during simulation
contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

# clinical parameters of an age-distributed population
assign_ages(contact_network, distribution=age_distribution)

# stochastic model #############################################################
# transition rates between each epidemiological state
transition_rates = TransitionRates(
    contact_network,
    latent_periods = 3.7,
    community_infection_periods = 3.2,
    hospital_infection_periods = 5.0,
    hospitalization_fraction = AgeDependentConstant(age_dep_h),
    community_mortality_fraction = AgeDependentConstant(age_dep_d),
    hospital_mortality_fraction = AgeDependentConstant(age_dep_dprime))

# simulate the growth and equilibration of an epidemic
health_service = HealthService(
    static_population_network = contact_network,
    health_workers = node_identifiers['health_workers'])

epidemic_simulator = EpidemicSimulator(
    contact_network,
    transition_rates = transition_rates,
    community_transmission_rate = community_transmission_rate,
    hospital_transmission_reduction = hospital_transmission_reduction,
    static_contact_interval = static_contact_interval,
    mean_contact_lifetime = mean_contact_lifetime,
    day_inception_rate = 22,
    night_inception_rate = 2,
    health_service = health_service,
    start_time = start_time)

# master equations #############################################################
transition_rates_ensemble = []

for i in range(ensemble_size):
  transition_rates_ensemble.append(
      TransitionRates(
        contact_network,
        latent_periods=np.random.normal(3.7,0.37),
        community_infection_periods=np.random.normal(3.2,0.32),
        hospital_infection_periods=np.random.normal(5.0,0.5),
        hospitalization_fraction=AgeDependentBetaSampler(mean=age_dep_h, b=4),
        community_mortality_fraction=AgeDependentBetaSampler(mean=age_dep_d, b=4),
        hospital_mortality_fraction=AgeDependentBetaSampler(mean=age_dep_dprime, b=4)))

community_transmission_rate_ensemble = np.random.normal(12.0, 1.0, size=(ensemble_size,1))

master_eqn_ensemble = MasterEquationModelEnsemble(
    contact_network=contact_network,
    transition_rates=transition_rates_ensemble,
    transmission_rate=community_transmission_rate_ensemble,
    hospital_transmission_reduction=hospital_transmission_reduction,
    ensemble_size=ensemble_size)

# assimilator ##################################################################
# methods for how to choose observed states
medical_infection_test = Observation(
    N=population,
    obs_frac=1.0,
    obs_status='I',
    obs_name="0.25 < Infected(100%) < 0.5",
    min_threshold=0.1,
    max_threshold=0.7)

observations = [medical_infection_test]

# which transition rates and transmission rate to assimilate
transition_rates_to_update_str   = ['latent_periods','hospitalization_fraction']
transmission_rate_to_update_flag = True

assimilator = DataAssimilator(
    observations=observations,
    errors=[],
    transition_rates_to_update_str=transition_rates_to_update_str,
    transmission_rate_to_update_flag=transmission_rate_to_update_flag)

# intervention #################################################################
intervention = Intervention(
    population,
    ensemble_size,
    compartment_index,
    E_thr = 0.5,
    I_thr = 0.5)

# initial conditions ###########################################################
statuses = random_epidemic(contact_network, fraction_infected=0.01)
ensemble_states = random_risk(
    contact_network,
    fraction_infected=0.01,
    ensemble_size=ensemble_size)
epidemic_simulator.set_statuses(statuses)
master_eqn_ensemble.set_states_ensemble(ensemble_states)

################################################################################
# main loop ####################################################################
################################################################################
for i in range(int(simulation_length/static_contact_interval)):
  # Within the epidemic_simulator:
  # health_service discharge and admit patients [changes the contact network]
  # contact_simulator run [changes the mean contact duration on the network]
  # set the new contact rates on the network
  # run the kinetic model [kinetic produces the current statuses used as data]
  epidemic_simulator.run(
      stop_time=epidemic_simulator.time + static_contact_interval)

  # as kinetic sets the weights, we do not need to update the contact network
  # run the master equation model
  master_eqn_ensemble.set_mean_contact_duration()
  ensemble_states = master_eqn_ensemble.simulate(static_contact_interval)

  # perform data assimilation:
  # update the master eqn states
  # update the transition rates
  # update the transmission rate (if supplied)
  (ensemble_states,
   transition_rates_ensemble,
   community_transmission_rate_ensemble
   ) = assimilator.update(
       ensemble_states,
       epidemic_simulator.kinetic_model.current_statuses,
       full_ensemble_transition_rates=transition_rates_ensemble,
       full_ensemble_transmission_rate=community_transmission_rate_ensemble,
       user_network=contact_network)

  # update master equation ensemble
  master_eqn_ensemble.update_transition_rates(transition_rates_ensemble)
  master_eqn_ensemble.update_transmission_rate(community_transmission_rate_ensemble)
  master_eqn_ensemble.set_states_ensemble(ensemble_states)

  # intervention
  sick_nodes = intervention.find_sick(ensemble_states)
  intervention.apply_intervention(contact_network, sick_nodes)
  print("Sick nodes: {:d}/{:d}".format(sick_nodes.size, population))

  print("Current time", epidemic_simulator.time)


################################################################################
# save #########################################################################
################################################################################
if SAVE_FLAG:
  np.savetxt(
      os.path.join(SIMULATION_PATH, 'NYC_DA_interventions_1e3.txt'),
      np.c_[
        kinetic_model.times,
        kinetic_model.statuses['S'],
        kinetic_model.statuses['E'],
        kinetic_model.statuses['I'],
        kinetic_model.statuses['H'],
        kinetic_model.statuses['R'],
        kinetic_model.statuses['D']],
      header = 'S E I H R D seed: %d'%seed)

################################################################################
# plot #########################################################################
################################################################################
if PLOT_FLAG:
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

  image_path = os.path.join(
      FIGURES_PATH, 'simple_epidemic_with_slow_contact_simulator_',
      'maxlambda_{:d}.png'.format(contact_simulator.mean_contact_rate.maximum_i))

  print("Saving a visualization of results at", image_path)
  plt.savefig(image_path, dpi=480)


