#!/usr/bin/python3 --

import os, sys; sys.path.append(os.path.join(".."))

from epiforecast.temporal_adjacency import TemporalAdjacency
from epiforecast.kinetic_model_simulator import KineticModel
from epiforecast.kinetic_model_helper import KM_print_states, KM_print_start


################################################################################
# constants section ############################################################
################################################################################
# general
DEBUG = False

# simulation; all times in [day] units
T0 = 0.0      # start time
T1 = 25.0      # end time
dt_KM = 0.125 # time step
t = T0        # current time

# output
output_dt = 1.0 # how often to do outputs
output_t = T0   # current output time

# parameters
beta = 1.0             # [1/day] the beta from overleaf; one global scalar
alpha_hosp = 0.1       # [1] fraction of beta for healtcare workers
initial_active = 0.034 # [1] fraction of initially active edges
mu = 1920              # [1/day] (mean contact duration)**(-1)

################################################################################
# main section #################################################################
################################################################################
edges_filename = os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt')

ta = TemporalAdjacency(edges_filename, initial_active, mu)

ta.generate() # MC generation of active edges over the day
ta.average_wjis(dt_averaging=dt_KM) # averaging those over dt_KM intervals
ta.multiply_wjis(beta, beta * alpha_hosp) # wji *= beta, wjip *= beta*alpha_hosp

km = KineticModel(ta.edge_list)
km.set_IC(
    I0 = range(len(km.static_graph) // 2, len(km.static_graph) // 2 + 10)
)

# XXX the following code will be fixed at the next iteration of PR's
km.set_ages() # should go away completely
km.set_independent_rates() # should be: km.set_independent_rates(clinical_rates)

km.set_return_statuses('all') # can be 'SIR' or 'HRD' etc.

KM_print_start(t, km.IC, 'SEIHRD')
while t < T1:
  beta_dict, betap_dict = ta.get_wjis(t, structure='dict')
  km.update_beta_rates(beta_dict, betap_dict)

  res = km.do_Gillespie_step(t=t, dt=dt_KM)

  times, states = res.summary()
  node_status = res.get_statuses(time=times[-1])

  km.update_IC(node_status)
  km.vacate_placeholder() # remove from hospital whoever recovered/died
  km.populate_placeholder() # move into hospital those who need it

  if t >= output_t:
    KM_print_states(t, states, 'SEIHRD')
    output_t += output_dt
    if states['E'][-1] + states['I'][-1] + states['H'][-1] == 0:
      # stop the simulation if infection has died out
      break

  t += dt_KM


