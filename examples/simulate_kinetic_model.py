#!/usr/bin/python3 --

import os, sys; sys.path.append(os.path.join(".."))

from epiforecast.kinetic_model_simulator import KineticModel
from epiforecast.kinetic_model_helper import KM_print_states, KM_print_start


################################################################################
# constants section ############################################################
################################################################################
# general
DEBUG = False

# simulation; all times in [day] units
T0 = 0.0      # start time
T1 = 5.0      # end time
dt_KM = 0.125 # time step
t = T0        # current time

# output
output_dt = 1.0 # how often to do outputs
output_t = T0   # current output time

################################################################################
# main section #################################################################
################################################################################
km = KineticModel()
km.load_edge_list()
km.set_IC(
    I0 = range(len(km.static_graph) // 2, len(km.static_graph) // 2 + 10)
)
km.set_ages() # this should read ages in instead
km.set_independent_rates() # this should read rates in instead
km.set_return_statuses('all') # can be 'SIR' or 'HRD' etc.

km.generate_temporal_adjacency(muc=1920)
km.average_betas(dt_averaging=dt_KM)

KM_print_start(t, km.IC, 'SEIHRD')
while t < T1:
  km.update_beta_rates(t)
  res = km.do_Gillespie_step(t=t, dt=dt_KM)

  times, states = res.summary()
  node_status = res.get_statuses(time=times[-1])

  km.update_IC(node_status)
  km.vacate_placeholder()
  km.populate_placeholder()

  if t >= output_t:
    KM_print_states(t, states, 'SEIHRD')
    output_t += output_dt

  t += dt_KM


