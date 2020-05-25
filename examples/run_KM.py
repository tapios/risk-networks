#!/usr/bin/python3 --

import sys
import getopt
import configparser

import os, sys; sys.path.append(os.path.join(".."))

from epiforecast.KineticModel import KineticModel
from epiforecast.KM_helper import KM_print_states, KM_print_start


################################################################################
# parser section ###############################################################
################################################################################
try:
  opts, args = getopt.getopt(sys.argv[1:], "c:", ['config='])
except getopt.GetoptError as e:
  print('Something wrong with getopt')
  raise

confname = ''
if not opts:
  confname = input('Config filename: ')
else:
  for opt, arg in opts:
    if opt in ('-c', '--config'):
      confname = arg

try:
  config = configparser.RawConfigParser()
  if not config.read(confname):
    raise IOError('File "{0}" not found'.format(confname))
except:
  raise

################################################################################
# constants section ############################################################
################################################################################
try:
  # general
  DEBUG = config.getboolean('GENERAL', 'DEBUG', fallback = False)

  # simulation
  T0 = config.getfloat('SIMULATION', 'T0', fallback=0.0)  # start time
  T1 = config.getfloat('SIMULATION', 'T1', fallback=50.0) # end time
  dt_KM = config.getfloat('SIMULATION', 'dt_KM', fallback=0.125) # time step

  # output
  output_dt = config.getfloat('OUTPUT', 'output_dt', fallback=1.0) # how often to do outputs
except (configparser.NoOptionError, configparser.NoSectionError):
  raise

t = T0 # current time
output_t = T0 # current output time

################################################################################
# main section #################################################################
################################################################################
km = KineticModel()
km.load_edge_list()
km.set_IC(
    I0 = range(len(km.static_graph) // 2, len(km.static_graph) // 2 + 10)
)
km.set_ages()
km.set_independent_rates()
km.set_return_statuses('all') # can be 'SIR' or 'HRD' etc.

# this is a dubious point... muc used to be 6 but in units [1/h]
# we have now switched to days throughout the code everywhere, so should it be
# muc = 144 instead? with 144 you get waaaay fewer contacts
# even worse, it's 1920 per day now? according to overleaf?
km.generate_temporal_adjacency(muc=6)

km.average_betas(dt_averaging=dt_KM)

KM_print_start(t, km.IC, 'SEIRHD')
while t < T1:
  km.update_beta_rates(t)
  res = km.do_Gillespie_step(t=t, dt=dt_KM)

  times, states = res.summary()
  node_status = res.get_statuses(time=times[-1])

  km.update_IC(node_status)
  km.vacate_placeholder()
  km.populate_placeholder()

  if t >= output_t:
    KM_print_states(t, states, 'SEIRHD')
    output_t += output_dt

  t += dt_KM


