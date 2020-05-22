#!/usr/bin/python3 --

import sys
import getopt
import configparser

from KineticModel import KineticModel
from KM_helper import KM_print_states


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
km.set_statuses('all') # can be 'SIR' or 'HRD' etc.

# this is a dubious point... muc used to be 6 but in units [1/h]
# we have now switched to days throughout the code everywhere, so should it be
# muc = 144 instead? with 144 you get waaaay fewer contacts
km.generate_temporal_adjacency(muc=6)

km.average_betas(dt_averaging=dt_KM)

while t < T1:
  km.update_beta_rates(t)
  res = km.do_Gillespie_step(t=t, dt=dt_KM)

  times, states = res.summary()
  node_status = res.get_statuses(time=times[-1])
  KM_print_states(t, times, states, 'SEIRHD')

  if t >= output_t:
    print('t = {:>7.2f}'.format(t))
    output_t += output_dt

  t += dt_KM

