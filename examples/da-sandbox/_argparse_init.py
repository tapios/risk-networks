import os
import argparse

from _utilities import print_start_of, print_end_of, print_info_module


print_start_of(__name__)
################################################################################
parser = argparse.ArgumentParser()

print_info_module(__name__, "parsing args of PID:", os.getpid())

# parallel #####################################################################
parser.add_argument('--parallel-flag', default=False, action='store_true')
parser.add_argument('--parallel-num-cpus', type=int, default=1)

# constants ####################################################################
parser.add_argument('--constants-output-path', type=str, default='')
parser.add_argument('--constants-save-path', type=str, default='')

# network ######################################################################
parser.add_argument('--network-node-count', type=str, default='1e3')

# user_network #################################################################
parser.add_argument('--user-network-user-fraction', type=float, default=1.0)
parser.add_argument('--user-network-seed-user', type=int, default=190)

# observations #################################################################
parser.add_argument('--observations-I-fraction-tested', type=float, default=0)
parser.add_argument('--observations-I-budget', type=int, default=0)
parser.add_argument('--observations-I-min-threshold', type=float, default=0.0)
parser.add_argument('--observations-I-max-threshold', type=float, default=1.0)
parser.add_argument('--observations-sensor-wearers', type=int, default=0)

# data assimilation ############################################################
parser.add_argument('--assimilation-batches-sensor', type=int, default=1)
parser.add_argument('--assimilation-batches-test', type=int, default=1)
parser.add_argument('--assimilation-batches-record', type=int, default=1)

# interventions ################################################################
parser.add_argument('--intervention-frequency', type=str, default='none')
parser.add_argument('--intervention-nodes', type=str, default='all')
parser.add_argument('--intervention-type', type=str, default='social_distance')

parser.add_argument('--intervention-E-min-threshold', type=float, default=0.999)#1.0 not allowed...
parser.add_argument('--intervention-I-min-threshold', type=float, default=0.999)
parser.add_argument('--intervention-start-time', type=float, default=10.0)
parser.add_argument('--intervention-interval', type=float, default=1.0)

# epidemic #####################################################################
parser.add_argument('--epidemic-load-data', default=True, action='store_false')
parser.add_argument('--epidemic-save-data', default=False, action='store_true')
parser.add_argument('--epidemic-storage-name',
                    type=str,
                    default='epidemic_storage.pkl')
parser.add_argument('--epidemic-kinetic-states-name',
                    type=str,
                    default='epidemic_kinetic_states.pkl')

# initial condition ############################################################
parser.add_argument('--ic-alpha', type=float, default=0.04)
parser.add_argument('--ic-beta', type=float, default=4)

# parser setup #################################################################
arguments = parser.parse_args()
print_info_module(__name__, arguments)

################################################################################
print_end_of(__name__)

