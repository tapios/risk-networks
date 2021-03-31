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
parser.add_argument('--parallel-memory', type=int, default=4_000_000_000) # 4GB
parser.add_argument('--parallel-temp-dir', type=str, default='')

# constants ####################################################################
parser.add_argument('--constants-seed-shift', type=int, default=0)
parser.add_argument('--constants-output-path', type=str, default='')
parser.add_argument('--constants-save-path', type=str, default='')

# network ######################################################################
parser.add_argument('--network-node-count', type=str, default='1e3')

# user_network #################################################################
parser.add_argument('--user-network-user-fraction', type=float, default=1.0)
parser.add_argument('--user-network-seed-user', type=int, default=190)
parser.add_argument('--user-network-type', type=str, default='random')
parser.add_argument('--user-network-weighted', default=False, action='store_true')
parser.add_argument('--user-network-weight-factor', type=float, default=1.0)

# transformation #################################################################
parser.add_argument('--transform-lengthscale', type=float, default=0.7)

# observations #################################################################
parser.add_argument('--observations-I-fraction-tested', type=float, default=0)
parser.add_argument('--observations-I-budget', type=int, default=0)
parser.add_argument('--observations-I-min-threshold', type=float, default=0.0)
parser.add_argument('--observations-I-max-threshold', type=float, default=1.0)
parser.add_argument('--observations-sensor-wearers', type=int, default=0)
parser.add_argument('--observations-noise', type=float, default=1e-4)
parser.add_argument('--observations-true-prevalence', default=False, action='store_true')

# data assimilation ############################################################
parser.add_argument('--assimilation-update-sensor', type=str, default='full_global')
parser.add_argument('--assimilation-update-test',   type=str, default='full_global')
parser.add_argument('--assimilation-update-record', type=str, default='full_global')
parser.add_argument('--assimilation-batches-sensor', type=int, default=1)
parser.add_argument('--assimilation-batches-test', type=int, default=1)
parser.add_argument('--assimilation-batches-record', type=int, default=1)
parser.add_argument('--sensor-assimilation-joint-regularization', type=float, default=1e-2)
parser.add_argument('--test-assimilation-joint-regularization',   type=float, default=1e-2)
parser.add_argument('--record-assimilation-joint-regularization', type=float, default=1e-2)
parser.add_argument('--sensor-assimilation-obs-regularization', type=float, default=0)
parser.add_argument('--test-assimilation-obs-regularization',   type=float, default=0)
parser.add_argument('--record-assimilation-obs-regularization', type=float, default=0)
parser.add_argument('--sensor-assimilation-elementwise-regularization', default=False, action='store_true')
parser.add_argument('--test-assimilation-elementwise-regularization', default=False, action='store_true')
parser.add_argument('--record-assimilation-elementwise-regularization', default=False, action='store_true')
parser.add_argument('--assimilation-inflation', default=True, action='store_false')
parser.add_argument('--assimilation-additive-inflation', default=False, action='store_true')
parser.add_argument('--assimilation-additive-inflation-factor', type=float, default=0.1)
parser.add_argument('--assimilation-sensor-inflation', type=float, default=1.0)
parser.add_argument('--assimilation-test-inflation',   type=float, default=1.0)
parser.add_argument('--assimilation-record-inflation', type=float, default=1.0)
parser.add_argument('--assimilation-inflate-I-only', default=True, action='store_false')
parser.add_argument('--distance-threshold', type=int, default=1)
parser.add_argument('--assimilation-window', type=float, default=1.0)
parser.add_argument('--assimilation-sweeps', type=int, default=1)
parser.add_argument('--sensor-ignore-mass-constraint',default=False, action='store_true')
parser.add_argument('--test-ignore-mass-constraint',  default=False, action='store_true')
parser.add_argument('--record-ignore-mass-constraint',default=False, action='store_true')

#prior run
parser.add_argument('--prior-run', default=False, action='store_true')

# parameters learning ##########################################################
parser.add_argument('--params-learn-transition-rates', default=False, action='store_true')
parser.add_argument('--params-transition-rates-str', type=str, default='latent_periods,community_infection_periods')

parser.add_argument('--params-learn-transmission-rate', default=False,action='store_true')
parser.add_argument('--params-transmission-rate-bias', type=float, default=0.0)
parser.add_argument('--params-transmission-rate-noise', type=float, default=0.1)
parser.add_argument('--params-transmission-inflation', type=float, default=1.0)

# interventions ################################################################
parser.add_argument('--intervention-frequency', type=str, default='none')
parser.add_argument('--intervention-nodes', type=str, default='all')
parser.add_argument('--intervention-type', type=str, default='social_distance')

parser.add_argument('--intervention-E-min-threshold', type=float, default=0.999)#1.0 not allowed...
parser.add_argument('--intervention-I-min-threshold', type=float, default=0.999)
parser.add_argument('--intervention-sum-EI', default=False, action='store_true')
parser.add_argument('--intervention-random-isolate-budget', type=int, default=300)
parser.add_argument('--intervention-isolate-node-lambda', type=float, default=2.0)
parser.add_argument('--intervention-start-time', type=float, default=10.0)
parser.add_argument('--intervention-interval', type=float, default=1.0)
parser.add_argument('--intervention-sick-isolate-time', type=float, default=7.0)

parser.add_argument('--intervention-contact-trace-days', type=float, default=10.0)

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
parser.add_argument('--ic-alpha', type=float, default=0.0025)
parser.add_argument('--ic-beta', type=float, default=1)

#for 1% IC infected
#parser.add_argument('--ic-alpha', type=float, default=0.04)
#parser.add_argument('--ic-beta', type=float, default=4)

# parser setup #################################################################
arguments = parser.parse_args()
print_info_module(__name__, arguments)

################################################################################
print_end_of(__name__)

