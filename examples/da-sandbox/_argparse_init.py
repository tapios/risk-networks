import os
import argparse

from _utilities import print_start_of, print_end_of, print_info_module


print_start_of(__name__)
################################################################################
parser = argparse.ArgumentParser()

print_info_module(__name__, "parsing args of PID:", os.getpid())

# constants ####################################################################
parser.add_argument('--constants-output-path', type=str, default='')

# network ######################################################################
parser.add_argument('--network-node-count', type=str, default='1e3')

# user_network #################################################################
parser.add_argument('--user-network-user-fraction', type=float, default=1.0)
parser.add_argument('--user-network-seed-user', type=int, default=190)

# observations #################################################################
parser.add_argument('--observations-I-fraction-tested', type=float, default=0.01)
parser.add_argument('--observations-I-budget', type=int, default=10)
parser.add_argument('--observations-I-min-threshold', type=float, default=0.0)
parser.add_argument('--observations-I-max-threshold', type=float, default=1.0)


# parser setup #################################################################
arguments = parser.parse_args()
print_info_module(__name__, arguments)

################################################################################
print_end_of(__name__)

