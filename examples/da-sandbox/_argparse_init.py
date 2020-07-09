import os
import argparse

from _utilities import print_start_of, print_end_of, print_info


print_start_of(__name__)
################################################################################
parser = argparse.ArgumentParser()

print_info(__name__, "parsing args of PID:", os.getpid())

# user_network #################################################################
parser.add_argument('--user-network-user-fraction', type=float, default=1.0)
parser.add_argument('--user-network-seed-user', type=int, default=190)

# constants ####################################################################
parser.add_argument('--constants-output-path', type=str, default="")

# observations #################################################################
parser.add_argument('--observations-I-fraction-tested', type=float, default=0.1)
parser.add_argument('--observations-I-min-threshold', type=float, default=0.3)


# parser setup #################################################################
arguments = parser.parse_args()
print_info(__name__, arguments)

################################################################################
print_end_of(__name__)

