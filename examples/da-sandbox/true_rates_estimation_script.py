import os
import sys
import time
import subprocess
import numpy as np
from timeit import default_timer as timer

# functions ####################################################################
def time_elapsed():
    global WALL_START_TIME
    return timer() - WALL_START_TIME

def print_info(*args, **kwargs):
    global WALL_START_TIME

    print('[{:>10.2f}] '.format(time_elapsed()), end='')
    print(*args, **kwargs, flush=True)


# main #########################################################################
WALL_START_TIME = timer()
SLEEP_INTERVAL_MAX = 60
OUTPUT_PATH = 'output'
EXEC = ['python3', 'true_rates_estimation.py']
CONST_ARGUMENTS = ['--observations-I-min-threshold', '0.0',
                   '--network-node-count', '1e4']

sleep_interval = 1
children = []

user_fractions = np.array([0.03, 0.05, 0.1, 0.5, 1.0])
fractions_tested = np.array([0.5, 0.2, 0.1, 0.02, 0.01])
for user_fraction, tested in zip(user_fractions, fractions_tested):
    CURRENT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, str(user_fraction))
    if not os.path.exists(CURRENT_OUTPUT_PATH):
        os.makedirs(CURRENT_OUTPUT_PATH)

    current_arguments =  ['--user-network-user-fraction', str(user_fraction)]
    current_arguments += ['--constants-output-path', CURRENT_OUTPUT_PATH]
    current_arguments += ['--observations-I-fraction-tested', str(tested)]

    stdout_file = open(os.path.join(CURRENT_OUTPUT_PATH, 'stdout'), 'w')
    stderr_file = open(os.path.join(CURRENT_OUTPUT_PATH, 'stderr'), 'w')

    child = subprocess.Popen(EXEC + CONST_ARGUMENTS + current_arguments,
                             stdout=stdout_file,
                             stderr=stderr_file)
    children.append(child)

n_children = len(children)
print_info("Processes started...")
print_info("Total:", n_children)

completed = []
while len(completed) != n_children:
    time.sleep(sleep_interval)
    for child in children:
        if child in completed or child.poll() is None:
            pass
        else:
            completed.append(child)

    print_info("Remaining processes count:", n_children - len(completed))

    if (sleep_interval < SLEEP_INTERVAL_MAX
            and time_elapsed() > sleep_interval**2):
        sleep_interval += 1

