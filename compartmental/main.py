#!/usr/bin/python --

import sys # for exit
import numpy as np
import scipy.integrate as scint
from matplotlib import pyplot
from compart import Compartmental
from time import perf_counter as timer

################################################################################
# utils section ################################################################
################################################################################
LEFT_PAD = 23
RIGHT_PAD = 15
istub = ' # '
infomsg = {
    'runbegin' : '{{:<{0}}}'.format(LEFT_PAD),
    'runend' : 'steps: {:>6d}' + ' '*5 + 'elapsed: {:>9.2f}',
    'outfloat' : '{{:>{0}}} = {{:{1}.10g}}'.format(LEFT_PAD, RIGHT_PAD),
    'outint' : '{{:>{0}}} = {{:{1}d}}'.format(LEFT_PAD, RIGHT_PAD),
    'outstr' : '{{:>{0}}} = {{!s:>{1}}}'.format(LEFT_PAD, RIGHT_PAD)
}

################################################################################
# constants section ############################################################
################################################################################
EPS = 1e-14
PRINTLINE = '-'*42
DEBUG = False

T = 80 # integration time
dt = 1e-3 # maximum step size

################################################################################
# IC section ###################################################################
################################################################################
compart = Compartmental()

z0 = np.array( [0.95, 0.05, 0.0] )

################################################################################
# main section #################################################################
################################################################################

# SEIR
print(infomsg['runbegin'].format( '(SEIR)' ), end='', flush=True)
start_seir = timer()
sol_seir = scint.solve_ivp(
    compart.SEIR,
    [0, T],
    z0,
    method = 'RK45',
    max_step = dt)
elapsed_seir = timer() - start_seir
print(infomsg['runend'].format( len(sol_seir.t), elapsed_seir ), flush=True)
R = 1 - (sol_seir.y[0] + sol_seir.y[1] + sol_seir.y[2])

################################################################################
# save section #################################################################
################################################################################
#np.save("t_seir.npy", sol_seir.t)
#np.save("seir.npy", sol_seir.y)
#sys.exit()

################################################################################
# plot section #################################################################
################################################################################
pyplot.close('all')

# SEIR
pyplot.plot(sol_seir.t, sol_seir.y[0], label = 'S')
pyplot.plot(sol_seir.t, sol_seir.y[1], label = 'E')
pyplot.plot(sol_seir.t, sol_seir.y[2], label = 'I')
pyplot.plot(sol_seir.t, R,             label = 'R')
pyplot.legend()

pyplot.show()


