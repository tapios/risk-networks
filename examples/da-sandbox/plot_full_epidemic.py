import os, sys; sys.path.append(os.path.join('..', '..'))

import numpy as np

from epiforecast.epiplots import plot_epidemic_data
from matplotlib import pyplot as plt


EXP_NAME = 'u75rand_s0_d1_i0.03_exog' 
OUTDIR = 'output'

# user base percentage
user_base=75
population = 97942
if user_base == 100:    
    EXP_PARAM_VALUES = [0, 4897, 9794, 24485, 97942]

elif user_base == 75:
    EXP_PARAM_VALUES = [0, 3672, 7347, 18364, 73456]
    EXP_PARAM_VALUES = [0, 3672, 7347, 18364, 73353]

elif user_base == 25:
    EXP_PARAM_VALUES = [0, 1224, 2448, 6121, 24485]

elif user_base == 5:
    EXP_PARAM_VALUES = [0, 245, 489, 1224, 4897]

exp_run = [EXP_NAME + '_' + str(val) for val in EXP_PARAM_VALUES]
output_dirs = [os.path.join('.',OUTDIR, exp) for exp in exp_run ]  


for i,output_dir in enumerate(output_dirs):
    
    statuses_sum_trace = np.load(os.path.join(output_dir, 'trace_full_kinetic_statuses_sum.npy'))
    time_span = np.load(os.path.join(output_dir, 'time_span.npy'))

    fig, axes = plt.subplots(1, 3, figsize = (16, 4))
    axes = plot_epidemic_data(population, statuses_sum_trace, axes, time_span)
    plt.savefig(os.path.join(output_dir, 'full_epidemic.png'), rasterized=True, dpi=150)
    plt.close()
