import os, sys; sys.path.append(os.path.join('..', '..'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

import datetime as dt
import numpy as np

# Settings
CASE_NAME = 'ensemble_closure'
end_time = 90 
intervention_I_min_threshold = 1.
plot_threshold = True 

num_bins = 100
x_min = 0.95
x_max = 1.02
y_min = 1e-1
y_max = 1e6 

title_name = '<SI>'
varname = 'CM_SI'

OUTPUT_PATH = os.path.join('output', CASE_NAME)

FIGS_FOLDER = os.path.join(OUTPUT_PATH, varname + '_coeff')
if not os.path.exists(FIGS_FOLDER):
    os.makedirs(FIGS_FOLDER)

coeffs = np.load(os.path.join(OUTPUT_PATH, \
        varname + '_coeff.npy'), allow_pickle=True)

base = dt.datetime(2020, 3, 5)
title_list = [base + dt.timedelta(days=i) for i in range(coeffs.shape[0])]

for i in range(int(end_time)):
    plt.figure(i)
    current_step = int(i)

    coeff = coeffs[i] 
    
    matplotlib.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})

    plt.hist(coeff, bins=np.arange(x_min, x_max+0.0025, 0.0025), density=False,
             alpha=0.6, color='b', edgecolor='black', linewidth=1)
    if plot_threshold:
        plt.vlines(intervention_I_min_threshold, 
                y_min, y_max,
                color='r', linestyles='dashed', lw=2)

    plt.title(str(title_list[i].date()))
    plt.xlabel("Coefficient value")
    plt.ylabel("Number of counts")
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_FOLDER, varname + '_coeff_day_'+str(i)+'.pdf'))
    plt.close()

#images = []
#for i in range(int(end_time)):
#    filename = os.path.join(FIGS_FOLDER, \
#            varname + 'coeff_day_'+str(i+1)+'.png')
#    images.append(imageio.imread(filename))
#imageio.mimsave(os.path.join(OUTPUT_PATH, varname + '_coeff.gif'), images, fps=4)

