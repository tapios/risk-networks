import os, sys; sys.path.append(os.path.join('..', '..'))

import numpy as np

from epiforecast.performance_metrics import TrueNegativeRate, TruePositiveRate, PerformanceTracker
from epiforecast.epiplots import plot_roc_curve

EXP_NAME = 'short_roc_itest'
EXP_PARAM_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1.0]
THRESHOLD_NUM = 50
OUTDIR = 'output'

# data file for mean states of master equations
# np.array of size [user_population,time_span]
master_eqns_fname = 'master_eqns_mean_states.npy'

# data file for statuses of kinetic equations
# list (size time_span) of dicts (of size user_population)
kinetic_eqns_fname = 'kinetic_eqns_statuses.npy'


thresholds = 1.0/THRESHOLD_NUM*np.arange(THRESHOLD_NUM)
exp_run = [EXP_NAME + '_' + str(val) for val in EXP_PARAM_VALUES]
output_dirs = [os.path.join('.',OUTDIR, exp) for exp in exp_run]  
    
#container for true rates as a [num expts  x num thresholds] 
true_positive_rates = np.zeros([len(output_dirs),THRESHOLD_NUM])
true_negative_rates = np.zeros([len(output_dirs),THRESHOLD_NUM])

for i,output_dir in enumerate(output_dirs):
    
    master_eqns_mean_states = np.load(os.path.join(output_dir,master_eqns_fname))
    kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()

    
    for j,threshold in enumerate(thresholds):
        performance_tracker = PerformanceTracker(metrics=[TrueNegativeRate(),TruePositiveRate()],
                                                 threshold = threshold,
                                                 method = 'or')
        #obtain performance at end time
        performance_tracker.update(kinetic_eqns_statuses[-1],
                                   master_eqns_mean_states[:,-1])

        #obtain the performance at all states for this threshold
        true_negative_rates[i,j] = performance_tracker.performance_track[0,0]
        true_positive_rates[i,j] = performance_tracker.performance_track[0,1]

    print("extracted true rates from: ", output_dir)
#plot all the ROCs on one plot
labels = [str(100*param) + '% tested per day' for param in EXP_PARAM_VALUES] 
fig,ax = plot_roc_curve(true_negative_rates,true_positive_rates,labels=labels,show=False)
fig.savefig(os.path.join(OUTDIR,'or_'+EXP_NAME+'.png'),dpi=300)

print("plotted ROC curves in figure: ", os.path.join(OUTDIR,'roc'+EXP_NAME+'.png'))
