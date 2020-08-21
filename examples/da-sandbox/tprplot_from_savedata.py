import os, sys; sys.path.append(os.path.join('..', '..'))

import numpy as np

from epiforecast.performance_metrics import PredictedPositiveFraction, TruePositiveRate, PerformanceTracker
from epiforecast.epiplots import plot_roc_curve, plot_tpr_curve

#file parameters
EXP_NAME = '1e4_test'
OUTDIR = 'output'
# data file for mean states of master equations
# np.array of size [user_population,time_span]
master_eqns_fname = 'master_eqns_mean_states.npy'
# data file for statuses of kinetic equations
# list (size time_span) of dicts (of size user_population)
kinetic_eqns_fname = 'kinetic_eqns_statuses.npy'

EXP_PARAM_VALUES = [0, 49, 98, 196, 294,  392, 491, 982]

#day to measure
time_day = 30
intervals_per_day = 8

#Classifier_parameters
N_THRESHOLD = 100
CLASS_METHOD='sum' #'or' or 'sum'
CLASS_STATUSES=['I']  
CRITICAL_TPR = 0.5 # pick classifier to attain this TPR
print("Threshold choice if TPR > ", CRITICAL_TPR)


thresholds = 1.0/(5*N_THRESHOLD-1)*np.arange(N_THRESHOLD-1)
thresholds = np.hstack([thresholds,1])

#thresholds = 1.0/(N_THRESHOLD)*np.arange(N_THRESHOLD)
print(thresholds)
exp_run = [EXP_NAME + '_' + str(val) for val in EXP_PARAM_VALUES]
output_dirs = [os.path.join('.',OUTDIR, exp) for exp in exp_run]  
    
#container for true rates as a [num expts  x num thresholds] 
true_positive_rates = np.zeros([len(output_dirs),N_THRESHOLD])
predicted_positive_fractions = np.zeros([len(output_dirs),N_THRESHOLD])
threshold_choice = np.zeros(len(output_dirs))
for i,output_dir in enumerate(output_dirs):
    
    master_eqns_mean_states = np.load(os.path.join(output_dir,master_eqns_fname))
    kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()

    
    for j,threshold in enumerate(thresholds):
        performance_tracker = PerformanceTracker(metrics=[PredictedPositiveFraction(),TruePositiveRate()],
                                                 statuses=CLASS_STATUSES,
                                                 threshold = threshold,
                                                 method = CLASS_METHOD)
        #obtain performance at end time
        performance_tracker.update(kinetic_eqns_statuses[ time_day * intervals_per_day],
                                   master_eqns_mean_states[:, time_day * intervals_per_day])

        #obtain the performance at all states for this threshold
        predicted_positive_fractions[i,j] = performance_tracker.performance_track[0,0]
        true_positive_rates[i,j] = performance_tracker.performance_track[0,1]
    
    #max threshold such that we get above a certain TPR
    threshold_choice[i] = np.max([idx_val_pair[1] for idx_val_pair in enumerate(thresholds) if true_positive_rates[i, idx_val_pair[0]] > CRITICAL_TPR])
    

    

    print("extracted true rates from: ", output_dir, "threshold choice", threshold_choice[i])


#plot all the ROCs on one plot
labels =[str(param) + ' tests per day' for param in EXP_PARAM_VALUES] 
fig,ax = plot_tpr_curve(predicted_positive_fractions,true_positive_rates,labels=labels,show=False)
fig_name = os.path.join(OUTDIR,'tprplot_'+CLASS_METHOD+'_'+EXP_NAME+'_'+str(time_day)+'.png')
fig.savefig(fig_name,dpi=300)

print("plotted ROC curves in figure: ", fig_name)
