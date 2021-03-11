import os, sys; sys.path.append(os.path.join('..', '..'))

import numpy as np

from epiforecast.performance_metrics import PredictedPositiveFraction, TrueNegativeRate, TruePositiveRate, PerformanceTracker
from epiforecast.epiplots import plot_roc_curve, plot_tpr_curve

#file parameters

EXP_NAME = 'u75_s0_d1_i0.03_exog' #1e5_nohdmass_WRIIO_1.0_5e-2_3.0_0.1_1e-12'
OUTDIR = 'output'

#include prior
noda_flag=False
if noda_flag:
    noda_EXP_NAME='noda_1e5_u25_parsd0.25_nosd_0'


#include test_and_isolate values
test_and_isolate_flag = False

if test_and_isolate_flag:
    TAI_NAME='test_and_isolate'

# user base percentage
user_base=75
population = 97942
if user_base == 100:    
    EXP_PARAM_VALUES = [0, 4897, 9794, 24485, 97942]
    # EXP_PARAM_VALUES = [0, 49, 98, 245,982]
elif user_base == 75:
    EXP_PARAM_VALUES = [0, 3672, 7347, 18364, 73456]
    #EXP_PARAM_VALUES = [0, 3672, 7347, 18364, 73353]

elif user_base == 25:
    EXP_PARAM_VALUES = [0, 1224, 2448, 6121, 24485]

elif user_base == 5:
    EXP_PARAM_VALUES = [0, 245, 489, 1224, 4897]
    
if noda_flag:
    EXP_PARAM_VALUES.insert(0,'noda')

if test_and_isolate_flag:
    TAI_VALUES = [4897, 9794, 24485, 97942]

# data file for mean states of master equations
# np.array of size [user_population,time_span]

days = [12,16,20,24,28]
intervals_per_day = 8

#Plot parameters
plot_min=1e-3


#Classifier_parameters
CLASS_METHOD='sum' #'or' or 'sum'
CLASS_STATUSES=['I']  
CRITICAL_TPR = 0.6 # pick classifier to attain this TPR
PLOT_NAME=CLASS_METHOD

N_THRESHOLD = 100
logthreshold_min = -4 # if max tpr < 1 then increase this (in magnitude)
thresholds = np.logspace(logthreshold_min,0,num=N_THRESHOLD)

exp_run = [EXP_NAME + '_' + str(val) for val in EXP_PARAM_VALUES]

if test_and_isolate_flag:
    tai_run = [TAI_NAME + '_' + str(val) for val in TAI_VALUES]
    tai_output_dirs = [os.path.join('.',OUTDIR, exp) for exp in tai_run ]  
    
#add the noda case
if noda_flag:
    exp_run[0] = noda_EXP_NAME #overwrite

#Plot labels
if noda_flag:
    labels = [str(run) + ' virus tests per day' if i>=1 else 'no assimilation' for (i,run) in enumerate(EXP_PARAM_VALUES)] 
else:
    labels = [str(run) + ' virus tests per day' for run in EXP_PARAM_VALUES] 



output_dirs = [os.path.join('.',OUTDIR, exp) for exp in exp_run ]  


for day in days:
    
    print("processing day ", day)
    interval_of_recording = day*intervals_per_day

    master_eqns_fname = 'master_eqns_mean_states_at_step_'+str(interval_of_recording)+'.npy'
    # data file for statuses of kinetic equations
    # dict (of size user_population)
    kinetic_eqns_fname = 'kinetic_eqns_statuses_at_step_'+str(interval_of_recording)+'.npy'
    
    #container for true rates as a [num expts  x num thresholds] 
    predicted_positive_fractions = np.zeros([len(output_dirs),N_THRESHOLD])
    true_positive_rates = np.zeros([len(output_dirs),N_THRESHOLD])
    true_negative_rates = np.zeros([len(output_dirs),N_THRESHOLD])
    threshold_choice = np.zeros(len(output_dirs))

    if test_and_isolate_flag:
        tai_ppf=np.zeros(len(tai_output_dirs))
        tai_tpr=np.zeros(len(tai_output_dirs))
        for i,output_dir in enumerate(tai_output_dirs):
            #n_tested_nodes 
            n_test_nodes = TAI_VALUES[i]
            if n_test_nodes > 0:
                #load isolated nodes (i.e the positively tested nodes)
                kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()
                stored_nodes_to_intervene = np.load(os.path.join(output_dir, 'isolated_nodes.npy'),
                                                    allow_pickle=True)
                stored_nodes_to_intervene = stored_nodes_to_intervene.item()
            
                #positively tested nodes:
                positive_test_nodes = stored_nodes_to_intervene[float(day)]
                n_positive_nodes = positive_test_nodes.shape[0]
                
                #actual positive nodes:
                true_positive_nodes = [k for (k,status) in kinetic_eqns_statuses.items() if status == 'I' ]
                prevalence = len(true_positive_nodes)
                #rates
                tp = sum([1 for node in positive_test_nodes if node in true_positive_nodes])
                tai_tpr[i] = tp / prevalence # predicted positives / actual positives
                tai_ppf[i] = n_positive_nodes / population # predicted positives / population           
        print(tai_ppf)
        print(tai_tpr)
    else:
        tai_ppf=None
        tai_tpr=None


    for i,output_dir in enumerate(output_dirs):
    
        master_eqns_mean_states = np.load(os.path.join(output_dir,master_eqns_fname))
        kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()
    
        for j,threshold in enumerate(thresholds):
            performance_tracker = PerformanceTracker(metrics=[PredictedPositiveFraction(),
                                                              TrueNegativeRate(),
                                                              TruePositiveRate()],
                                                     statuses=CLASS_STATUSES,
                                                     threshold = threshold,
                                                     method = CLASS_METHOD)
            

            performance_tracker.update(kinetic_eqns_statuses, master_eqns_mean_states)

            #obtain the performance at all states for this threshold
            predicted_positive_fractions[i,j] = performance_tracker.performance_track[0,0]
            true_negative_rates[i,j] = performance_tracker.performance_track[0,1]
            true_positive_rates[i,j] = performance_tracker.performance_track[0,2]
            
            #print("Threshold", threshold, "PPF", predicted_positive_fractions[i,j],"TPR", true_positive_rates[i,j])
        
        # threshold that is "furthest" from the x-y curve 
        # coords are fpr and tpr
        false_positive_rates = 1 - true_negative_rates
        dist_to_xy = (true_positive_rates[i,:] - false_positive_rates[i,:])/2
        threshold_choice[i] = np.max([val for (k,val) in enumerate(thresholds) if dist_to_xy[k] == np.max(dist_to_xy)])

        

        #max threshold such that we get above a certain TPR
        #threshold_choice[i] = np.max([idx_val_pair[1] for idx_val_pair in enumerate(thresholds) if true_positive_rates[i, idx_val_pair[0]] > CRITICAL_TPR])    
        print("extracted true rates from: ", output_dir, "threshold choice", threshold_choice[i])


    #plot all the ROCs on one plot
    
    fig,ax = plot_roc_curve(true_negative_rates,
                            true_positive_rates,
                            labels=labels,
                            show=False)
    fig_name = os.path.join(OUTDIR,'roc_'+PLOT_NAME+'_'+EXP_NAME+'_'+str(day)+'.png')
    fig.savefig(fig_name,dpi=300)
    
    print("plotted TPR-FPR (ROC) curves in figure: ", fig_name)
    #plot all the ROCs on one plot
    fig,ax = plot_tpr_curve(predicted_positive_fractions,
                            true_positive_rates,
                            test_and_isolate_flag=test_and_isolate_flag,
                            tai_predicted_positive_fraction=tai_ppf,
                            tai_true_positive_rates=tai_tpr,
                            xmin=plot_min,
                            labels=labels,
                            show=False)
    fig_name = os.path.join(OUTDIR,'tpr_'+PLOT_NAME+'_'+EXP_NAME+'_'+str(day)+'.png')
    fig.savefig(fig_name,dpi=300)

    print("plotted TPR-PPF curves in figure: ", fig_name)
