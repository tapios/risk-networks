import os, sys; sys.path.append(os.path.join('..', '..'))

import numpy as np
import matplotlib.pyplot as plt

from epiforecast.performance_metrics import PredictedPositiveFraction, TrueNegativeRate, TruePositiveRate, PerformanceTracker

#Here we plot classifiers for several curves with no interventions at day 28 and 56
# Test capacity 5%,10%,25% per day
# 100% user base 
# 50% user base
# we plot a single 0% case for 100% user base
# we plot another 0% case for 100% user base and 75% sensor wearers

OUTDIR = 'output'
OUT_NAME = 'u100_and_u50rand'
#which days to plot
days = [28,56]


#file parameters
EXP_NAME = 'u100_s0_d1' 
#user_EXP_NAME = 'u50_s0_d1' 
user_EXP_NAME = 'u50rand_s0_d1' 

noda_flag=True
noda_EXP_NAMES=['u100_s0_d1_0','u100_s75_d1_0']
noda_labels = ['no data','sensors only']
#test and no isolation flag (tani)
tani_flag = True

if tani_flag:
    tani_EXP_NAME='test_and_no_isolate'

# user base percentage
population = 97942
user_base=50
EXP_LABELS=['5%','10%','25%']
EXP_PARAM_VALUES = [4897, 9794, 24485]
if user_base == 75:
    user_EXP_PARAM_VALUES = [3672, 7347, 18364]
elif user_base == 50:
    user_EXP_PARAM_VALUES = [2448, 4897, 12242]    

if tani_flag:
    TANI_VALUES = [4897, 9794, 24485]#, 97942]

intervals_per_day = 8

#Plot parameters (need 1e-3 when tani_flag=True)
plot_min=1e-3


#Classifier_parameters
CRITICAL_TPR = 0.6 # pick classifier to attanin this TPR

N_THRESHOLD = 100
logthreshold_min = -4 # if max tpr < 1 then increase this (in magnitude)
thresholds = np.logspace(logthreshold_min,0,num=N_THRESHOLD)

exp_run = [EXP_NAME + '_' + str(val) for val in EXP_PARAM_VALUES]
user_exp_run = [user_EXP_NAME + '_' + str(val) for val in user_EXP_PARAM_VALUES]

if tani_flag:
    tani_run = [tani_EXP_NAME + '_' + str(val) for val in TANI_VALUES]
    tani_output_dirs = [os.path.join('.',OUTDIR, exp) for exp in tani_run ]  

    
#Plot labels
labels = [run + ' per day' for run in EXP_LABELS] 


output_dirs = [os.path.join('.',OUTDIR, exp) for exp in exp_run ]  
user_output_dirs = [os.path.join('.',OUTDIR, exp) for exp in user_exp_run ]  
noda_output_dirs = [os.path.join('.',OUTDIR, exp) for exp in noda_EXP_NAMES]  

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
    
    #and for user-base rates
    user_predicted_positive_fractions = np.zeros([len(user_output_dirs),N_THRESHOLD])
    user_true_positive_rates = np.zeros([len(user_output_dirs),N_THRESHOLD])
    user_true_negative_rates = np.zeros([len(user_output_dirs),N_THRESHOLD])


    threshold_choice = np.zeros(len(output_dirs))
    # for the test-only cases
    if tani_flag:
        tani_ppf=np.zeros(len(tani_output_dirs))
        tani_tpr=np.zeros(len(tani_output_dirs))
        for i,output_dir in enumerate(tani_output_dirs):
            #n_tested_nodes 
            n_test_nodes = TANI_VALUES[i]
            if n_test_nodes > 0:
                #load isolated nodes (i.e the positively tested nodes)
                kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()
                stored_nodes_to_intervene = np.load(os.path.join(output_dir, 'positive_nodes.npy'),
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
                tani_tpr[i] = tp / prevalence # predicted positives / actual positives
                tani_ppf[i] = n_positive_nodes / population # predicted positives / population                       
        
        
        print(tani_ppf)
        print(tani_tpr)
    else:
        tani_ppf=None
        tani_tpr=None

    #for no da cases
    if noda_flag:
        #and for user-base rates
        noda_predicted_positive_fractions = np.zeros([len(noda_output_dirs),N_THRESHOLD])
        noda_true_positive_rates = np.zeros([len(noda_output_dirs),N_THRESHOLD])
        noda_true_negative_rates = np.zeros([len(noda_output_dirs),N_THRESHOLD])

        for i,output_dir in enumerate(noda_output_dirs):

            if os.path.isfile(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy')):
                classifier_rates = np.load(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'))
                noda_predicted_positive_fractions[i,:] = classifier_rates[0,:]
                noda_true_negative_rates[i,:] = classifier_rates[1,:]
                noda_true_positive_rates[i,:] = classifier_rates[2,:]
                print("loaded saved rates from: ", output_dir)
            
            else:
            
                master_eqns_mean_states = np.load(os.path.join(output_dir,master_eqns_fname))
                kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()    

                for j,threshold in enumerate(thresholds):
            
                    performance_tracker = PerformanceTracker(metrics=[PredictedPositiveFraction(),
                                                                      TrueNegativeRate(),
                                                                      TruePositiveRate()],
                                                             user_nodes = None,
                                                             statuses=['I'],
                                                             threshold = threshold,
                                                             method = 'sum')
                    performance_tracker.update(kinetic_eqns_statuses, master_eqns_mean_states)
                    #obtain the performance at all states for this threshold
                    noda_predicted_positive_fractions[i,j] = performance_tracker.performance_track[0,0]
                    noda_true_negative_rates[i,j] = performance_tracker.performance_track[0,1]
                    noda_true_positive_rates[i,j] = performance_tracker.performance_track[0,2]
                
                #save the data
                classifier_rates = np.zeros([3,thresholds.shape[0]])
                classifier_rates[0,:] = noda_predicted_positive_fractions[i,:]
                classifier_rates[1,:] = noda_true_negative_rates[i,:]
                classifier_rates[2,:] = noda_true_positive_rates[i,:]
                np.save(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'), classifier_rates) 
            
                print("extracted true rates from: ", output_dir)

        
    #for full user base
    for i,output_dir in enumerate(output_dirs):

        if os.path.isfile(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy')):
            classifier_rates = np.load(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'))
            predicted_positive_fractions[i,:] = classifier_rates[0,:]
            true_negative_rates[i,:] = classifier_rates[1,:]
            true_positive_rates[i,:] = classifier_rates[2,:]
            print("loaded saved rates from: ", output_dir)
            
        else:
            
            master_eqns_mean_states = np.load(os.path.join(output_dir,master_eqns_fname))
            kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()    

            for j,threshold in enumerate(thresholds):
            
                performance_tracker = PerformanceTracker(metrics=[PredictedPositiveFraction(),
                                                                  TrueNegativeRate(),
                                                                  TruePositiveRate()],
                                                         user_nodes = None,
                                                         statuses=['I'],
                                                         threshold = threshold,
                                                         method = 'sum')
                performance_tracker.update(kinetic_eqns_statuses, master_eqns_mean_states)
                #obtain the performance at all states for this threshold
                predicted_positive_fractions[i,j] = performance_tracker.performance_track[0,0]
                true_negative_rates[i,j] = performance_tracker.performance_track[0,1]
                true_positive_rates[i,j] = performance_tracker.performance_track[0,2]
                
            #save the data
            classifier_rates = np.zeros([3,thresholds.shape[0]])
            classifier_rates[0,:] = predicted_positive_fractions[i,:]
            classifier_rates[1,:] = true_negative_rates[i,:]
            classifier_rates[2,:] = true_positive_rates[i,:]
            np.save(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'), classifier_rates) 
            
            print("extracted true rates from: ", output_dir)
     
    #for user base
    for i,output_dir in enumerate(user_output_dirs):

        if os.path.isfile(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy')):
            classifier_rates = np.load(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'))
            user_predicted_positive_fractions[i,:] = classifier_rates[0,:]
            user_true_negative_rates[i,:] = classifier_rates[1,:]
            user_true_positive_rates[i,:] = classifier_rates[2,:]
            print("loaded saved rates from: ", output_dir)
            
        else:
            
            master_eqns_mean_states = np.load(os.path.join(output_dir,master_eqns_fname))
            kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()    
            user_nodes = np.load(os.path.join(output_dir,"user_nodes.npy"))
            
            for j,threshold in enumerate(thresholds):
            
                performance_tracker = PerformanceTracker(metrics=[PredictedPositiveFraction(),
                                                                  TrueNegativeRate(),
                                                                  TruePositiveRate()],
                                                         user_nodes = user_nodes,
                                                         statuses=['I'],
                                                         threshold = threshold,
                                                         method = 'sum')
                performance_tracker.update(kinetic_eqns_statuses, master_eqns_mean_states)
                #obtain the performance at all states for this threshold
                user_predicted_positive_fractions[i,j] = performance_tracker.performance_track[0,0]
                user_true_negative_rates[i,j] = performance_tracker.performance_track[0,1]
                user_true_positive_rates[i,j] = performance_tracker.performance_track[0,2]
                
            #save the data
            classifier_rates = np.zeros([3,thresholds.shape[0]])
            classifier_rates[0,:] = user_predicted_positive_fractions[i,:]
            classifier_rates[1,:] = user_true_negative_rates[i,:]
            classifier_rates[2,:] = user_true_positive_rates[i,:]
            np.save(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'), classifier_rates) 
            
            print("extracted true rates from: ", output_dir)


    #plot the TPR curves:
    #shorthands
    ppf = predicted_positive_fractions
    tpr = true_positive_rates
    user_ppf = user_predicted_positive_fractions
    user_tpr = user_true_positive_rates
        
    #take out the prior
    if noda_flag:
        noda_ppf = noda_predicted_positive_fractions
        noda_tpr = noda_true_positive_rates
        
    #get colors from a color map
    colors = [plt.cm.OrRd(x) for x in np.linspace(0.3,0.9,tpr.shape[0])]

    fig, ax = plt.subplots(figsize=(10,5))    
    # plot full user base curves [plot_min,1]
    for xrate,yrate,clr,lbl in list(zip(ppf,tpr,colors,labels))[::-1]:
        #first sort the lower bound with interpolation 
        # xrate,yrate are monotone DECREASING)
        idxabovemin = np.max(np.where(xrate>=plot_min))
        xabovemin = xrate[idxabovemin]
        xbelowmin = xrate[idxabovemin+1]
        yabovemin = yrate[idxabovemin]
        ybelowmin = yrate[idxabovemin+1]
        yatmin = ybelowmin + (plot_min - xbelowmin) / (xabovemin - xbelowmin) * (yabovemin - ybelowmin)

        xplot = np.hstack((xrate[xrate>=plot_min], plot_min))
        yplot = np.hstack((yrate[xrate>=plot_min], yatmin))
        # plt.plot(xrate, yrate, color=clr, label=lbl, marker='|')
        plt.plot(xplot, yplot, color=clr, label=lbl)

    #plot user base curves 
    for xrate,yrate,clr,lbl in list(zip(user_ppf,user_tpr,colors,labels))[::-1]:
        #first sort the lower bound with interpolation 
        # xrate,yrate are monotone DECREASING)
        idxabovemin = np.max(np.where(xrate>=plot_min))
        xabovemin = xrate[idxabovemin]
        xbelowmin = xrate[idxabovemin+1]
        yabovemin = yrate[idxabovemin]
        ybelowmin = yrate[idxabovemin+1]
        yatmin = ybelowmin + (plot_min - xbelowmin) / (xabovemin - xbelowmin) * (yabovemin - ybelowmin)

        xplot = np.hstack((xrate[xrate>=plot_min], plot_min))
        yplot = np.hstack((yrate[xrate>=plot_min], yatmin))
        # plt.plot(xrate, yrate, color=clr, label=lbl, marker='|')
        plt.plot(xplot, yplot, color=clr, label=lbl, linestyle="--")
            
    #plot no da cases
    if noda_flag:
        noda_linestyles = [":","-."]
        noda_colors = ["black", plt.cm.OrRd(0.15)]

        for xrate,yrate,lbl,clr,ls in list(zip(noda_ppf,noda_tpr,noda_labels,noda_colors,noda_linestyles))[::-1]:
            idxabovemin = np.max(np.where(xrate>=plot_min))
            xabovemin = xrate[idxabovemin]
            xbelowmin = xrate[idxabovemin+1]
            yabovemin = yrate[idxabovemin]
            ybelowmin = yrate[idxabovemin+1]
            yatmin = ybelowmin + (plot_min - xbelowmin) / (xabovemin - xbelowmin) * (yabovemin - ybelowmin)
            
            xplot = np.hstack((xrate[xrate>=plot_min], plot_min))
            yplot = np.hstack((yrate[xrate>=plot_min], yatmin))
            plt.plot(xplot, yplot, color=clr, label=lbl, linestyle=ls)
        

    #plot test_and_isolate curves
    if tani_flag:
        for (xplot,yplot,clr) in zip(tani_ppf, tani_tpr, colors[-len(tani_tpr):]):
            plt.scatter([xplot],[yplot], color=[clr], marker='o')

    #plot random case
    #plt.plot([1e-3, 1], [1e-3, 1], color='darkblue', linestyle='--')
    plt.plot(np.logspace(np.log10(plot_min),0,num=100),np.logspace(np.log10(plot_min),0,num=100),color='black', linestyle='--')
    ax.set_xscale('log')
    plt.xlim([plot_min,1.])
    plt.xlabel('PPF')#Predicted Positive Fraction')
    plt.ylabel('TPR') #True Positive Rate')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.title('PPF vs TPR Curve')
    plt.legend(loc='upper left')# 'lower right'
    fig_name = os.path.join(OUTDIR,'tpr_'+OUT_NAME+'_'+str(day)+'.pdf')
    fig.savefig(fig_name,dpi=300)

    print("plotted TPR-PPF curves in figure: ", fig_name)
