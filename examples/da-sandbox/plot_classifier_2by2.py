import os, sys; sys.path.append(os.path.join('..', '..'))

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from epiforecast.performance_metrics import PredictedPositiveFraction, TrueNegativeRate, TruePositiveRate, PerformanceTracker

#Here we plot classifiers for several curves with no interventions at day 35 and 56
#column 1 = day 35, column 2 = day 56
#row 1 = 100% curve, with contact trace, with test-only, with 75% sensors only
#row 2 = 25%, 50%, 75% user base cases

# Test capacity 5%,10%,25% per day


###########################
# Some general parameters #
###########################

#'rand' or 'nbhd'
user_base_type = 'rand'


OUTDIR = 'output'
if user_base_type == 'nbhd':
    OUT_NAME = '2by2_classifiers'
elif user_base_type == 'rand':
    OUT_NAME = '2by2_classifiers_rand'

#which days to plot
days = [35,56] 

intervals_per_day = 8
#Plot parameters (need 1e-3 when tani_flag=True)
plot_min=1e-3

#Classifier_parameters
CRITICAL_TPR = 0.6 # pick classifier to attanin this TPR

N_THRESHOLD = 100
logthreshold_min = -4 # if max tpr < 1 then increase this (in magnitude)
thresholds = np.logspace(logthreshold_min,0,num=N_THRESHOLD)

#################################
# Set up for the non-user cases #
#################################
#u100 - 100% user base
# s75 - 75% sensor-only case
# tani - test and no isolate
# tact - test and contact trace

u100_EXP_NAME = 'u100_s0_d1'
tani_EXP_NAME = 'test_and_no_isolate'
tact_EXP_NAME = 'contact_trace'
s75_EXP_NAME  = 'u100_s75_d1_0'

u100_EXP_LABELS = ['5%','10%','25%']
s75_EXP_LABEL = 'Sensors only'

u100_EXP_PARAM_VALUES = [4897, 9794, 24485]

u100_exp_run = [u100_EXP_NAME + '_' + str(val) for val in u100_EXP_PARAM_VALUES]
tani_exp_run = [tani_EXP_NAME + '_' + str(val) for val in u100_EXP_PARAM_VALUES]
tact_exp_run = [tact_EXP_NAME + '_' + str(val) for val in u100_EXP_PARAM_VALUES]
s75_exp_run  = [s75_EXP_NAME]

u100_output_dirs = [os.path.join('.',OUTDIR, exp) for exp in u100_exp_run ]  
s75_output_dirs  = [os.path.join('.',OUTDIR, exp) for exp in s75_exp_run  ]  
tani_output_dirs = [os.path.join('.',OUTDIR, exp) for exp in tani_exp_run ]  
tact_output_dirs = [os.path.join('.',OUTDIR, exp) for exp in tact_exp_run ]  

##################################
# Set up for the user base cases #
##################################

#file parameters
if user_base_type == 'nbhd':
    user_EXP_NAMES = [
        'u25_s0_d1', 
        'u50_s0_d1',
        'u75_s0_d1']
elif user_base_type == 'rand':
    user_EXP_NAMES = [
        'u25rand_s0_d1', 
        'u50rand_s0_d1',
        'u75rand_s0_d1']


# user base percentage
population = 97942
user_base=[25,50,75]

user_EXP_LABELS=['25%','50%','75%']
user_EXP_PARAM_VALUES = np.zeros((len(user_base),len(user_EXP_LABELS)), dtype=int)
for idx,n_user in enumerate(user_base):
    if n_user== 75:
        user_EXP_PARAM_VALUES[idx,:] = np.array([3672, 7347, 18364])
    elif n_user  == 50:
        user_EXP_PARAM_VALUES[idx,:] = np.array([2448, 4897, 12242])
    elif n_user == 25:
        user_EXP_PARAM_VALUES[idx,:] = np.array([1224, 2448, 6121])

user_exp_run = np.empty(user_EXP_PARAM_VALUES.shape, dtype='<U100')
for i in range(len(user_base)):
    user_exp_run[i,:] = [user_EXP_NAMES[i] + '_' + str(val) for val in user_EXP_PARAM_VALUES[i,:]]


#Plot labels
labels = [run + '/day' for run in u100_EXP_LABELS] 

user_output_dirs = np.empty(user_exp_run.shape, dtype='<U100')
for i in range(len(user_base)):
    user_output_dirs[i,:] = [os.path.join('.',OUTDIR, exp) for exp in user_exp_run[i,:] ]

#set up figure:
#2x2 plot:

# Set figure sizes - from Lucas script
fig_width_pt = 368*1.5    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width*ratio  # height in inches
fig_size = [1.5*fig_width, 1.5*fig_height]

params = {  # 'backend': 'ps',
    #'font.family': 'sans-serif',
    #'font.sans-serif': 'Helvetica',
    'font.size': 11,
    'axes.labelsize': 'large',
    'axes.titlesize': 'large',
    'legend.fontsize': 'large',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'savefig.dpi': 150,
    #'text.usetex': True,
    'figure.figsize': fig_size}
rcParams.update(params)


fig, axs = plt.subplots(nrows = 2, ncols = 2)

#######################################
# Create/load all the classifier data #
#######################################

for day_idx,day in enumerate(days):

    print("processing day ", day)
    interval_of_recording = day*intervals_per_day

    master_eqns_fname = 'master_eqns_mean_states_at_step_'+str(interval_of_recording)+'.npy'
    # data file for statuses of kinetic equations
    # dict (of size user_population)
    kinetic_eqns_fname = 'kinetic_eqns_statuses_at_step_'+str(interval_of_recording)+'.npy'
    
    
    threshold_choice = np.zeros(len(u100_output_dirs))

    #[1.]
    ### for the test-only ###
    tani_ppf=np.zeros(len(tani_output_dirs))
    tani_tpr=np.zeros(len(tani_output_dirs))
    for i,output_dir in enumerate(tani_output_dirs):
        #n_tested_nodes 
        n_test_nodes = u100_EXP_PARAM_VALUES[i]
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
        
    print("PPF and TPR for 'Test only'")
    print(tani_ppf)
    print(tani_tpr)

    #[2.]
    ### for the contact trace ###
    tact_ppf=np.zeros(len(tact_output_dirs))
    tact_tpr=np.zeros(len(tact_output_dirs))
    for i,output_dir in enumerate(tact_output_dirs):
        #n_tested_nodes 
        n_test_nodes = u100_EXP_PARAM_VALUES[i]
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
            tact_tpr[i] = tp / prevalence # predicted positives / actual positives
            tact_ppf[i] = n_positive_nodes / population # predicted positives / population                       
                
    print("PPF and TPR for 'contact_trace'")
    print(tact_ppf)
    print(tact_tpr)
    
    #[3.]
    ### The sensors case ###
    s75_ppf = np.zeros([len(s75_output_dirs),N_THRESHOLD])
    s75_tpr = np.zeros([len(s75_output_dirs),N_THRESHOLD])
    s75_tnr = np.zeros([len(s75_output_dirs),N_THRESHOLD])

    for i,output_dir in enumerate(s75_output_dirs):
        
        if os.path.isfile(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy')):
            classifier_rates = np.load(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'))
            s75_ppf[i,:] = classifier_rates[0,:]
            s75_tnr[i,:] = classifier_rates[1,:]
            s75_tpr[i,:] = classifier_rates[2,:]
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
                s75_ppf[i,j] = performance_tracker.performance_track[0,0]
                s75_tnr[i,j] = performance_tracker.performance_track[0,1]
                s75_tpr[i,j] = performance_tracker.performance_track[0,2]
                
            #save the data
            classifier_rates = np.zeros([3,thresholds.shape[0]])
            classifier_rates[0,:] = s75_ppf[i,:]
            classifier_rates[1,:] = s75_tnr[i,:]
            classifier_rates[2,:] = s75_tpr[i,:]
            np.save(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'), classifier_rates) 
            
            print("extracted true rates from: ", output_dir)

    #[4.]
    #The 100% user_base cases 
    #containers for true rates as a [num expts  x num thresholds] 
    ppf = np.zeros([len(u100_output_dirs),N_THRESHOLD])
    tpr = np.zeros([len(u100_output_dirs),N_THRESHOLD])
    tnr = np.zeros([len(u100_output_dirs),N_THRESHOLD])

    for i,output_dir in enumerate(u100_output_dirs):
        if os.path.isfile(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy')):
            classifier_rates = np.load(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'))
            ppf[i,:] = classifier_rates[0,:]
            tnr[i,:] = classifier_rates[1,:]
            tpr[i,:] = classifier_rates[2,:]
            print("loaded saved rates from: ", output_dir)
            for thr,ppf_i,tpr_i in zip(thresholds,ppf[i,:],tpr[i,:]):
                print(thr,ppf_i,tpr_i)

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
                ppf[i,j] = performance_tracker.performance_track[0,0]
                tnr[i,j] = performance_tracker.performance_track[0,1]
                tpr[i,j] = performance_tracker.performance_track[0,2]
                
            #save the data
            classifier_rates = np.zeros([3,thresholds.shape[0]])
            classifier_rates[0,:] = ppf[i,:]
            classifier_rates[1,:] = tnr[i,:]
            classifier_rates[2,:] = tpr[i,:]
            np.save(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'), classifier_rates) 
            
            print("extracted true rates from: ", output_dir)
     
    #[5.]
    ### user base cases

    user_ppf = np.zeros([user_output_dirs.shape[0],user_output_dirs.shape[1],N_THRESHOLD])
    user_tpr = np.zeros([user_output_dirs.shape[0],user_output_dirs.shape[1],N_THRESHOLD])
    user_tnr = np.zeros([user_output_dirs.shape[0],user_output_dirs.shape[1],N_THRESHOLD])
 
    for k in range(user_output_dirs.shape[0]):
        for i,output_dir in enumerate(user_output_dirs[k,:]):
            print(output_dir)
            if os.path.isfile(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy')):
                classifier_rates = np.load(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'))
                user_ppf[k,i,:] = classifier_rates[0,:]
                user_tnr[k,i,:] = classifier_rates[1,:]
                user_tpr[k,i,:] = classifier_rates[2,:]
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
                    user_ppf[k,i,j] = performance_tracker.performance_track[0,0]
                    user_tnr[k,i,j] = performance_tracker.performance_track[0,1]
                    user_tpr[k,i,j] = performance_tracker.performance_track[0,2]
                
                #save the data
                classifier_rates = np.zeros([3,thresholds.shape[0]])
                classifier_rates[0,:] = user_ppf[k,i,:]
                classifier_rates[1,:] = user_tnr[k,i,:]
                classifier_rates[2,:] = user_tpr[k,i,:]
                np.save(os.path.join(output_dir,'classifier_arrays_'+str(day)+'.npy'), classifier_rates) 
            
                print("extracted true rates from: ", output_dir)

    
    ############################
    # Plotting the information #
    ############################
    # everything is now stored in 
    # X_tpr
    # X_ppf

    #get colors from a color map
    colors = [plt.cm.YlGn(x) for x in np.linspace(0.4,0.9,tpr.shape[0])]

    #the day gives the column:
    ax0 = axs[0][day_idx]
    ax1 = axs[1][day_idx]

    #where to put subplot a,b,c,d
    text_x = 0.45*plot_min
    text_y = 0.95
    fontweight = "bold"
    if day_idx == 0:
        ax0.set_title("April 9")
        ax0.text(text_x,text_y,'a', fontweight=fontweight, fontsize='large')
        ax0.set_ylabel("TPR")
        ax1.text(text_x,text_y,'c', fontweight=fontweight, fontsize='large')
        ax1.set_ylabel("TPR")
        
    if day_idx == 1:
        ax0.text(text_x,text_y,'b', fontweight=fontweight, fontsize='large')
        ax0.set_title("April 30")
        ax1.text(text_x,text_y,'d', fontweight=fontweight, fontsize='large')
    

    ##################
    # The 100% plot: #
    ##################

    # u100, s75, tani, tact on one plot:

    # plot u100 curves 
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
        ax0.plot(xplot, yplot, color=clr, label=lbl)
        #ax1.plot(xplot, yplot, color="black")

    #plot sensor-only case   
    #only 1 case
    s75_ppfvec = s75_ppf[0,:]
    s75_tprvec = s75_tpr[0,:]
    idxabovemin = np.max(np.where(s75_ppfvec>=plot_min))
    xabovemin = s75_ppfvec[idxabovemin]
    xbelowmin = s75_ppfvec[idxabovemin+1]
    yabovemin = s75_tprvec[idxabovemin]
    ybelowmin = s75_tprvec[idxabovemin+1]
    yatmin = ybelowmin + (plot_min - xbelowmin) / (xabovemin - xbelowmin) * (yabovemin - ybelowmin)
    
    xplot = np.hstack((s75_ppfvec[s75_ppfvec>=plot_min], plot_min))
    yplot = np.hstack((s75_tprvec[s75_ppfvec>=plot_min], yatmin))
    ax0.plot(xplot, yplot, color=plt.cm.YlGn(0.25), label=s75_EXP_LABEL, linestyle="-")

    # plot test_and_no_isolate points
    for (xplot,yplot,clr) in zip(tani_ppf, tani_tpr, colors[-len(tani_tpr):]):
        ax0.scatter([xplot],[yplot], color=[clr], marker='o')

    # plot test and contact trace points
    for (xplot,yplot,clr) in zip(tact_ppf, tact_tpr, colors[-len(tact_tpr):]):
        ax0.scatter([xplot],[yplot], color=[clr], marker='o', facecolors='none')
 
    #plot random classifier
    ax0.plot(np.logspace(np.log10(plot_min),0,num=100),np.logspace(np.log10(plot_min),0,num=100),color='black', linestyle='--')

    ax0.set_xscale('log')
    ax0.set_xlim([plot_min,1.])
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)

    if day_idx == 0:
        ax0.legend(loc='upper left',title='Tests', frameon=False)# 'lower right'
        leg = ax0.get_legend()
        leg._legend_box.align = "left"
    ##################
    # The user plots: #
    ##################

    
    #plot user base curves 
    styles = [':','-.','--']
    for k in range(len(user_base)):
        lbls = [lbl if i==k else '' for i,lbl in enumerate(user_EXP_LABELS)]

        # a trick to get the entry in the legend
        #if k == 0:
        #    ax1.hlines(0.5,xmin=plot_min,xmax=1.0,label='100%', alpha=0.0)
        
        for xrate,yrate,clr,lbl in list(zip(user_ppf[k,:,:],user_tpr[k,:,:],colors,lbls))[::-1]:
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
            ax1.plot(xplot, yplot, color=clr, label=lbl, linestyle=styles[k])
 
    #plot random classifier
    ax1.plot(np.logspace(np.log10(plot_min),0,num=100),np.logspace(np.log10(plot_min),0,num=100),color='black', linestyle='--')

    ax1.set_xscale('log')
    ax1.set_xlim([plot_min,1.])
    ax1.set_xlabel('PPF')#Predicted Positive Fraction')
    #ax1.set_ylabel('TPR') #True Positive Rate')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    if day_idx == 0:
        ax1.legend(loc='upper left', title='User base', frameon=False)# 'lower right'
        #set these to be black
        leg = ax1.get_legend()
        [lgd.set_color('grey') for lgd in leg.legendHandles]
        leg._legend_box.align = "left"

#when all days are plotted   
fig.tight_layout(pad=2.0)
fig_name = os.path.join(OUTDIR,'tpr_'+OUT_NAME+'.pdf')
fig.savefig(fig_name,dpi=300)

print("plotted TPR-PPF curves in figure: ", fig_name)
