import os, sys; sys.path.append(os.path.join('..', '..'))
import matplotlib.pyplot as plt
import numpy as np

#file parameters
EXP_NAME = 'new_assimilation_itest'
OUTDIR = 'output'
# data file for mean states of master equations
# np.array of size [user_population,time_span]
master_eqns_fname = 'master_eqns_mean_states.npy'
# data file for statuses of kinetic equations
# list (size time_span) of dicts (of size user_population)
kinetic_eqns_fname = 'kinetic_eqns_statuses.npy'

EXP_PARAM_VALUES = [0.0, 0.005, 0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1.0]

exp_run = [EXP_NAME + '_' + str(val) for val in EXP_PARAM_VALUES]
output_dirs = [os.path.join('.',OUTDIR, exp) for exp in exp_run]  
    
#container for true rates as a [num expts  x num thresholds] 
for i,output_dir in enumerate(output_dirs):
    
    master_eqns_mean_states = np.load(os.path.join(output_dir,master_eqns_fname))
    kinetic_eqns_statuses = np.load(os.path.join(output_dir,kinetic_eqns_fname),allow_pickle=True).tolist()
    
    statuses_final_time = kinetic_eqns_statuses[-1]
    mean_states_final_time = master_eqns_mean_states[:,-1]
    population = int(mean_states_final_time.size / 5)
    true_infected = np.array([1 if status == 'I' else 0 for status in statuses_final_time.values()])
    infectiousness = mean_states_final_time[population:2*population] 
    
    #sorting
    sorted_idx = np.argsort(infectiousness)
    infectiousness = infectiousness[sorted_idx]
    true_infected = true_infected[sorted_idx]

    xticks = np.arange(population) 
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(xticks, true_infected, color='C0', marker='x', label='True infected')
    ax.scatter(xticks, infectiousness, color='C1', marker='o', label='Predicted infectiousness')
    plt.legend(loc='lower left')
    fig_name = os.path.join(OUTDIR,'ordered_infected_'+exp_run[i]+'.png')
    fig.savefig(fig_name,dpi=300)
    print("saved figure at", fig_name)
