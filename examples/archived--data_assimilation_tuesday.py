import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import time
import pickle

from epiforecast.observations import HighProbRandomStatusObservation
from epiforecast.data_assimilator import DataAssimilator
#When they exist:
from epiforecast.risk_simulation import MasterEquationModelEnsemble

from sandbox.models import MasterEqn 
from sandbox.utilities import load_G, model_settings, get_model, get_IC 

def add_noises(q, v_min, v_max, n_samples):
    q = np.exp(q)
    params_noises = np.random.uniform(v_min, v_max, n_samples)
    q = np.clip(q + params_noises.reshape(n_samples,1), 0.01, 0.1)
    q = np.log(np.maximum(q, 1e-6))
    return q

np.random.seed(10)

# Ensemble size (required>=2)
n_samples = 100 # 100

# Number of status for each node
# Statuses: S, I, H, R, D
n_status = 5

#We have a short window for updates with data static_network_interval.    
intervention_interval = 1.0 #1 day 
static_network_interval = 0.25 #1/4 day update with data
steps_intervention_interval = int(intervention_interval/static_network_interval)
intervals_per_window=np.flip(np.arange(intervention_interval,0.0,-static_network_interval)) # [static_network_interval, ... ,intervention_interval] (Excludes '0')


#### EvolvingContactNetworkGenerator goes here
contact_network = load_G()
N = len(contact_network)

# MasterEquationModelEnsemble goes here
master_eqn_model_n_samples = n_samples 
network_rates = model_settings(master_eqn_model_n_samples, contact_network)

master_eqn_model = get_model(network_rates, master_eqn_model_n_samples, N)
master_eqn_model.init(beta = 0.06, hom = False)

# Set prior for unknown parameters
params_from_file = pickle.load(open(os.getcwd()+'/../sandbox/sigma_ens.pkl', 'rb'))
params = np.zeros([params_from_file.shape[0],params_from_file.shape[1],steps_intervention_interval+1])
params[:,:,0] = params_from_file
#######

#### KineticModel goes here
# Load the true data (beta=0.04, T=100, static_network_interval=1)
# print("run forward_example.py in ../sandbox/ then comment these lines")
# exit()
    
fin = os.getcwd()+'/../sandbox/states_truth_citynet.pkl'
data = pickle.load(open(fin, 'rb'))
synthetic_data = np.mean(data, 0)
synthetic_data = synthetic_data.T
##### 


# TUESDAY ICS GO HERE
states_IC = np.zeros([n_samples, n_status*N])
states_IC[:, :] = get_IC(master_eqn_model, master_eqn_model_n_samples, N)
#####

# Container for forward model evaluations
x_forward_all = np.empty([n_samples, n_status*N, 1+steps_intervention_interval])
#with the initial
x_forward_all[:,:,0]=states_IC

#Set initial solver
master_eqn_model.set_solver(T = intervention_interval, dt = np.diff(intervals_per_window).min(), reduced = True)


#noise distribution:
obs_var=1e-2

#Observations    
infectious_obs=HighProbRandomStatusObservation(N,1.0,[1],0.02,0.98,'mean',obs_var,'All_Infected>=0.5')   
omodel= [infectious_obs] 
    
#emodel - error model, one, for each observation mode to check effectiveness of our observations
#emodel = HighProbRandomStatusObservation(N,1.0,[1],0.5,1.0,'mean',0.0,'All_Infected>=0.5')   

emodel=[]

#Build the DA
assimilator=DataAssimilator(omodel,emodel)

#For each DA intervention window
x_forward=np.zeros([n_samples,n_status*N,intervals_per_window.size])

#For each static contact interval:
for idx_local,tt in enumerate(intervals_per_window):

    #local idx does not see ICs so we add 1 where a more global index is required e.g indexing data (starts at 0)
    
    #### WHEN WE HAVE A SERIES OF CONTACT NETWORKS
    #master_eqn_model.set_contact_network(contact_networks[idx_local+1])

    ## Forward model evaluation of all ensemble members
    start = time.time()
    if idx_local==0: 
        x_forward = master_eqn_model.ens_solve_euler(states_IC, static_network_interval)
    else:
        x_forward[:,:,idx_local] = master_eqn_model.ens_solve_euler(x_forward[:,:,idx_local-1], static_network_interval)
    print(x_forward[:,1::5,idx_local])    
    end = time.time()
    print('Time elapsed for forward model: ', end - start)
    
    ## EAKF to update joint states
    start = time.time()
    x_forward[:,:,idx_local],params[:,:,idx_local+1],_ = assimilator.update(x_forward[:,:,idx_local],
                                                                          synthetic_data[idx_local+1,:],
                                                                          ensemble_transition_rates=params[:,:,idx_local+1])
    end = time.time()
    print('Assimilation time: ', tt,', Time elapsed for EAKF: ', end - start)

    #update master equation model parameters
    master_eqn_model.update_parameters(params[idx_local+1])
    #this next line is overwritten at master_eqn_model runtime
    master_eqn_model.set_solver(T = intervention_interval, dt = np.diff(intervals_per_window).min(), reduced = True)
    
x_forward_all[:,:,1:] = x_forward
states_IC = x_forward[:,:,-1]

#print("Error: ", assimilator.damethod.error[-1])

## Output files 
## Overwrite for each EAKF step to facilitate debugging
pickle.dump(assimilator.params, open("data/u.pkl", "wb"))
pickle.dump(assimilator.damethod.error, open("data/error.pkl", "wb"))
pickle.dump(x_forward_all, open("data/x.pkl", "wb"))
