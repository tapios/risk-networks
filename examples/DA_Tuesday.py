import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import multiprocessing
import time
import pickle

from epiforecast.observations import HighProbRandomStatusObservation
from epiforecast.data_assimilation import DataAssimilator
from sandbox.models import MasterEqn 
from sandbox.utilities import load_G, model_settings, get_model, get_IC 


def add_noises(q, v_min, v_max, n_samples):
    q = np.exp(q)
    params_noises = np.random.uniform(v_min, v_max, n_samples)
    q = np.clip(q + params_noises.reshape(n_samples,1), 0.01, 0.1)
    q = np.log(np.maximum(q, 1e-6))
    return q


np.random.seed(10)
print("Number of cpu : ", multiprocessing.cpu_count())

# Ensemble size (required>=2)
n_samples = 100 # 100
# Number of status for each node
# Statuses: S, I, H, R, D
n_status = 5

    
#### STATICNETWORKGENERATOR GOES HERE 
contact_network = load_G()
N = len(contact_network)

####
    

# MASTEREQUATIONMODELENSEMBLE GOES HERE
master_eqn_model_n_samples = n_samples 
network_rates = model_settings(master_eqn_model_n_samples, contact_network)

master_eqn_model = get_model(network_rates, master_eqn_model_n_samples, N)
master_eqn_model.init(beta = 0.06, hom = False)

# Set prior for unknown parameters
params = np.zeros([n_samples, n_status*N])
params = pickle.load(open(os.getcwd()+'/../sandbox/sigma_ens.pkl', 'rb'))

#######

#### KINETICMODEL GOES HERE
# Load the true data (beta=0.04, T=100, static_network_interval=1)
#print("run forward_example.py in ../sandbox/ then comment these lines")
#exit()
    
fin = os.getcwd()+'/../sandbox/states_truth_citynet.pkl'
data = pickle.load(open(fin, 'rb'))
data_mean = np.mean(data, 0)
data_cov = np.var(data, 0)
data_cov = np.maximum(data_cov, 1e-3)
data_mean = data_mean.T
data_cov = data_cov.T
##### 



# TUESDAY ICS GO HERE
states_IC = np.zeros([n_samples, n_status*N])
states_IC[:, :] = get_IC(master_eqn_model, master_eqn_model_n_samples, N)

#We have a short window for updates with data static_network_interval.    
intervention_interval = 1.0 #1 day per intervention 
static_network_interval = 0.25 #1/4 day update with data
steps_intervention_interval = int(intervention_interval/static_network_interval)
static_network_interval_range=np.flip(np.arange(intervention_interval,0.0,-static_network_interval)) # [static_network_interval, ... ,intervention_interval] (Excludes '0')

# Container for forward model evaluations
x_forward_all = np.empty([n_samples, n_status*N, 1+steps_intervention_interval])
#with the initial
x_forward_all[:,:,0]=states_IC

#Set initial solver
master_eqn_model.set_solver(T = intervention_interval, dt = np.diff(static_network_interval_range).min(), reduced = True)


#noise distribution:
good_obs_var=1e-2

#Observations    

smart_infectious_obs = HighProbRandomStatusObservation(N,n_status,1.0,[1],0.1,0.75,'mean','0.25<=0.1_SmartInfected<=0.75',good_obs_var)
#dumb_infectious_obs = HighProbRandomStatusObservation(N,n_status,0.5,[1],0.25,0.75,'mean','0.25<=0.5_DumbInfected<=0.75',0.1)
#smart_deceased_obs= HighProbRandomStatusObservation(N,n_status,0.5,[1],0.5,0.75,'mean','0.5<=0.5_SmartDecesed<=0.75',0.01)

#omodel=[smart_infectious_obs, dumb_infectious_obs ,smart_deceased_obs ]
omodel= [] 
    
#EModel - error model, one, for each observation mode to check effectiveness of our observations
#emodel = HighProbRandomStatusObservation(N,n_status,1.0,[2],0.5,1.0,'mean','All_Infected>=0.5',0.0)   
emodel=[]

#Build the DA
ekf=DataAssimilator(params,[],[])

#ekf = EAKF(params, states_IC)
#For each DA intervention window
x_forward=np.zeros([n_samples,n_status*N,static_network_interval_range.size])

#For each static contact interval:
for idx_local,tt in enumerate(static_network_interval_range):
    #local idx does not see ICs so we add 1 where 'global' is required
    #When we have a series of contact networks
    #master_eqn_model.set_contact_network(contact_networks[idx_local+1])

    ## Forward model evaluation of all ensemble members
    start = time.time()
    if idx_local==0: 
        xftmp = master_eqn_model.ens_solve_euler(states_IC, [static_network_interval,static_network_interval+1])
        x_forward[:,:,idx_local] = xftmp[:,:,0]#as atm master_eqn can't deal with single "static_network_interval" input
    else:
        xftmp = master_eqn_model.ens_solve_euler(x_forward[:,:,idx_local-1], [static_network_interval,static_network_interval+1])
        x_forward[:,:,idx_local] = xftmp[:,:,0]
    print(x_forward[:,1::5,idx_local])    
    end = time.time()
    print('Time elapsed for forward model: ', end - start)
    
    ## EAKF to update joint states
    start = time.time()
            
    new_params,x_forward[:,:,idx_local]=ekf.update(x_forward,idx_local,data_mean,idx_local+1)
         
    end = time.time()
    print('Assimilation time: ', tt,', Time elapsed for EAKF: ', end - start)
        
    #update master equation model parameters
    master_eqn_model.update_parameters(new_params)
    #this next line is overwritten at master_eqn_model runtime
    master_eqn_model.set_solver(T = intervention_interval, dt = np.diff(static_network_interval_range).min(), reduced = True)
    
x_forward_all[:,:,1:] = x_forward
states_IC = x_forward[:,:,-1]
#print("Error: ", ekf.damethod.error[-1])

## Output files 
## Overwrite for each EAKF step to facilitate debugging
pickle.dump(ekf.params, open("data/u.pkl", "wb"))
#pickle.dump(ekf.x, open("data/g.pkl", "wb"))
pickle.dump(ekf.damethod.error, open("data/error.pkl", "wb"))
pickle.dump(x_forward_all, open("data/x.pkl", "wb"))


