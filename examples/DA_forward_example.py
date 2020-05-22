import os, sys; sys.path.append(os.path.join(".."))

import EoN
import numpy as np
import multiprocessing
from multiprocessing import get_context
import networkx as nx
import time
import pickle
import pdb

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

if __name__ == "__main__":

    np.random.seed(10)
    print("Number of cpu : ", multiprocessing.cpu_count())

    # Number of steps for the data assimilation algorithm.
    # This data assimilation algorithm is called the Ensemble Adjusted Kalman Filter (EAKF).
    steps_DA = 50
    # Ensemble size (required>=2)
    n_samples = 100 # 100
    # Number of status for each node
    # Statuses: S, I, H, R, D
    n_status = 5

    
    # Load the true data (beta=0.04, T=100, dt=1)
    print("run forward_example.py in ../sandbox/ then comment these lines")
    #exit()
    
    fin = os.getcwd()+'/../sandbox/states_truth_citynet.pkl'
    data = pickle.load(open(fin, 'rb'))
    data_mean = np.mean(data, 0)
    data_cov = np.var(data, 0)
    data_cov = np.maximum(data_cov, 1e-3)
    data_mean = data_mean.T
    data_cov = data_cov.T

    # Load network
    G = load_G()
    N = len(G)
    model_n_samples = n_samples 
    Gs = model_settings(model_n_samples, G)

    master_eqn_model = get_model(Gs, model_n_samples, N)
    master_eqn_model.init(beta = 0.06, hom = False)

    # Set prior for unknown parameters
    params = np.zeros([n_samples, n_status*N])
    params = pickle.load(open(os.getcwd()+'/../sandbox/sigma_ens.pkl', 'rb'))

    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    states_IC[:, :] = get_IC(master_eqn_model, model_n_samples, N)

    #We have a long intervention window of time T 
    #We have a short window for updates with data dt.
    
    T = 1.0 #1 day per intervention 
    dt = 0.25 #1/4 day update with data
    steps_T = int(T/dt)
    t_range=np.flip(np.arange(T,0.0,-dt)) # [dt,2dt,3dt,...,T-dt,T] (Excludes '0')
    dt_fsolve =0.25/4.0
   
    # Container for forward model evaluations
    x_forward_all = np.empty([n_samples, n_status*N, steps_T*steps_DA])
    #with the initial
    x_forward_all[:,:,0]=states_IC
   
    
    #Observations
    #HighProbRandomStatusObservation(N,
    #                                n_status,
    #                                fraction of nodes to observe,  
    #                                [statuses to observe] ([1] = I),  
    #                                min probability threshold for observation,
    #                                max probability threshold for observation,
    #                                type of threshold requirement: 'mean' ensemble within threshold,
    #                                observation name,
    #                                Size of Gaussian noise variance in observation)   

    #noise distribution:
    #JL: x_cov = np.diag(np.maximum((0.01*x_obs)**2, 1e-3))
    good_obs_var=1e-2

    
    SmartInfectiousObs = HighProbRandomStatusObservation(N,n_status,1.0,[1],0.1,0.75,'mean','0.25<=0.1_SmartInfected<=0.75',good_obs_var)
    #DumbInfectiousObs = HighProbRandomStatusObservation(N,n_status,0.5,[1],0.25,0.75,'mean','0.25<=0.5_DumbInfected<=0.75',0.1)
    #SmartDeceasedObs= HighProbRandomStatusObservation(N,n_status,0.5,[1],0.5,0.75,'mean','0.5<=0.5_SmartDecesed<=0.75',0.01)

    #OModel=[SmartInfectiousObs,DumbInfectiousObs,SmartDeceasedObs]
    OModel=SmartInfectiousObs
    
    #EModel - error model, one, for each observation mode to check effectiveness of our observations
    EModel = HighProbRandomStatusObservation(N,n_status,1.0,[2],0.5,1.0,'mean','All_Infected>=0.5',0.0)   

    #Build the DA
    ekf=DataAssimilator(params,OModel,EModel)

    #ekf = EAKF(params, states_IC)
    #For each DA intervention window
    for DAstep in range(steps_DA):
        print('DA step: ', DAstep+1)

        x_forward=np.zeros([n_samples,n_status*N,t_range.size])

        #For each static contact interval:
        for idx_local,tt in enumerate(t_range):

            #local index idx_local
            #global index idx_global
            idx_global=1 + DAstep*steps_T + idx_local
            tt_global=DAstep*T+tt

            #When we have a series of contact networks
            #master_eqn_model.set_contact_network(contact_networks[idx_global])

            ## Forward model evaluation of all ensemble members
            start = time.time()
            if idx_local==0: 
                xftmp = master_eqn_model.ens_solve_euler(states_IC, [dt])
                x_forward[:,:,idx_local] = xftmp[:,:,0]#as atm master_eqn can't deal with single "dt" input
            else:
                xftmp = master_eqn_model.ens_solve_euler(x_forward[:,:,idx_local-1], [dt])
                x_forward[:,:,idx_local] = xftmp[:,:,0]
                
            end = time.time()
            print('Time elapsed for forward model: ', end - start)
    
            ## EAKF to update joint states
            start = time.time()
            
            new_params,x_forward[:,:,idx_local]=ekf.update(x_forward,idx_local,data_mean,idx_global)
            end = time.time()
            print('Assimilation time: ', tt_global,', Time elapsed for EAKF: ', end - start)
        
            #update master equation model parameters
            master_eqn_model.update_parameters(new_params)
            #this next line is overwritten at master_eqn_model runtime
            master_eqn_model.set_solver(T = T, dt = np.diff(t_range).min(), reduced = True)
            
        x_forward_all[:,:,DAstep*steps_T:(DAstep+1)*steps_T] = x_forward
        states_IC = x_forward[:,:,-1]
        #print("Error: ", ekf.damethod.error[-1])

        ## Output files 
        ## Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.params, open("data/u.pkl", "wb"))
        #pickle.dump(ekf.x, open("data/g.pkl", "wb"))
        pickle.dump(ekf.damethod.error, open("data/error.pkl", "wb"))
        pickle.dump(x_forward_all, open("data/x.pkl", "wb"))


