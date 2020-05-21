import EoN
import numpy as np
import multiprocessing
from multiprocessing import get_context
from models import MasterEqn 
import networkx as nx
#from eakf import EAKF
from eakf_svd_truncated import EAKF
import time
import pickle
from utilities import load_G, model_settings, get_model, get_IC 
import pdb

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
    steps_DA = 10
    # Ensemble size (required>=2)
    n_samples = 100 # 100
    # Number of status for each node
    n_status = 5

    # Load the true data (beta=0.04, T=100, dt=1)
    fin = './states_truth_citynet.pkl'
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

    model = get_model(Gs, model_n_samples, N)
    model.init(beta = 0.06, hom = False)

    # Set prior for unknown parameters
    params = np.zeros([n_samples, n_status*N])
    params = pickle.load(open('sigma_ens.pkl', 'rb'))

    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    states_IC[:, :] = get_IC(model, model_n_samples, N)

    # Set time informations inside an EAKF step
    T = 4.0
    dt = 0.25 #timestep for OUTPUT not solver
    steps_T = int(T/dt)
    #OD: Are you sure you want to exclude T here? .
    t_range = np.linspace(0.0, T, num=steps_T+1, endpoint=True)
    #t_range=np.arange(0.0,T,dt) 
    dt_fsolve =T/16.
    
    # Container for forward model evaluations
    x_forward_all = np.empty([n_samples, n_status*N, steps_T*steps_DA])

    ekf = EAKF(params, states_IC)
    for iterN in range(steps_DA):
        print('DA step: ', iterN+1)

        ## Get observations for EAKF
        x_obs = data_mean[(iterN+1)*int(T/dt),:]
        #x_cov = np.identity(x_obs.shape[0]) * 0.01 
        x_cov = np.diag(np.maximum((0.01*x_obs)**2, 1e-3))
        #x_cov = np.diag(np.maximum((0.1*x_obs)**2, 1e-2))
        #x_cov = np.diag(data_cov[(iterN+1)*int(T/dt),:])
        ekf.obs(x_obs, x_cov)

        ## Forward model evaluation of all ensemble members
        start = time.time()
        # A simple decayed noise term to add into paramters during EAKF
        #if iterN > 0:
        #    ekf.q[iterN] = add_noises(ekf.q[iterN], -1e-2/np.sqrt(iterN+1), 1e-2/np.sqrt(iterN+1), n_samples)
        #    ekf.q[iterN] = np.clip(ekf.q[iterN], np.log(0.01), np.log(0.1))
        #x_forward = ensemble_forward_model(Gs, model_n_samples, N, \
        #                                   ekf.q[iterN], states_IC,T, dt_fsolve, t_range, \
        #                                   parallel_flag = True)

        ## Currently params are not updated by EAKF (just for code verification). 
        #model.update_parameters(ekf.q[iterN])
        model.update_parameters(params)

        model.set_solver(T = T, dt = np.diff(t_range).min(), reduced = True)
        x_forward = model.ens_solve_euler(states_IC, t_range)
        end = time.time()
        print('Time elapsed for forward model: ', end - start)
    
        ## EAKF to update joint states
        start = time.time()
        ekf.update(x_forward[:,:,-1])
        x_forward[:,:,-1] = ekf.x[-1]
        x_forward_all[:,:,iterN*steps_T:(iterN+1)*steps_T] = x_forward[:,:,:-1]
        end = time.time()
        print('Time elapsed for EAKF: ', end - start)
        states_IC = x_forward[:,:,-1]
        print("Error: ", ekf.error[-1])

        ## Output files 
        ## Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.q, open("data/u.pkl", "wb"))
        pickle.dump(ekf.x, open("data/g.pkl", "wb"))
        pickle.dump(ekf.error, open("data/error.pkl", "wb"))
        pickle.dump(x_forward_all, open("data/x.pkl", "wb"))
