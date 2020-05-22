import EoN
import numpy as np
import multiprocessing
from multiprocessing import get_context
from models import MasterEqn 
import networkx as nx
from eakf import EAKF
#from eakf_svd_truncated import EAKF
import time
import pickle

def forward_model(G, params, state0, T, dt_max, t_range):
    model = MasterEqn(G)
    # Dimension of states: num_status * num_nodes, num_time_steps
    states = model.solve(params, state0, T, dt_max, t_range)
    return states.T

def ensemble_forward_model(G, qi, xi, T, dt_max, t_range, parallel_flag = True):
    if parallel_flag == True:
        with get_context("spawn").Pool(multiprocessing.cpu_count()) as pool:
            results = []
            for iterN in range(qi.shape[0]):
                results.append(pool.apply_async(forward_model, (G, qi[iterN,:], xi[iterN,:], T, dt_max, t_range)))
            pool.close()
            pool.join()
            iterN = 0
            states_all = np.zeros([qi.shape[0], len(t_range), xi.shape[1]])
            for result in results:
                states_all[iterN,:,:] = result.get()
                iterN = iterN + 1
    else:
        states_all = np.zeros([qi.shape[0], len(t_range), xi.shape[1]])
        for iterN in range(qi.shape[0]):
            states_all[iterN,:,:] = forward_model(G, qi[iterN,:], xi[iterN,:], T, dt_max, t_range)
    return states_all 

def load_G():
    N = 1000
    G = nx.fast_gnp_random_graph(N, 5./(N-1))
    # G = nx.powerlaw_cluster_graph(N, 20, 5./(N-1))

    edge_list = np.loadtxt('../data/networks/High-School_data_2013.csv', usecols = [1,2], dtype = int)
    # edge_list = np.loadtxt('networks/thiers_2012.csv', usecols = [1,2], dtype = int)

    G = nx.Graph([tuple(k) for k in edge_list])
    G = nx.relabel_nodes(G,dict(zip(list(G.nodes()), range(len(list(G.nodes()))))))
    return G, len(list(G.nodes)) 

def random_IC():
    infected = np.random.randint(N, size = 10)
    E, I, H, R, D =0.01*np.ones([5, N])
    S = 0.95*np.ones(N,)
    I[infected] = 0.95
    S[infected] = 0.01
    state0 = np.hstack((S, E, I, H, R, D))
    return state0

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
    steps_DA = 9 
    # Ensemble size (required>=2)
    n_samples = 100 # 100
    # Number of status for each node
    n_status = 6

    # Load the true data (beta=0.04, T=100, dt=1)
    fin = 'data/states_truth_beta_0p04.pkl'
    data = pickle.load(open(fin, 'rb'))
    data = data.T

    # Load network
    G, N = load_G()

    # Set prior for unknown parameters
    params = np.zeros([n_samples,1])
    params[:, 0] = np.random.uniform(np.log(0.01), np.log(0.1), n_samples)

    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    for iterN in range(n_samples):
        states_IC[iterN, :] = random_IC()

    # Set time informations inside an EAKF step
    T = 10.0
    dt = 1.0 #timestep for OUTPUT not solver
    steps_T = int(T/dt)
    #OD: Are you sure you want to exclude T here? .
    t_range = np.linspace(0.0, T, num=steps_T+1, endpoint=True)
    dt_fsolve =T/10.
    
    # Container for forward model evaluations
    x_forward_all = np.empty([n_samples, steps_T*steps_DA, n_status*N])

    ekf = EAKF(params, states_IC)
    for iterN in range(steps_DA):
        print('DA step: ', iterN+1)

        ## Get observations for EAKF
        x_obs = data[(iterN+1)*int(T),:]
        #x_cov = np.identity(x_obs.shape[0]) * 0.01 
        x_cov = np.diag(np.maximum((0.01*x_obs)**2, 1e-3))
        ekf.obs(x_obs, x_cov)

        ## Forward model evaluation of all ensemble members
        start = time.time()
        # A simple decayed noise term to add into paramters during EAKF
        if iterN > 0:
            ekf.q[iterN] = add_noises(ekf.q[iterN], -2e-2/np.sqrt(iterN+1), 2e-2/np.sqrt(iterN+1), n_samples)
            ekf.q[iterN] = np.clip(ekf.q[iterN], np.log(0.01), np.log(0.1))
        x_forward = ensemble_forward_model(G, ekf.q[iterN], states_IC,T, dt_fsolve, t_range, parallel_flag = True)
        end = time.time()
        print('Time elapsed for forward model: ', end - start)
    
        ## EAKF to update joint states
        start = time.time()
        ekf.update(x_forward[:,-1,:])
        x_forward[:,-1,:] = ekf.x[-1]
        x_forward_all[:,iterN*steps_T:(iterN+1)*steps_T,:] = x_forward[:,:-1,:]
        end = time.time()
        print('Time elapsed for EAKF: ', end - start)
        states_IC = x_forward[:,-1,:]
        print("Error: ", ekf.error[-1])

        ## Output files 
        ## Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.q, open("data/u.pkl", "wb"))
        pickle.dump(ekf.x, open("data/g.pkl", "wb"))
        pickle.dump(ekf.error, open("data/error.pkl", "wb"))
        pickle.dump(x_forward_all, open("data/x.pkl", "wb"))
