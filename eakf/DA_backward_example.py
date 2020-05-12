import EoN
import numpy as np
import multiprocessing
from models import MasterEqn 
import networkx as nx
from eakf import EAKF
import time
import pickle
import pdb

def backward_model(G, params, stateT0, T0, T, dt_max, t_range):
    model = MasterEqn(G)
    # Dimension of states: num_status * num_nodes, num_time_steps
    states = model.backwards_solve(params, stateT0, T0, T, dt_max, t_range)
    return states.T

def ensemble_backward_model(G, qi, xi, T0, T, dt_max, t_range, parallel_flag = True):
    if parallel_flag == True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = []
        for iterN in range(qi.shape[0]):
            results.append(pool.apply_async(backward_model, (G, qi[iterN,:], xi[iterN,:], T0, T, dt_max, t_range)))
            iterN = 0
        states_all = np.zeros([qi.shape[0], len(t_range), xi.shape[1]])
        for result in results:
            states_all[iterN,:,:] = result.get()
            iterN = iterN + 1
        pool.close()
    else:
        states_all = np.zeros([qi.shape[0], len(t_range), xi.shape[1]])
        for iterN in range(qi.shape[0]):
            states_all[iterN,:,:] = backward_model(G, qi[iterN,:], xi[iterN,:], T0, T, dt_max, t_range)
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
    E, I, H, R, D = 0.01*np.ones([5, N])
    S = 0.95*np.ones(N,)
    I[infected] = 0.95
    S[infected] = 0.01
    state0 = np.hstack((S, E, I, H, R, D))
    return state0

if __name__ == "__main__":

    np.random.seed(10)
    print("Number of cpu : ", multiprocessing.cpu_count())

    # Number of EAKF steps
    steps_DA = 5
    # Ensemble size (required>=2)
    n_samples = 10
    # Number of status for each node
    n_status = 6

    # Load initial conditions of backward DA from a forward DA run
    # Need to run DA_forward_example.py to output g.pkl
    fin = 'data/x.pkl'
    x_forward_all = pickle.load(open(fin, 'rb'))
    x_backward_IC = x_forward_all[:,-1,:]
    
    # Load the true data (beta=0.04, T=100, dt=1)
    fin = 'data/states_truth_beta_0p04.pkl'
    data = pickle.load(open(fin, 'rb'))
    print(data.shape)
    data = data[:,:x_forward_all.shape[1]]
    data = np.fliplr(data)
    data = data.T

    
    # Load network
    G, N = load_G()

    # in epimodels i use clipping
    print('using clipping')
    # Set prior for unknown parameters
    #fin = 'data/u.pkl'
    #params_forward=pickle.load(open(fin,'rb'))
    params = np.zeros([n_samples,1])
    #print(params_forward.shape)
    #params[:,0]=params_forward[-1,:,0]
    params[:,0] = np.random.uniform(np.log(0.001), np.log(0.1), n_samples)

    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    for iterN in range(n_samples):
        #states_IC[iterN, :] = random_IC()
        states_IC[iterN, :] = x_backward_IC[iterN, :] 

    # Set time informations inside an EAKF step
    T = 5.0
    Tinit = 0.0
    dt = 1.0 #timestep for OUTPUT not solver
    steps_T = int(T/dt)
    # t_range = np.linspace(0.0, T, num=steps_T+1, endpoint=True)#includes 0 and T
    t_range=np.arange(0.0,T,dt) 
    dt_bsolve =0.1
    
    # Container for backward model evaluations
    x_backward_all = np.empty([n_samples, steps_T*steps_DA, n_status*N])

    ekf = EAKF(params, states_IC)
    for iterN in range(steps_DA):
        print('DA step: ', iterN+1)
        print(np.exp(ekf.q[iterN]))
        ## Get observations for EAKF
        x_obs = data[(iterN+1)*int(T)-1,:]
        x_cov = np.identity(x_obs.shape[0]) * 0.01 
        #x_cov = np.diag(np.maximum((0.1*x_obs)**2, 1e-2))
        ekf.obs(x_obs, x_cov)

        ## Forward model evaluation of all ensemble members
        start = time.time()
        # A simple decayed noise term to add into paramters during EAKF
        params_noises = np.random.uniform(-2./np.sqrt(iterN+1), 2./np.sqrt(iterN+1), n_samples) 
        x_backward = ensemble_backward_model(G, ekf.q[iterN] + params_noises.reshape(n_samples,1), \
                                             states_IC, \
                                             T, Tinit, dt_bsolve, t_range, parallel_flag = False)
        end = time.time()
        print('Time elapsed for backward model: ', end - start)
    
        ## EAKF to update joint states
        start = time.time()
        ekf.update(x_backward[:,-1,:])
        x_backward[:,-1,:] = ekf.x[-1]
        x_backward_all[:,iterN*steps_T:(iterN+1)*steps_T,:] = x_backward
        end = time.time()
        print('Time elapsed for EAKF: ', end - start)
        states_IC = x_backward[:,-1,:]
        print("Error: ", ekf.error[-1])

        ## Output files 
        ## Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.q, open("data/u_back.pkl", "wb"))
        pickle.dump(ekf.x, open("data/g_back.pkl", "wb"))
        pickle.dump(ekf.error, open("data/error_back.pkl", "wb"))
        pickle.dump(x_backward_all, open("data/x_back.pkl", "wb"))


        
