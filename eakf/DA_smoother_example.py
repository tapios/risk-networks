import EoN
import numpy as np
import multiprocessing
from models import MasterEqn 
import networkx as nx
from eakf import EAKF
import time
import pickle

def forward_model(G, params, state0, T, dt_max, t_range):
    model = MasterEqn(G)
    # Dimension of states: num_status * num_nodes, num_time_steps
    states = model.solve(params, state0, T, dt_max, t_range)
    return states.T

def ensemble_forward_model(G, qi, xi, T, dt_max, t_range, parallel_flag = True):
    if parallel_flag == True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = []
        for iterN in range(qi.shape[0]):
            results.append(pool.apply_async(forward_model, (G, qi[iterN,:], xi[iterN,:], T, dt_max, t_range)))
        iterN = 0
        states_all = np.zeros([qi.shape[0], len(t_range), xi.shape[1]])
        for result in results:
            states_all[iterN,:,:] = result.get()
            iterN = iterN + 1
        pool.close()
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
    E, I, H, R, D = np.zeros([5, N])
    S = np.ones(N,)
    I[infected] = 1.
    S[infected] = 0.
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
    steps_DA = 6 
    iterI_max = 3  # maximum iterations for Kalman smoother
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
    t_range=np.linspace(0.0,T,num=steps_T+1) 
    dt_fsolve = 1.
    
    # Container for forward model evaluations
    x_forward_all = np.empty([n_samples, (steps_T+1)*steps_DA, n_status*N])

    ekf = EAKF(params, np.hstack((states_IC, states_IC)))
    for iterN in range(steps_DA):
        print('DA step: ', iterN+1)

        # Get observations for EAKF
        x_obs = np.hstack((data[(iterN)*int(T),:], data[(iterN+1)*int(T),:]))
        x_cov = np.diag(np.maximum((0.01*x_obs)**2, 1e-3))
        ekf.obs(x_obs, x_cov)

        if iterN == 0:
            q_tmp = ekf.q[iterN]
            states_IC_tmp = states_IC

        # Iterative ensemble Kalman smoother:
        for iterI in range(iterI_max):
            print("Iteration " + str(iterI+1) + " for ensemble Kalman smoother:")

            start = time.time()
            # Add a simple noise term into paramters during EAKF
            q_tmp = add_noises(q_tmp, -2e-2, 2e-2, n_samples)
            q_tmp = np.clip(q_tmp, np.log(0.01), np.log(0.1))
            states_IC_tmp = np.clip(states_IC_tmp, 0., 1.)
            # Forward model evaluation of all ensemble members
            x_forward = ensemble_forward_model(G, q_tmp, states_IC_tmp, T, dt_fsolve, t_range, parallel_flag = True)
            end = time.time()
            print('Time elapsed for forward model: ', end - start)
    
            # EAKF to update joint states
            start = time.time()
            x_forward_joint = np.hstack((states_IC_tmp, x_forward[:,-1,:]))
            ekf.update(x_forward_joint)

            q_tmp = ekf.q[-1]
            states_IC_tmp = ekf.x[-1,:,:n_status*N]
            ekf.q = np.delete(ekf.q, -1, 0)
            ekf.x = np.delete(ekf.x, -1, 0)
            end = time.time()
            print('Time elapsed for EAKS: ', end - start)
            print("Error: ", ekf.error[-1])
            if ekf.error[-1] < 1e-4:
                break

        start = time.time()
        # Add a simple noise term into paramters during EAKF
        q_tmp = add_noises(q_tmp, -1e-2, 1e-2, n_samples)
        q_tmp = np.clip(q_tmp, np.log(0.01), np.log(0.1))
        ekf.q[-1] = q_tmp
        states_IC_tmp = np.clip(states_IC_tmp, 0., 1.)
        x_forward = ensemble_forward_model(G, q_tmp, states_IC_tmp, T, dt_fsolve, t_range, parallel_flag = True)
        end = time.time()
        print('Time elapsed for forward model: ', end - start)
    
        # EAKF to update joint states
        print('Start EAKF:')
        start = time.time()
        x_forward_joint = np.hstack((states_IC_tmp, x_forward[:,-1,:]))
        ekf.update(x_forward_joint)

        x_forward[:,-1,:] = ekf.x[-1,:,n_status*N:]
        x_forward_all[:,iterN*(steps_T+1):(iterN+1)*(steps_T+1),:] = x_forward
        ekf.q[-1] = np.clip(ekf.q[-1], np.log(0.01), np.log(0.1))
        q_tmp = ekf.q[-1]
        states_tmp_IC = x_forward[:,-1,:]

        end = time.time()
        print('Time elapsed for EAKF: ', end - start)
        print("Error: ", ekf.error[-1])

        # Output files 
        # Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.q, open("data/u_smoother.pkl", "wb"))
        pickle.dump(ekf.x, open("data/g_smoother.pkl", "wb"))
        pickle.dump(ekf.error, open("data/error_smoother.pkl", "wb"))
        pickle.dump(x_forward_all, open("data/x_smoother.pkl", "wb"))
