import EoN
import numpy as np
import multiprocessing
from models import MasterEqn 
import networkx as nx
from eakf import EAKF
import time
import pickle
from DA_forward_plot import plot_states

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

if __name__ == "__main__":
    """ 
    This runs the EAKF as a moving 'forward smoother'.
    Step 0:   Run forward solver for ensemble of solutions U(t) for some initial time Tinit = T_0
    ***ENTER LOOP k=0,1,2,...***
    Step 1:   Run forward solver for solution U(t) over assimilation time window T_k -> T_{k+1}
    Step 2:   Receive data {y_i} i=1,...,M at times (T_{k,k+1}^i) within the window 
    Step 3:   Assimilate all data as a smoother over the window with an EAKF update with combined loss function
    Step 4:   Draw new ensemble of U(t_{k+1}), repeat from step 1.
    """

    ### Preliminaries ###
    np.random.seed(10)
    print("Number of cpu : ", multiprocessing.cpu_count())

    #parallel ensembles?
    parflag=False
    ### Set up Network ##

    # Load network
    G, N = load_G()

    # Number of status for each node
    n_status = 6
   
    ### Set up DA ###

    # Number of EAKF steps
    steps_DA = 20
    # Ensemble size (required>=2)
    n_samples = 50


    # Set prior for unknown parameters
    params = np.zeros([n_samples,1])
    params[:,0] = np.random.uniform(np.log(0.01), np.log(0.5), n_samples)

    # Load the true data (beta=0.04, T=100, dt=1)
    fin = 'data/states_truth_beta_0p04.pkl'
    data = pickle.load(open(fin, 'rb'))
    data = data.T
    
    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    for iterN in range(n_samples):
        states_IC[iterN, :] = random_IC()
    
    # Construct EAKF object
    ekf = EAKF(params, states_IC)
    
    ### Step 0 ###
    T_init = 0.0
    dt_init = 0.1 
    steps_T_init=int(T_init/dt_init)
    t_range_init = np.linspace(0.0, T_init, num=steps_T_init+1, endpoint=True)#includes 0 and T_init
    dt_fsolve =T_init/10.

    # Parameters for each EAKF step
    T = 5.0
    dt = 1.0 #timestep for OUTPUT not solver
    steps_T = int(T/dt)
    t_range=np.flip(np.arange(T,0.0,-dt)) # [dt,2dt,3dt,...,T-dt,T] (Excludes '0')
    t_range0=np.hstack([0.0,t_range])
    dt_fsolve =T/10.

    # Container for forward model evaluations
    x_forward_all = np.empty([n_samples, (1+steps_T_init)+steps_T*steps_DA, n_status*N])
    x_forward_all[:,0,:]=states_IC
    #Run initial step
    if (T_init>0.0):
        #Run forward model 0->T_init inclusive
        start=time.time()
        x_forward_init = ensemble_forward_model(G, ekf.q[0], states_IC,T_init, dt_fsolve, t_range_init, parallel_flag = parflag)
        x_forward_all[:,1:steps_T_init+1,:]=x_forward_init #This inserts values in 0:steps_T_init (inclusive)
        end=time.time()
        print('Time elapsed for initialization forward model: ', end - start)

    #Choice of observations  [Make object for this info]
    #When: Create (distinct) indices at which we obtain data

    obs_type="single_rand"
    if obs_type=="fixed": 
        #Case 1: single repeated obs time each window
        obs_times=np.arange(steps_DA)*steps_T + (steps_T-1)
    elif obs_type=="single_rand":
        #Case 2: single observation time per assimilation window 
        obs_times=steps_T*np.arange(steps_DA) + np.random.randint(steps_T,size=steps_DA)
        #print(obs_times)
    #elif obs_type=="mult_rand":
        #<Not Yet Implemented> Case 3:random observation times in whole path
        #obs_times_tot=20
        #obs_times=shuffle(np.arange(steps_T*steps_DA))
        #obs_times=np.array(sorted(obs_times[:obs_times_tot]))
    else:
        print('observation type not recognised')
        exit()

    # Which states to observe
    # S E I H R D
    obs_status=np.array([0,1,2,3,4,5])
    obs_status=np.hstack([np.arange(N)+i*N for i in obs_status])

    # Which nodes to observe
    obs_nodes_tot=int(0.9*N)
    obs_nodes=shuffle(np.arange(N))
    obs_nodes=np.array(sorted(obs_nodes[:obs_nodes_tot]))
    
        
    
    ### ***ENTER LOOP*** ###
    for iterN in range(steps_DA):
        print('DA step: ', iterN+1)

        ### STEP 1 ###
        ## Forward model evaluation of all ensemble members from the final time of the previous step.
        start = time.time()
        # A simple decayed noise term to add into paramters during EAKF
        params_noises = np.random.uniform(-2./np.sqrt(iterN+1), 2./np.sqrt(iterN+1), n_samples)
        x_forward0 = ensemble_forward_model(G, ekf.q[-1] + params_noises.reshape(n_samples,1), x_forward_all[:,steps_T_init+iterN*steps_T,:],T, dt_fsolve, t_range0, parallel_flag = parflag)
        x_forward=x_forward0[:,1:,:]#exclude 0
        end = time.time()
        print('Time elapsed for forward model: ', end - start)

        ### Step 2 ###
        ## Get observations for EAKF, and get index for the x_forward ensembles
        obs_in_window=np.flatnonzero(np.logical_and(obs_times>=steps_T*iterN,obs_times<steps_T*(iterN+1)))
        
        if (obs_in_window.size>0):
            obs_in_window=obs_times[obs_in_window]
            data_index=obs_in_window - 1- (iterN*steps_T)#the index of the window - iterations - the IC's in 0 index 
            print('obs at time (excluding T_init):',iterN*T+t_range[data_index[0]], ', i.e indices ', data_index[0]+1, 'in window ', iterN)
            
            x_obs = data[obs_in_window[0],:]
            x_cov = np.identity(x_obs.shape[0]) * 0.05 
            #currently, for EACH observation we need to add and update
            ekf.obs(x_obs, x_cov)

            ### Step 3 ###
            ## EAKF to update joint states
            start = time.time()
            ekf.update(x_forward[:,data_index[0],:]) #update based on single data point(obs_in_window)
            x_forward[:,data_index[0],:] = ekf.x[-1] #the x is appended to the end of a list
            end = time.time()
            print('Time elapsed for EAKF: ', end - start)
            print("Error: ", ekf.error[-1])
            
            ### Step 4 ###
            #Forward propogate the data adjusted trajectories (and model parameters) to the end of the time window from the data point
          
            T_prop=T-data_index[0]*dt
            t_range_prop=t_range[data_index[0]:]-t_range[data_index[0]]#note this includes 0
            if (t_range_prop.size>0):
                start = time.time()
                params_noises = np.random.uniform(-2./np.sqrt(iterN+1), 2./np.sqrt(iterN+1), n_samples)
                x_forward[:, data_index[0]: ,:] = ensemble_forward_model(G, ekf.q[-1] + params_noises.reshape(n_samples,1), ekf.x[-1], T_prop , dt_fsolve, t_range_prop, parallel_flag = parflag)
                end = time.time()
                print('Time elapsed for forward model (to end of window): ', end - start)
            else:
                print('no propagation required (to end of window)')
        else:
            print('no observations in current window')

        #store pre data trajectory
        x_forward_all[:,(1+steps_T_init)+iterN*steps_T:(1+steps_T_init)+(iterN+1)*steps_T,:] = x_forward

        ## Output files 
        ## Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.q, open("data/u.pkl", "wb"))
        pickle.dump(ekf.x, open("data/g.pkl", "wb"))
        pickle.dump(ekf.error, open("data/error.pkl", "wb"))
        pickle.dump(x_forward_all, open("data/x.pkl", "wb"))


    #plots
    t_range_total=np.hstack([t_range+steps_T*i for i in np.arange(steps_DA)])
    if T_init>0.0:
        t_range_total=np.hstack([t_range_init,t_range_total])
    else:
        t_range_total=np.hstack([0.0,t_range_total])

    print(t_range_total)
    plot_states(data[obs_times,:],x_forward_all,T_init+obs_times*dt,t_range_total,6,N,'forward_filter')

        
