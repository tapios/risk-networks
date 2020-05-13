import EoN
import numpy as np
import multiprocessing
from models import MasterEqn 
import networkx as nx
from data import HighSchoolData
from observations import FixedTimeObservation, RandomTimeObservation
from observations import FullStateObservation, RandomStateObservation
from data_assimilation import DAModel
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
    Step 0b:  Construct the DA model, Observations and data.
    ***ENTER LOOP k=0,1,2,...***
    Step 1:   Run forward solver for solution U(t) over assimilation time window T_k -> T_{k+1}
    Step 2:   Assimilate all data as a smoother over the window with an EAKF update with combined loss function
    Step 3:   Draw new ensemble of U(t_{k+1}), repeat from step 1.
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
    steps_DA = 10
    # Ensemble size (required>=2)
    n_samples = 20

    
    ######################
    ### --- Step 0 --- ###
    ######################
    # Set prior for unknown parameters
    params = np.zeros([n_samples,1])
    params[:,0] = np.random.uniform(np.log(0.01), np.log(0.1), n_samples)

    
    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    for iterN in range(n_samples):
        states_IC[iterN, :] = random_IC()


    T_init = 0.0
    dt_init = 1.0 #must be 1.0 for high school data set
    steps_T_init=int(T_init/dt_init)
    #t_range_init = np.linspace(0.0, T_init, num=steps_T_init+1, endpoint=True)#includes 0 and T_init
    t_range_init=np.flip(np.arange(T_init,0.0,-dt_init)) # [dt,2dt,3dt,...,T-dt,T] (Excludes '0')
    dt_fsolve =T_init/10.
 
    # Parameters for each EAKF step
    T = 5.0
    dt = 1.0 #timestep for OUTPUT not solver
    steps_T = int(T/dt)
    t_range=np.flip(np.arange(T,0.0,-dt)) # [dt,2dt,3dt,...,T-dt,T] (Excludes '0')
    dt_fsolve =T/10.

    # Container for forward model evaluations
    x_forward_all = np.empty([n_samples, (1+steps_T_init)+steps_T*steps_DA, n_status*N])
    x_forward_init=np.empty([n_samples,1,n_status*N])
    x_forward_init[:,0,:]=states_IC
    x_forward_all[:,0,:]=states_IC
    
    #Run initial step
    if (T_init>0.0):
        #Run forward model 0->T_init inclusive
        start=time.time()
        x_forward_init = ensemble_forward_model(G, params[:,0].reshape(n_samples,1), states_IC,T_init, dt_fsolve, t_range_init, parallel_flag = parflag)
        x_forward_all[:,1:steps_T_init+1,:]=x_forward_init #This inserts values in 0:steps_T_init (inclusive)
        end=time.time()
        print('Time elapsed for initialization forward model: ', end - start)

    #######################
    ### --- Step 0b --- ###
    #######################
    # Construct DA Models 
    # Data
    data=HighSchoolData(steps_T_init)
    
    # Observed states
    #OSModel=FullStateObservation(N,n_status)
    OSModel=RandomStateObservation(N,n_status,int(0.8*N))
                                   
    # Observed times
    OTModel = FixedTimeObservation(t_range,steps_T-1)# F(t_range,k) evaluates at t_range[k] at every window)
    #OTModel = RandomTimeObservation(t_range)
    
    #Build the DA
    ekf=DAModel(params,x_forward_init[:,-1,:],OSModel,OTModel,data)

    ### ***ENTER LOOP*** ###
    for iterN in range(steps_DA):
        print('DA step: ', iterN+1)

        ######################
        ### --- STEP 1 --- ###
        ######################
        
        ## Forward model evaluation of all ensemble members from the final time of the previous step.
        start = time.time()
        # A simple decayed noise term to add into paramters during EAKF
        params_noises = np.random.uniform(-2./np.sqrt(iterN+1), 2./np.sqrt(iterN+1), n_samples)        
        x_forward = ensemble_forward_model(G, ekf.damodel.q[-1] + params_noises.reshape(n_samples,1), x_forward_all[:,steps_T_init+iterN*steps_T,:],T, dt_fsolve, t_range, parallel_flag = parflag)
        end = time.time()
        print('Time elapsed for forward model: ', end - start)

        ######################
        ### --- Step 2 --- ###
        ######################
        
        # We Assume there is an observation in the window
        # EAKF to update joint states
        start = time.time()
        ekf.update(x_forward,iterN)
        #data_idx when obs took place, state_idx where was observed
        data_idx=ekf.otmodel.obs_time_in_window
        state_idx=ekf.osmodel.obs_states
        x_forward[:,data_idx,state_idx] = ekf.damodel.x[-1]
        
        end = time.time()
        print('Time elapsed for EAKF: ', end - start)
        print("Error: ", ekf.damodel.error[-1])
        print('new parameters,')
        print(np.exp(ekf.damodel.q[-1]))

        ######################
        ### --- Step 3 --- ###
        ######################
        
        #Forward propogate (if needed) the data adjusted trajectories (and model parameters) to the end of the time window from the data point
        
        T_prop=T-t_range[data_idx] #Time remaining in interval 'after' t_range[data_idx] (where DA performed)
        if (T_prop>0):
            t_range_prop=t_range[data_idx:]-t_range[data_idx]
            start = time.time()
            params_noises = np.random.uniform(-2./np.sqrt(iterN+1), 2./np.sqrt(iterN+1), n_samples)
            x_forward[:, data_idx: ,:] = ensemble_forward_model(G, ekf.damodel.q[-1] + params_noises.reshape(n_samples,1), ekf.damodel.x[-1], T_prop , dt_fsolve, t_range_prop, parallel_flag = parflag)
            end = time.time()
            print('Time elapsed for forward model (to end of window): ', end - start)
        else:
            print('no propagation required (to end of window)')
    
        #store pre data trajectory
        x_forward_all[:,(1+steps_T_init)+iterN*steps_T:(1+steps_T_init)+(iterN+1)*steps_T,:] = x_forward

        ## Output files 
        ## Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.damodel.q, open("data/u.pkl", "wb"))
        pickle.dump(ekf.damodel.x, open("data/g.pkl", "wb"))
        pickle.dump(ekf.damodel.error, open("data/error.pkl", "wb"))
        pickle.dump(x_forward_all, open("data/x.pkl", "wb"))


    #plots
    #t_range_total=np.hstack([t_range+steps_T*i for i in np.arange(steps_DA)])
    #if T_init>0.0:
    #    t_range_total=np.hstack([t_range_init,t_range_total])
    #else:
    #    t_range_total=np.hstack([0.0,t_range_total])

    #print(t_range_total)
    #plot_states(data[obs_times,:],x_forward_all,T_init+obs_times*dt,t_range_total,6,N,'forward_filter')

        
