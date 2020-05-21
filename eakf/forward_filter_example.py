import EoN
import numpy as np
import multiprocessing
from models import MasterEqn 
import networkx as nx
from data import HighSchoolData
from observations import RandomStatusObservation, HighProbRandomStatusObservation
from data_assimilation_forward import ForwardAssimilator
import time
import pickle
from DA_forward_plot import plot_states

def forward_model(G, params, state0, T, dt_max, t_range):
    model = MasterEqn(G)
    # Dimension of states: num_status * num_nodes, num_time_steps
    states = model.solve(params, state0, T, dt_max, t_range)
    return states.T

def ensemble_forward_model(G, qi, xi, T, dt_max, t_range, parallel_flag = True):
    if not isinstance(t_range,np.ndarray):
        t_range=np.array([t_range])    
     
    if parallel_flag == True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = []
        for iterN in range(qi.shape[0]):
            results.append(pool.apply_async(forward_model, (G, qi[iterN,:], xi[iterN,:], T, dt_max, t_range)))
        iterN = 0
        states_all = np.zeros([qi.shape[0], t_range.size, xi.shape[1]])
        for result in results:
            states_all[iterN,:,:] = result.get()
            iterN = iterN + 1
        pool.close()
    else:
        states_all = np.zeros([qi.shape[0], t_range.size, xi.shape[1]])
        for iterN in range(qi.shape[0]):
            states_all[iterN,:,:] = forward_model(G, qi[iterN,:], xi[iterN,:], T, dt_max, t_range)
    if t_range.size==1:
            states_all=states_all[:,0,:] #need this otherwise complaint when t_range size 1

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
    Step 0a:  Run forward solver for ensemble of solutions U(t) for some initial time Tinit = T_0
    Step 0b:  Construct the assimilator, Observations, Error model and data using modular structures.
    ***ENTER LOOP k=0,1,2,... K***
    ***ENTER LOOP i=0,1,2,... I***
    Step 1:   Run forward solver for solution U(t) over static network time window T_{k,i} -> T_{k,i+1}
    Step 2:   Assimilate data obtained from obsevation classes, as a filter at T_{k,i+1} with an EAKF update
    Step -:   return to Step 1 i->i+1
    **EXIT LOOP i=0,1,2,... I***
    Step 3:   Add all trajectories into storage, move to next window from Step 1 k -> k+1
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
    #NB we have 100 pieces of data, INCLUDING T=0, so only 99 steps
    steps_DA = 41
    # Ensemble size (required>=2)
    n_samples = 50

    
    #######################
    ### --- Step 0a --- ###
    #######################
    # Set prior for unknown parameters
    params = np.zeros([n_samples,1])
    params[:,0] = np.random.uniform(np.log(0.01), np.log(0.2), n_samples)

    
    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    for iterN in range(n_samples):
        states_IC[iterN, :] = random_IC()


    T_init = 0.0
    dt_init = 1.0 #must be 1.0 for high school data set
    steps_T_init=int(T_init/dt_init)
    #t_range_init = np.linspace(0.0, T_init, num=steps_T_init+1, endpoint=True)#includes 0 and T_init
    t_range_init=np.flip(np.arange(T_init,0.0,-dt_init)) # [dt,2dt,3dt,...,T-dt,T] (Excludes '0')
    dt_fsolve =1.0
 
    # Parameters for each EAKF step
    T = 1.0
    dt = 1.0 #timestep for OUTPUT not solver
    steps_T = int(T/dt)
    t_range=np.flip(np.arange(T,0.0,-dt)) # [dt,2dt,3dt,...,T-dt,T] (Excludes '0')
    dt_fsolve =1.0

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
        print('Time elapsed for initialization forward model run: ', end - start)

    #######################
    ### --- Step 0b --- ###
    #######################
    # Construct DA Models 

    # Data
    data=HighSchoolData(steps_T_init)

    #Observations 
    SmartInfectiousObs = HighProbRandomStatusObservation(N,n_status,0.1,[2],0.25,0.75,'mean','0.25<=0.1_SmartInfected<=0.75',0.01)
    DumbInfectiousObs = HighProbRandomStatusObservation(N,n_status,0.5,[2],0.25,0.75,'mean','0.25<=0.5_DumbInfected<=0.75',0.1)
    SmartDeceasedObs= HighProbRandomStatusObservation(N,n_status,0.5,[2],0.5,0.75,'mean','0.5<=0.5_SmartDecesed<=0.75',0.01)

    OModel=[SmartInfectiousObs,DumbInfectiousObs,SmartDeceasedObs]
    #OModel=SmartInfectiousObs
    
    #EModel - error model, one, for each observation mode to check effectiveness of our observations
    EModel = HighProbRandomStatusObservation(N,n_status,1.0,[2],0.5,1.0,'mean','All_Infected>=0.5',0.0)   

    #Build the DA
    ekf=ForwardAssimilator(params,x_forward_init[:,-1,:],OModel,EModel)

    print("Assimilating data from:")
    for i in range(len(ekf.omodel)):
        print(ekf.omodel[i].name)

    new_params=params[:]
    ### ***ENTER LOOP*** ### 
    for DAstep in range(steps_DA):

        x_forward=np.zeros([n_samples,t_range.size,n_status*N])
        
        for idx_tt,tt in enumerate(t_range):
            print('DA step: ', DAstep+1)

            ######################
            ### --- Step 1 --- ###
            ######################

            idx_global=(steps_T_init+1) + DAstep*steps_T + idx_tt
            tt_global=T_init+DAstep*T+tt

            ## Forward model evaluation of all ensemble members from the start of the window
            start = time.time()
            # A simple decayed noise term to add into paramters during EAKF
            params_noises = np.random.uniform(-5./np.sqrt(DAstep+1), 5./np.sqrt(DAstep+1), n_samples)        
            if idx_tt==0:
                x_forward[:,idx_tt,:] = ensemble_forward_model(G, new_params + params_noises.reshape(n_samples,1), x_forward_all[:,steps_T_init+DAstep*steps_T,:], dt, dt_fsolve, dt , parallel_flag = parflag)
            else:
                x_forward[:,idx_tt,:] = ensemble_forward_model(G, new_params + params_noises.reshape(n_samples,1), x_forward[:,idx_tt-1,:], dt, dt_fsolve, dt , parallel_flag = parflag)
            end = time.time()
            print('Time elapsed for forward model prediction run: ', end - start)


            ######################
            ### --- Step 2 --- ###
            ######################

            ## EAKF to update joint states
            start = time.time()
            #Assimilate the point
            new_params,x_forward[:,idx_tt,:]=ekf.update(x_forward,idx_tt,data,idx_global)
            end = time.time()
            print("average Beta (true=0.04)", np.exp(np.mean(new_params)))
            print('Assimilation time: ', tt_global,', Time elapsed for EAKF: ', end - start)


        ######################
        ### --- Step 3 --- ###
        ######################

        #store pre data trajectory
        x_forward_all[:,(1+steps_T_init)+DAstep*steps_T:(1+steps_T_init)+(DAstep+1)*steps_T,:] = x_forward

        ## Output files 
        ## Overwrite for each EAKF step to facilitate debugging
        pickle.dump(ekf.params, open("data/u.pkl", "wb"))
        #pickle.dump(ekf.damethod.x, open("data/g.pkl", "wb"))#note only includes observed states
        pickle.dump(ekf.damethod.error, open("data/error.pkl", "wb"))
        pickle.dump(x_forward_all, open("data/x.pkl", "wb"))


    #plots
    print("for plots, now use DA_forward_plot.py")
