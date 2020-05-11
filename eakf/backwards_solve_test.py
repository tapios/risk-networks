import EoN
import numpy as np
import multiprocessing
from models import MasterEqn 
import networkx as nx
from eakf import EAKF
from DA_forward_plot import plot_states
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib

def forward_model(G, params, state0, T, dt_max, t_range):
    model = MasterEqn(G)
    # Dimension of states: num_status * num_nodes, num_time_steps
    states = model.solve(params, state0, T, dt_max, t_range)
    return states.T

def backward_model(G, params, stateT0, T0, T, dt_max, t_range):
    model = MasterEqn(G)
    # Dimension of states: num_status * num_nodes, num_time_steps
    states = model.backwards_solve(params, stateT0, T0, T, dt_max, t_range)
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
    infected = np.random.randint(N, size = 3)
    E, I, H, R, D = np.zeros([5, N])
    S = 0.99*np.ones(N,)
    I = 0.01*np.ones(N,)
    I[infected] = 0.99
    S[infected] = 0.01
    state0 = np.hstack((S, E, I, H, R, D))
    return state0

if __name__ == "__main__":

    np.random.seed(10)
    print("Number of cpu : ", multiprocessing.cpu_count())

    # Ensemble size (required>=2)
    n_samples = 10
    # Number of status for each node
    n_status = 6

    # Load network
    G, N = load_G()

    # Set prior for unknown parameters
    params = np.zeros([n_samples,1])
    params[:,0] = np.random.uniform(np.log(0.01), np.log(0.5), n_samples)
    
    # Set initial states
    states_IC = np.zeros([n_samples, n_status*N])
    for iterN in range(n_samples):
        states_IC[iterN, :] = random_IC()

    # Set time informations inside an EAKF step
    T = 80.
    dt = 0.1 #timestep for OUTPUT not solver
    steps_T = int(T/dt)
    t_range = np.linspace(0.0, T, num=steps_T+1, endpoint=True)#includes 0 and T
    dt_fsolve =0.1
    dt_bsolve =0.1
    Tinit=0.0
    # Container for forward model evaluations
    x_forward_all = np.empty([n_samples, steps_T, n_status*N])

    ## Forward model evaluation of all ensemble members
    start = time.time()

    # A simple decayed noise term to add into paramters during EAKF
    #params_noise = np.random.uniform(-2./np.sqrt(0+1), 2./np.sqrt(0+1), n_samples) 
    params_noise = np.zeros([n_samples,1])
    print('beta parameter for each ensemble: ')
    print(np.exp(params))
    x_forward = ensemble_forward_model(G, params + params_noise.reshape(n_samples,1), states_IC,T, dt_fsolve, t_range, parallel_flag = False)
    end = time.time() 
    
    print('Time elapsed for forward model: ', end - start)
    print('Difference to ICs at time T: ', np.sqrt(np.sum(np.square(x_forward[:,-1,:]-states_IC))))

    ## Backward model evaluation of all ensemble members
    start = time.time()
    x_backward = ensemble_backward_model(G, params + params_noise.reshape(n_samples,1), x_forward[:,-1,:],T, Tinit, dt_bsolve, t_range, parallel_flag = False)
    
    for ss in  range(n_samples):
        for tt in range(t_range.shape[0]):
            for uu in range(10):
                if tt == t_range.shape[0]-1:
                    print(x_forward[ss,tt,uu::N])
                    print(x_backward[ss,-(tt+1),uu::N])
    end = time.time()


    
    print('Time elapsed for backward model: ', end - start)
    xdiff=np.zeros([t_range.shape[0],n_samples])
    for tt in range(t_range.shape[0]):
        for ss in range(n_samples):
            xdiff[tt,ss]=np.sqrt(np.sum(np.square(x_forward[ss,tt,:]-x_backward[ss,-(tt+1),:])))

    print('Average difference in trajectory over times ')
    print(t_range)
    print('Given by: ')
    print(np.sum(xdiff,axis=1)/n_samples) 
    print('Average difference to initial conditions and time 0:' ,np.sqrt(np.sum(np.square(x_backward[:,-1,:]-states_IC)))/n_samples)

        
     

    #plot difference
    plt.plot(t_range,xdiff,'k',alpha=0.3)
    plt.plot(t_range,(np.sum(xdiff,axis=1)/n_samples),'r',label='ens average')
    plt.legend()
    plt.title('Error in ensemble trajectory forward/backward model')
    plt.show()

    plt.plot(t_range[100:],xdiff[100:,:],'k',alpha=0.3)
    plt.plot(t_range[100:],(np.sum(xdiff[100:,:],axis=1)/n_samples),'r',label='ens average')
    plt.legend()
    plt.title('Error in ensemble trajectory forward/backward model')
    plt.show()

    
    #plot states
    plot_states(x_forward[1,:,:],x_forward,t_range,t_range,6,N,'forward')
    x_backward_flip = np.flip(x_backward,axis=1)
    plot_states(x_backward_flip[1,:,:],x_backward_flip,t_range,t_range,6,N,'backward')

   
