
#An Observation has 3 sections:
# - State observations - which portion of the state is observed when observation is carried out
# - Observation noise - what sort of noise distribution is there on the observation
# - Combined observations - this just combines the two to form the observation type

#parent containers at the top, the implemented children are below
# StateObservation children:
# - FullStateObservation: Observation of the entire state
# - RandomStateObservation: Observation of all statuses for random node subset
# - RandomStatusStateObservation: Observation of specified statuses for random node subset
# - HighProbRandomStatusObservation: Observation of states with a specfied probability interval
# ObservationNoise children:
# - IndependentGaussian: Adds independent Gaussian noise at fixed level to all observations

import numpy as np

### FRAMES FOR OBSERVATIONS ###

class StateObservation:

    def __init__(self):
        self.obs_states = np.array([])
        pass

    
    #updates the observation model when taking observation
    def make_new_obs(self,x):
        pass
    
#type of noise added to observation
class ObservationNoise:

    def __init__(self):
        pass

    def get_observational_cov(self,obs_states):
        cov=np.zeros(obs_states.size)
        return cov

#combine them    
class Observation(StateObservation,ObservationNoise):
    
    def __init__(self):
    
        StateObservation.__init__(self)
        ObservationNoise.__init__(self)
        self.name=obs_name

    def make_new_obs(self,x):
        StateObservation.make_new_obs(self,x)

    def get_observational_cov(self):
        cov=ObservationNoise.get_observational_cov(self,StateObservation.obs_states)
        return cov


    
### Implemented State Observations ###
    
#We observe the entire system during observation
class FullStateObservation(StateObservation):
    
    def __init__(self,N,status):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status

        #The states for observation
        self.obs_states = np.arange(status*N)
    
#We observe a random subset of nodes during observation
class RandomStateObservation(StateObservation):
    
    def __init__(self,N,status,obs_nodes):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status

        #The states for observation
        self.obs_nodes = obs_nodes
        #default init observation
        self.obs_states = np.arange(obs_nodes*status)

    #updates the observation model when taking observation
    def make_new_obs(self,x):
        #This if statement gives consistency of Random and Full State
        #otherwise the pseudorandom generator is used one extra time here
        if self.obs_nodes<self.N:
            onetoN=np.arange(self.N)
            np.random.shuffle(onetoN)#(in-place shuffle)
            tmp=np.array(sorted(onetoN[:self.obs_nodes]))
            self.obs_states=np.hstack([np.arange(self.status)+i*self.status for i in tmp])
            
#We observe a random subset of nodes at a particular status (S,E,I,H,R,D)
class RandomStatusStateObservation(StateObservation):
    def __init__(self,N,status,obs_frac,obs_status_idx):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status
        #array of status to observe
        self.obs_status_idx=obs_status_idx
        #The states for observation
        self.obs_nodes = int(self.N*obs_frac)
        #default init observation
        self.obs_states = np.arange(self.obs_nodes*obs_status_idx.size)

    #updates the observation model when taking observation
    def make_new_obs(self,x):
        #This if statement gives consistency of Random and Full State
        #otherwise the pseudorandom generator is used one extra time here
        if self.obs_nodes<self.N:
            onetoN=np.arange(self.N)
            np.random.shuffle(onetoN)#(in-place shuffle)
            tmp=np.array(sorted(onetoN[:self.obs_nodes]))
            self.obs_states=np.hstack([self.obs_status_idx+i*self.status for i in tmp])
        else:
            onetoN=np.arange(self.N)
            self.obs_states=np.hstack([self.obs_status_idx+i*self.status for i in onetoN])
        

   
#We observe a subset of nodes at a status, only if the state exceeds a given threshold value.
#e.g we have a probability of observing I_i if (I_i > 0.8) when the observation takes place.
class HighProbStatusStateObservation(StateObservation):
    def __init__(self,N,status,obs_frac,obs_status_idx,min_threshold,max_threshold,threshold_type):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status
        #array of status to observe
        self.obs_status_idx=obs_status_idx
        #The fraction of states
        self.obs_frac = np.clip(obs_frac,0.0,1.0)
        #The minimum threshold (in [0,1]) to be considered for test (e.g 0.7)
        self.obs_min_threshold = np.clip(min_threshold,0.0,1.0)
        self.obs_max_threshold = np.clip(max_threshold,0.0,1.0)
        #threshold flag
        self.obs_threshold_type = threshold_type
        #default init observation
        self.obs_states = np.arange(int(self.obs_frac*self.N)*obs_status_idx.size)
        
    #updates the observation model when taking observation
    def make_new_obs(self,x):
        #Candidates for observations are those with a required state >= threshold
        onetoN=np.arange(self.N)
        candidate_states= np.hstack([self.obs_status_idx+i*self.status for i in onetoN]) 
        if self.obs_threshold_type == 'any':
            #Case 1: candidate state if ANY ensemble member is > threshold
            candidate_states_ens=np.hstack([candidate_states[(x[i,candidate_states]>=self.obs_min_threshold) & \
                                                             (x[i,candidate_states]<=self.obs_max_threshold)] \
                                            for i in range(x.shape[0])])
            candidate_states_ens=np.unique(candidate_states_ens)
        elif self.obs_threshold_type == 'mean':
            #Case 2: candidate state if MEAN ensemble members is > threshold
            xmean = np.mean(x[:,candidate_states],axis=0)
            candidate_states_ens=candidate_states[(xmean>=self.obs_min_threshold) & \
                                                  (xmean<=self.obs_max_threshold)]
        else:
            print('threshold type not recognised for observation, choose mean or any')
            exit()
        M=candidate_states_ens.size     
        if (int(self.obs_frac*M)>=1)&(self.obs_frac < 1.0) :
            onetoM=np.arange(M)
            np.random.shuffle(onetoM)#(in-place shuffle)
            tmp=np.array(sorted(onetoM[:int(self.obs_frac*M)]))
            self.obs_states=candidate_states_ens[tmp]
        elif (self.obs_frac == 1.0):
            self.obs_states=candidate_states_ens
        else: #The value is too small
            self.obs_states=np.array([],dtype=int)
            print("no observation was above the threshold")
        
        

### Implemented Observation Noise ###

#Independent Gaussian - same variance for all observations
class IndependentGaussian(ObservationNoise):

    def __init__(self,variance):
        self.variance=variance

    #returns a vector
    def get_observational_cov(self,obs_states):
        cov=self.variance*np.ones(obs_states.shape[0])
        return cov

        
### Implemented Combined Observations ###

#Random Observation 
class RandomObservation(RandomStateObservation,IndependentGaussian):

    def __init__(self,N,status,obs_nodes,obs_name,noise_var):
    
        RandomStateObservation.__init__(self,N,status,obs_nodes)
        IndependentGaussian.__init__(self,noise_var)
        self.name=obs_name

    def make_new_obs(self,x):
        RandomStateObservation.make_new_obs(self,x)

    def get_observational_cov(self):
        cov=IndependentGaussian.get_observational_cov(self,RandomStateObservation.obs_states)
        return cov
    
#Random Observation of a defined set of status's SEIHRD (by idx)
class RandomStatusObservation(RandomStatusStateObservation,IndependentGaussian):

    def __init__(self,N,status,obs_frac,obs_status_idx,obs_name,noise_var):

        if not isinstance(obs_status_idx,np.ndarray):#if it's a scalar, not array
            obs_status_idx=np.array(obs_status_idx)
            
        RandomStatusStateObservation.__init__(self,N,status,obs_frac,obs_status_idx)
        IndependentGaussian.__init__(self,noise_var)
        self.name=obs_name

    def make_new_obs(self,x):
        RandomStatusStateObservation.make_new_obs(self,x)
        
    def get_observational_cov(self):
        cov=IndependentGaussian.get_observational_cov(self,self.obs_states)
        return cov


class HighProbRandomStatusObservation(HighProbStatusStateObservation,IndependentGaussian):

    def __init__(self,N,status,obs_frac,obs_status_idx,min_threshold,max_threshold,threshold_type,obs_name,noise_var):
        if not isinstance(obs_status_idx,np.ndarray):#if it's a scalar, not array
            obs_status_idx=np.array(obs_status_idx)

        HighProbStatusStateObservation.__init__(self,N,status,obs_frac,obs_status_idx,min_threshold,max_threshold,threshold_type)
        IndependentGaussian.__init__(self,noise_var)
        self.name=obs_name
        
    def make_new_obs(self,x):
        HighProbStatusStateObservation.make_new_obs(self,x)
              
    def get_observational_cov(self):
        cov=IndependentGaussian.get_observational_cov(self,self.obs_states)
        return cov

    
