
#This File has 3 sections:
# - State observations - which portion of the state is observed when observation is carried out
# - Observation noise - what sort of noise distribution is there on the observation
# - Combined observations - this just combines the two to form the observation type.

# Can create multiple combined observation types and put them in a list to feed into the data assimilation model
# # - Time observations  - what time(s) during DA window is a data collected



import numpy as np

### FRAMES FOR OBSERVATIONS ###
# tells you how to build new ones: 

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

#Independent Gaussian
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

    


#Taking multiple different observations        
# class MultiObservation:
    
#     def __init__(self,ObservationArray):

#         if not isinstance(obs_status_idx,list):#if it's a scalar, not array
#             [ObservationArray]
       
#         for i in np.arange(len(ObservationArray)):
#             ObservationArray[i].__init__(self)

#         self.obs_array=ObservationArray
#         self.name=obs_name
        
        
#     def make_new_obs(self,x):
#         observation_states=np.empty(0)
#         for i in np.arange(self.obs_array):
#             self.obs_array.make_new_obs(self,x)
#             observation_states=self.obs_array.obs_states
        
            
#     def sum_to_one(self,x):
#         Observation[0].sum_to_one(self)

#     def get_observational_cov(self,x):
#         cov=[ np.array(ObservationNoise[i].get_observational_cov(self,x)) for i in np.arange(ObservationArray.size)
#         return cov
    








# class Observation(TimeObservation,StateObservation,ObservationNoise):
    
#     def __init__(self):
    
#         TimeObservation.__init__(self)
#         StateObservation.__init__(self)
#         ObservationNoise.__init__(self)
#         self.name=obs_name

#     def initialize_new_window(self,DAstep):
#         TimeObservation.initialize_new_window(self,DAstep)
#         StateObservation.initialize_new_window(self,DAstep)

#     def make_new_obs(self,x):
#         StateObservation.make_new_obs(self,x)

#     def sum_to_one(self,x):
#         StateObservation.sum_to_one(self,DAstep)

#     def get_observational_cov(self,x):
#         cov=ObservationNoise.get_observational_cov(self,x)
#         return cov

# ### --- Time Observations --- ###
            
# #We observe at a fixed time in each DA window
# class FixedTimeObservation(TimeObservation):

#     def __init__(self,t_range,obs_idx):

#         #The timesteps of a DA window
#         self.t_range = t_range #[dt,2dt,....,T-dt,T]
#         self.Tsize = t_range.size 

#         #storage for time overall, and local time wrt window
#         #note the index corresponds to the index in t_range
#         self.obs_time_in_window=obs_idx
#         self.obs_time= self.obs_time_in_window
        
#     def initialize_new_window(self,DAstep):
#         #time_in_window is fixed
#         self.obs_time = DAstep*self.Tsize  + self.obs_time_in_window

# ### --- We observe at a random time in each DA window --- ###       
# class RandomTimeObservation(TimeObservation):

#     def __init__(self,t_range):

#         #The timesteps of a DA window
#         self.t_range = t_range #[dt,2dt,....,T-dt,T]
#         self.Tsize = t_range.size 
        
#         #storage for time overall, and local time wrt window
#         self.obs_time_in_window=np.random.randint(self.Tsize)
#         self.obs_time=self.obs_time_in_window
        
#     def initialize_new_window(self,DAstep):
#         #randomly generate new time
#         self.obs_time_in_window=np.random.randint(0,self.Tsize-1)
#         #as t_range does not include "0" we must +1 here
#         self.obs_time = DAstep*self.Tsize +self.obs_time_in_window+1

        
    
