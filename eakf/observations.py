
#This File has 3 sections:
# - Time observations  - what time(s) during DA window is a data collected
# - State observations - which portion of the state is observed when observation is carried out
# - Combined observations - this just combines the two to form the observation type.
#
# E.g if i would like to observe Infectiousness for all nodes at the first time in the window
#     I can create:
#         - "RandomStatusStateObservation" + "RandomTimeObservation" types
#     if i also want to observe 5% of complete states of nodes at the beginning of each window
#     I can create:
#         - "RandomStateObservation" (with obs_nodes=0.05*N input)+ "FixedTimeObservation" (with 0 input)
# Can create multiple combined observation types and put them in a list to feed into the data assimilation model

import numpy as np

### FRAMES FOR OBSERVATIONS ###
#Useless, but tells you how to build new ones: 
class TimeObservation:
    
    def __init__(self):
        pass

    #updates the observation model if required (here nothing changes)            
    def initialize_new_window(self,DAstep):    
        pass

class StateObservation:

    def __init__(self):
        pass

    #updates the observation model at new window            
    def initialize_new_window(self,DAstep):    
        pass
    
    #updates the observation model when taking observation if required
    def initialize_new_obs(self,x):
        pass

    #enforces statuses sum to 1 if required 
    def sum_to_one(self,x):
        pass

class Observation(TimeObservation,StateObservation):
    
    def __init__(self):
    
        TimeObservation.__init__(self)
        StateObservation.__init__(self)
        self.name=obs_name

    def initialize_new_window(self,DAstep):
        TimeObservation.initialize_new_window(self,DAstep)
        StateObservation.initialize_new_window(self,DAstep)

    def initialize_new_obs(self,x):
        StateObservation.initialize_new_obs(self,DAstep)

    def sum_to_one(self,x):
        StateObservation.sum_to_one(self,DAstep)


### --- Time Observations --- ###
            
#We observe at a fixed time in each DA window
class FixedTimeObservation(TimeObservation):

    def __init__(self,t_range,obs_idx):

        #The timesteps of a DA window
        self.t_range = t_range #[dt,2dt,....,T-dt,T]
        self.Tsize = t_range.size 

        #storage for time overall, and local time wrt window
        #note the index corresponds to the index in t_range
        self.obs_time_in_window=obs_idx
        self.obs_time= self.obs_time_in_window
        
    def initialize_new_window(self,DAstep):
        #time_in_window is fixed
        self.obs_time = DAstep*self.Tsize  + self.obs_time_in_window

### --- We observe at a random time in each DA window --- ###       
class RandomTimeObservation(TimeObservation):

    def __init__(self,t_range):

        #The timesteps of a DA window
        self.t_range = t_range #[dt,2dt,....,T-dt,T]
        self.Tsize = t_range.size 
        
        #storage for time overall, and local time wrt window
        self.obs_time_in_window=np.random.randint(self.Tsize)
        self.obs_time=self.obs_time_in_window
        
    def initialize_new_window(self,DAstep):
        #randomly generate new time
        self.obs_time_in_window=np.random.randint(self.Tsize)
        self.obs_time = DAstep*self.Tsize +self.obs_time_in_window

        
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

    #updates observation model (here generates a random set of nodes)
    def initialize_new_window(self,DAstep):
        #This if statement gives consistency of Random and Full State
        #otherwise the pseudorandom generator is used one extra time here
        if self.obs_nodes<self.N:
            onetoN=np.arange(self.N)
            np.random.shuffle(onetoN)#(in-place shuffle)
            tmp=np.array(sorted(onetoN[:self.obs_nodes]))
            self.obs_states=np.hstack([np.arange(self.status)+i*self.status for i in tmp])
            print(self.obs_states.shape)

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

    #updates observation model (here generates random set of nodes)
    def initialize_new_window(self,DAstep):
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

    #as we measure a subset of states, we may need to enforce other states to sum to one
    def sum_to_one(self,x):

        #CUMBERSOME... (10,6,N), sum over 2nd axis ignoring S
        #First enforce probabilities == 1, by placing excess in susceptible
        sumx=x.reshape(x.shape[0],self.status,self.N)
        sumx=np.sum(sumx[:,1:,:],axis=1)
        x[:,0:self.N]=1.0 - sumx

        #put any overflow (where sumx>1) from susceptible into exposed...
        xneg=np.minimum(x[:,0:self.N],0.0)
        x[:,0:self.N]=x[:,0:self.N]-xneg
        x[:,self.N:2*self.N]=x[:,self.N:2*self.N]-xneg

        #print("total mass put from susceptible => exposed: ", np.sum(np.sum(xneg,axis=1),axis=0))
   
#We observe a subset of nodes at a status, only if the state exceeds a given threshold value.
#e.g we have a probability of observing I_i if (I_i > 0.8) when the observation takes place.
class HighProbStatusStateObservation(StateObservation):
    def __init__(self,N,status,obs_frac,obs_status_idx,threshold,threshold_type):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status
        #array of status to observe
        self.obs_status_idx=obs_status_idx
        #The fraction of states
        self.obs_frac = np.clip(obs_frac,0.0,1.0)
        #The minimum threshold to be considered for test (e.g 0.7)
        self.obs_threshold = np.clip(threshold,0.0,1.0)
        #threshold flag
        self.obs_threshold_type = threshold_type
        #default init observation
        self.obs_states = np.arange(int(self.obs_frac*self.N)*obs_status_idx.size)
        
    #updates the observation model when taking observation
    def initialize_new_obs(self,x):
        #Candidates for observations are those with a required state >= threshold
        onetoN=np.arange(self.N)
        candidate_states= np.hstack([self.obs_status_idx+i*self.status for i in onetoN]) 
        if self.obs_threshold_type == 'any':
            #Case 1: candidate state if ANY ensemble member is > threshold
            candidate_states_ens=np.hstack([candidate_states[x[i,candidate_states]>=self.obs_threshold] for i in range(x.shape[0])])
            candidate_states_ens=np.unique(candidate_states_ens)
        elif self.obs_threshold_type == 'mean':
            #Case 2: candidate state if MEAN ensemble members is > threshold
            xmean = np.mean(x[:,candidate_states],axis=0)
            candidate_states_ens=candidate_states[xmean>=self.obs_threshold]
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
            self.obs_states=np.array([])
            print("no observation was above the threshold")

    #as we measure a subset of states, we may need to enforce other states to sum to one
    def sum_to_one(self,x):

        #First enforce probabilities == 1, by placing excess in susceptible and Exposed
        #split based on their current proportionality.
        #(Put all in S or E leads quickly to [0,1] bounding issues.
        #CUMBERSOME... (10,6,N), sum over 2nd axis ignoring S
        sumx=x.reshape(x.shape[0],self.status,self.N)
        sumx=np.sum(sumx[:,2:,:],axis=1) #sum over I H R D
        x1mass=np.sum(x[:,  0:self.N],axis=1)#mass in S
        x2mass=np.sum(x[:,  self.N:2*self.N],axis=1) #mass in E
        fracS=x1mass/(x1mass+x2mass)#get the proportion of mass in frac1
        fracE=1.0-fracS
        x[:,0:self.N]=((1.0-sumx).T*fracS).T #mult rows by fracS
        x[:,self.N:2*self.N]= ((1.0-sumx).T*(fracE)).T 


### --- Combined Observation --- ###

#Random Observation 
class RandomObservation(RandomTimeObservation,RandomStateObservation):

    def __init__(self,t_range,N,status,obs_nodes,obs_name):
    
        RandomTimeObservation.__init__(self,t_range)
        RandomStateObservation.__init__(self,N,status,obs_nodes)
        self.name=obs_name

    def initialize_new_window(self,DAstep):
        RandomTimeObservation.initialize_new_window(self,DAstep)
        RandomStateObservation.initialize_new_window(self,DAstep)

#Random Observation of a defined set of status's SEIHRD (by idx)
class RandomStatusObservation(RandomTimeObservation,RandomStatusStateObservation):

    def __init__(self,t_range,N,status,obs_frac,obs_status_idx,obs_name):

        if not isinstance(obs_status_idx,np.ndarray):#if it's a scalar, not array
            obs_status_idx=np.array(obs_status_idx)

        RandomTimeObservation.__init__(self,t_range)
        RandomStatusStateObservation.__init__(self,N,status,obs_frac,obs_status_idx)
        self.name=obs_name
        
    def initialize_new_window(self,DAstep):
        RandomTimeObservation.initialize_new_window(self,DAstep)
        RandomStatusStateObservation.initialize_new_window(self,DAstep)

    def sum_to_one(self,x):
        RandomStatusStateObservation.sum_to_one(self,x)


class HighProbRandomStatusObservation(RandomTimeObservation,HighProbStatusStateObservation):

    def __init__(self,t_range,N,status,obs_frac,obs_status_idx,threshold,threshold_type,obs_name):

        if not isinstance(obs_status_idx,np.ndarray):#if it's a scalar, not array
            obs_status_idx=np.array(obs_status_idx)

        RandomTimeObservation.__init__(self,t_range)
        HighProbStatusStateObservation.__init__(self,N,status,obs_frac,obs_status_idx,threshold,threshold_type)
        self.name=obs_name
        
    def initialize_new_window(self,DAstep):
        RandomTimeObservation.initialize_new_window(self,DAstep)
        HighProbStatusStateObservation.initialize_new_window(self,DAstep)

    def initialize_new_obs(self,x):
        HighProbStatusStateObservation.initialize_new_obs(self,x)


    def sum_to_one(self,x):
        HighProbStatusStateObservation.sum_to_one(self,x)
