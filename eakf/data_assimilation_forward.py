import numpy as np
from eakf_multiobs import EAKF

#This class holds the data model, the observation model(s), the DA for each observation type
#the update method applies Per Window.
#it will find all data from window, put them in forwards chronological order, and filter them:
#i.e DA on first point, update ensemble, run to 2nd point, DA on 2nd point, etc.
class ForwardAssimilator:

    def __init__(self,parameters,states,Observations,Errors):

        if not isinstance(Observations,list):#if it's a scalar, not array
            Observations=[Observations]

        if not isinstance(Errors,list):#if it's a scalar, not array
            Errors=[Errors]
       
        #observation models(s)
        self.omodel = Observations      
        
        #store the parameters
        self.params=parameters[np.newaxis]
        
        obs_states=[states[:,self.omodel[i].obs_states] for i in range(len(self.omodel))]
        #the data assimilation models (One for each observation model)
        self.damethod = EAKF(obs_states[0])

        #online evaluations of errors, one needs an observation class to check differences in data
        self.online_emodel= Errors


    def make_new_observation(self,x):
        for i in range(len(self.omodel)):
            self.omodel[i].make_new_obs(x)
        
        observed_states=np.hstack([self.omodel[i].obs_states for i in range(len(self.omodel))])
        observed_states=np.unique(observed_states)
        return observed_states


    def get_observation_cov(self):
        #for each observation we have a covariance
        covs=[ self.omodel[i].get_observational_cov() for i in range(len(self.omodel))]
        #we need to create a padded matrix of large numbers, then take min on the columns,
        #and remove any of the large numbers
        maxcov=np.max(np.hstack([self.omodel[i].obs_states for i in range(len(self.omodel))]))
        Bigval=10**5
        pad_covs=(Bigval)*np.ones([len(covs),maxcov+1],dtype=float)
        for i in range(len(covs)):
            idx=self.omodel[i].obs_states
            pad_covs[i,idx]=covs[i]
            
        #now we can take min down the columns to extract the smaller of non-distinct covariances
        pad_covs=np.min(pad_covs, axis=0)
        distinct_cov=pad_covs[pad_covs<Bigval]

      
        #make into matrix
        distinct_cov=np.diag(distinct_cov)
        
        return distinct_cov
    
    def update(self,x,local_time,data,global_time):

        om=self.omodel
        dam=self.damethod
      
        #Restrict x to the the observation time
        x=x[:,local_time,:]
          
        obs_states=self.make_new_observation(x) #Generate states to observe at observation time 
      
        if (obs_states.size>0):

            print("partial states to be assimilated", obs_states.size)

            #get the truth indices, for the observation(s)
            truth = data.make_observation(global_time,obs_states)
            #get the covariances for the observation(s), with the minimum returned if two overlap
            cov = self.get_observation_cov()
            
            #perform da model update with x: states,q parameters.
            q,x[:,obs_states]=dam.update(x[:,obs_states],self.params[-1],truth,cov)
            self.params=np.append(self.params, [q], axis=0)

            #Force probabilities to sum to one
            #self.sum_to_one(x)
            
            print("EAKF error:", dam.error[-1])
        else:
            print("no assimilation required")

        #Error to truth
        self.error_to_truth_state(x,local_time,data,global_time)
        
        #return x at the observation time
        return self.params[-1],x
        
    #defines a method to take a difference to the data state
    def error_to_truth_state(self,state,local_time,data,global_time):

        pt=0
        em=self.online_emodel[pt] #get corresponding error model
        #Make sure you have a deterministic ERROR model - or it will not match truth
        #without further seeding
        em.make_new_obs(state)
        predicted_infected = em.obs_states
        truth=data.make_observation(global_time,np.arange(em.status*em.N))
        #print(truth[predicted_infected])
        em.make_new_obs(truth[np.newaxis,:])
        
        #take error
        actual_infected= em.obs_states
        #print(truth[actual_infected])
        different_states=np.zeros(em.status*em.N)
        different_states[predicted_infected]=1.0
        different_states[actual_infected] = different_states[actual_infected]-1.0
        number_different=np.maximum(different_states,0.0)-np.minimum(different_states,0.0)
        print("Differences between predicted and true I>0.5:",np.sum(number_different).astype(int))


    #as we measure a subset of states, we may need to enforce other states to sum to one
    def sum_to_one(self,x):
        N=self.omodel[0].N
        status=self.omodel[0].status
        #First enforce probabilities == 1, by placing excess in susceptible and Exposed
        #split based on their current proportionality.
        #(Put all in S or E leads quickly to [0,1] bounding issues.
        sumx=x.reshape(x.shape[0],status,N)
        sumx=np.sum(sumx[:,2:,:],axis=1) #sum over I H R D
        x1mass=np.sum(x[:,0:N],axis=1)#mass in S
        x2mass=np.sum(x[:,N:2*N],axis=1) #mass in E
        fracS=x1mass/(x1mass+x2mass)#get the proportion of mass in frac1
        fracE=1.0-fracS
        x[:,0:N]=((1.0-sumx).T*fracS).T #mult rows by fracS
        x[:,N:2*N]= ((1.0-sumx).T*(fracE)).T 
