import numpy as np

from epiforecast.ensemble_adjusted_kalman_filter import EnsembleAdjustedKalmanFilter

class DataAssimilator:

    def __init__(self, observations, errors, *, transition_rates_to_update_str=None, transmission_rate_to_update_flag=None):
        """
           A data assimilator, to perform updates of model parameters and states using an
           ensemble adjusted Kalman filter (EAKF) method. 
           
           Positional Args
           ---------------
           observations (list, [], or Observation): A list of Observations, or a single Observation.
                                                     Generates the indices and covariances of observations
        
           errors (list, [],  or Observation): Observation for the purpose of error checking. Error 
                                                observations are used to compute online differences at 
                                                the observed (according to Errors) between Kinetic and 
                                                Master Equation models
           Keyword Args
           ------------ 
           transition_rates_to_update_str (list): list of strings naming the transition_rates we would like
                                                  update with data. must coincide with naming found in
                                                  epiforecast/populations.py.
                                                  If not provided, will set [] 
                                                         
           transmission_rate_to_update_flag (boolean): bool to update transmission rate with data
                                                       If not provided will set False
           Methods
           -------
           
           update(ensemble_state,data,contact_network=[],full_ensemble_transition_rates, full_ensemble_transmission_rate):
                                                          Perform an update of the ensemble states `ensemble_state`, and if provided, ensemble
                                                          parameters `ensemble_transition rates`, `ensemble_transmission_rate` and the network
                                                          `contact network`. Returns the updated model parameters and updated states.

           make_new_observation(state): For every Observation model, update the list of indices at which to observe (given by omodel.obs_states).
                                        Returns a concatenated list of indices `observed_states` with duplicates removed.
           
           get_observation_cov(): For every Observation model, obtain the relevant variances when taking a measurement of data. Note we account for multiple node 
                                  measurements by using the minimum variance at that node (I.e if same node is queried twice in a time interval we take the most 
                                  accurate test). Returns a diagonal covariance matrix for the distinct observed states, with the minimum variances on the diagonal.
           
           sum_to_one(state): Takes the state `state` and enforces that all statuses at a node sum to one. Does this by distributing the mass (1-(I+H+R+D)) into S and E, where 
                          the mass is divided based on the previous state's relative mass in S and E. i.e Snew= S/(S+E)*(1-(I+H+R+D)), Enew = E/(S+E)*(1-(I+H+R+D)) 
           
           error_to_truth_state(state,data): updates emodel.obs_states and measures (user prescribed) differences between the data and state online.
                                             #Current implementation sums the difference in number of predicted states
                                             #and actual states in an given interval e.g 0.5 <= I <= 1.0
        """

        if not isinstance(observations,list):#if it's a scalar, not array
            observations=[observations]

        if not isinstance(errors,list):#if it's a scalar, not array
            observations=[errors]

        if transition_rates_to_update_str is None:
            transition_rates_to_update_str=[]
            
        if transmission_rate_to_update_flag is None:
            transmission_rate_to_update_flag = False 
            
        if not isinstance(transition_rates_to_update_str,list):#if it's a string, not array
            transition_rates_to_update_str = [transition_rates_to_update_str]
            
        #observation models(s)
        self.omodel = observations      
                
        #the data assimilation models (One for each observation model)
        self.damethod = EnsembleAdjustedKalmanFilter()

        #online evaluations of errors, one needs an observation class to check differences in data
        self.online_emodel= errors

        #which parameter to assimilate joint with the state
        self.transition_rates_to_update_str = transition_rates_to_update_str
        self.transmission_rate_to_update_flag= transmission_rate_to_update_flag

    def make_new_observation(self,ensemble_state,contact_network):
        for i in range(len(self.omodel)):
            self.omodel[i].make_new_obs(ensemble_state,contact_network)
        
        observed_states=np.hstack([self.omodel[i].obs_states for i in range(len(self.omodel))])
        observed_states=np.unique(observed_states)
        return observed_states


    def get_observation_cov(self):
        #for each observation we have a covariance
        covs=[ self.omodel[i].get_observational_cov() for i in range(len(self.omodel))]
        #we need to create a padded matrix of large numbers, then take min on the columns,
        #and remove any of the large numbers
        maxcov=np.max(np.hstack([self.omodel[i].obs_states for i in range(len(self.omodel))]))
        bigval=10**5
        pad_covs=(bigval)*np.ones([len(covs),maxcov+1],dtype=float)
        for i in range(len(covs)):
            idx=self.omodel[i].obs_states
            pad_covs[i,idx]=covs[i]
            
        #now we can take min down the columns to extract the smaller of non-distinct covariances
        pad_covs=np.min(pad_covs, axis=0)
        distinct_cov=pad_covs[pad_covs<bigval]

      
        #make into matrix
        distinct_cov=np.diag(distinct_cov)
        
        return distinct_cov
    # ensemble_state np.array([ensemble size, num status * num nodes]
    # data np.array([ensemble size, num status * num nodes])
    # contact network networkx.graph (if provided)
    # full_ensemble_transition_rates list[ensemble size] of  TransitionRates objects from epiforecast.populations 
    # full_ensemble_transmission_rate np.array([ensemble size])
    def update(self,
               ensemble_state,
               data,
               full_ensemble_transition_rates,
               full_ensemble_transmission_rate,
               contact_network=None):

        if len(self.omodel)>0:
            
            #extract the transition_rates to update
            if len(self.transition_rates_to_update_str)>0:
                ensemble_transition_rates=np.array([])
                for member in range(len(full_ensemble_transition_rates)):
                    #this returns an [ensemble size x transition rates (to be updated)] np.array
                    rates_tmp = np.hstack([full_ensemble_transition_rates[member].get_transition_rates_from_str(rate_type) \
                                                           for rate_type in self.transition_rates_to_update_str])

                    #have to create here as rates_tmp unknown in advance
                    if member == 0:
                      ensemble_transition_rates=np.empty((0,rates_tmp.size), dtype=float)
                      
                    ensemble_transition_rates = np.append(ensemble_transition_rates,[rates_tmp],axis=0)

                ensemble_transition_rates = np.vstack(ensemble_transition_rates)
            else:
                ensemble_transition_rates = []    
                
                
            if self.transmission_rate_to_update_flag is True:
                ensemble_transmission_rate = full_ensemble_transmission_rate
            else:
                ensemble_transmission_rate = [] 
                
            om = self.omodel
            dam = self.damethod

            obs_states = self.make_new_observation(ensemble_state,contact_network) #Generate states to observe
      
            if (obs_states.size>0):
                
                print("partial states to be assimilated", obs_states.size)
                #get the truth indices, for the observation(s)
                truth = data[obs_states]
                
                #get the covariances for the observation(s), with the minimum returned if two overlap
                cov = self.get_observation_cov()
                print(cov)
                
                #perform da model update with ensemble_state: states,transition and transmission rates
                ensemble_state[:,obs_states], new_ensemble_transition_rates, new_ensemble_transmission_rate = dam.update(ensemble_state[:,obs_states],
                                                                                                                          ensemble_transition_rates,
                                                                                                                          ensemble_transmission_rate,
                                                                                                                          truth,cov)
                
                #update the new transition rates if required
                if len(self.transition_rates_to_update_str)>0:
                    for member in range(len(full_ensemble_transition_rates)):
                        new_member_rates = new_ensemble_transition_rates[member,:]
                        for rate_type in self.transition_rates_to_update_str:
                            #need to go back from numpy array to setting rates
                            #we obtain the size, then update the corresponding transition rate
                            #then delete this an move onto the next rate
                            
                            rate_size = full_ensemble_transition_rates[member].get_transition_rates_from_str(rate_type).size
                            full_ensemble_transition_rates[member].set_transition_rates_from_str(rate_type,new_member_rates[:rate_size])
                            new_member_rates = np.delete(new_member_rates, np.arange(rate_size))
                                                                                                           
                #update the transmission_rate if required
                if self.transmission_rate_to_update_flag is True:
                    full_ensemble_transmission_rate=new_ensemble_transmission_rate
                
                #Force probabilities to sum to one
                self.sum_to_one(ensemble_state)
            
                print("EAKF error:", dam.error[-1])
            else:
                print("no assimilation required")

            #Error to truth
            if len(self.online_emodel)>0:
                self.error_to_truth_state(ensemble_state,data)
        
            #return ensemble_state and model params
            return ensemble_state, full_ensemble_transition_rates, full_ensemble_transmission_rate

        else: #if no observations performed
            return ensemble_state, full_ensemble_transition_rates, full_ensemble_transmission_rate

            
    #defines a method to take a difference to the data state
    def error_to_truth_state(self,ensemble_state,data):
        
        em=self.online_emodel #get corresponding error model
        #Make sure you have a deterministic ERROR model - or it will not match truth
        #without further seeding
        em.make_new_obs(ensemble_state)
        predicted_infected = em.obs_states
        truth=data[np.arange(em.n_status*em.N)]

        #print(truth[predicted_infected])
        em.make_new_obs(truth[np.newaxis,:])
        
        #take error
        actual_infected= em.obs_states
        #print(truth[actual_infected])
        different_states=np.zeros(em.n_status*em.N)
        different_states[predicted_infected]=1.0
        different_states[actual_infected] = different_states[actual_infected]-1.0
        number_different=np.maximum(different_states,0.0)-np.minimum(different_states,0.0)
        print("Differences between predicted and true I>0.5:",np.sum(number_different).astype(int))


    #as we measure a subset of states, we may need to enforce other states to sum to one
    
    def sum_to_one(self,ensemble_state):
        N=self.omodel[0].N
        n_status=self.omodel[0].n_status
        if n_status == 6:
            #First enforce probabilities == 1, by placing excess in susceptible and Exposed
            #split based on their current proportionality.
            #(Put all in S or E leads quickly to [0,1] bounding issues.
            sumx=ensemble_state.reshape(ensemble_state.shape[0],n_status,N)
            sumx=np.sum(sumx[:,2:,:],axis=1) #sum over I H R D
            x1mass=np.sum(ensemble_state[:,0:N],axis=1)#mass in S
            x2mass=np.sum(ensemble_state[:,N:2*N],axis=1) #mass in E
            fracS=x1mass/(x1mass+x2mass)#get the proportion of mass in frac1
            fracE=1.0-fracS
            ensemble_state[:,0:N]=((1.0-sumx).T*fracS).T #mult rows by fracS
            ensemble_state[:,N:2*N]= ((1.0-sumx).T*(fracE)).T 
        elif n_status==5:
            #All mass automatically lumped into empty field: E
            #If this requires smoothing - input here
            pass
