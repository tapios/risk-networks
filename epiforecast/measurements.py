import numpy as np

class TestMeasurement:
    def __init__(self,
                 status,
                 sensitivity = 0.80,
                 specificity = 0.99,
                 reduced_system=True,
              ):

        self.sensitivity = sensitivity
        self.specificity = specificity

        if reduced_system == True:
            self.status_catalog = dict(zip(['S', 'I', 'H', 'R', 'D'], np.arange(5)))
        else :
            self.status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))

        self.status = status
        self.n_status = len(self.status_catalog.keys())

    def _set_prevalence(self, ensemble_states):
        """
        Inputs:
        -------
            ensemble_states : `np.array` of shape (ensemble_size, num_status * population) at a given time
            status_idx      : status id of interest. Following the ordering of the reduced system SIRHD.
        """

        population      = ensemble_states.shape[1]/self.n_status
        ensemble_size   = ensemble_states.shape[0]
        self.prevalence = ensemble_states.reshape(ensemble_size,self.n_status,-1)[:,self.status_catalog[self.status],:].sum(axis = 1)/population

    def _set_ppv(self, scale = 'log'):
        PPV = self.sensitivity * self.prevalence / \
             (self.sensitivity * self.prevalence + (1 - self.specificity) * (1 - self.prevalence))

        FOR = (1 - self.sensitivity) * self.prevalence / \
             ((1 - self.sensitivity) * self.prevalence + self.specificity * (1 - self.prevalence))

        if scale == 'log':
            logit_ppv  = np.log(PPV/(1 - PPV))
            logit_for  = np.log(FOR/(1 - FOR))

            self.logit_ppv_mean = logit_ppv.mean()
            self.logit_ppv_var  = logit_ppv.var()

            self.logit_for_mean = logit_for.mean()
            self.logit_for_var  = logit_for.var()

        else:
            self.ppv_mean = PPV.mean()
            self.ppv_var  = PPV.var()

            self.for_mean = FOR.mean()
            self.for_var  = FOR.var()

    def update_prevalence(self, ensemble_states, scale = 'log' ):
        self._set_prevalence(ensemble_states)
        self._set_ppv(scale = scale)

    def get_mean_and_variance(self, positive_test = True, scale = 'log'):
        if scale == 'log':
            if positive_test:
                return self.logit_ppv_mean, self.logit_ppv_var
            else:
                return self.logit_for_mean, self.logit_for_var
        else:
            if positive_test:
                return self.ppv_mean, self.ppv_var
            else:
                return self.for_mean, self.for_var

    def take_measurements(self, nodes_state_dict, scale = 'log', noisy_measurement = False):
        """
        Queries the diagnostics from a medical test with defined `self.sensitivity` and `self.specificity` properties in
        population with a certain prevelance (computed from an ensemble of master equations).

        Noisy measurement can be enabled which will report back, for example, in a true infected a negative result with measurement `FOR`.

        Inputs:
        -------

        """


        measurements = {}
        uncertainty  = {}

        for node in nodes_state_dict.keys():
            if nodes_state_dict[node] == self.status:
                measurements[node], uncertainty[node] = self.get_mean_and_variance(scale = scale,
                                   positive_test = not (noisy_measurement and (np.random.random() > self.sensitivity)))
            else:
                measurements[node], uncertainty[node] = self.get_mean_and_variance(scale = scale,
                                   positive_test =     (noisy_measurement and (np.random.random() < 1 - self.specificity)))

        return measurements, uncertainty




#### Adding Observations in here



#We observe a subset of nodes at a status, only if the state exceeds a given threshold value.
#e.g we have a probability of observing I_i if (I_i > 0.8) when the observation takes place.
class StateInformedObservation:
    def __init__(self,
                 N,
                 obs_frac,
                 obs_status,
                 min_threshold,
                 max_threshold,
                 reduced_system):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
      
        if reduced_system == True:
            self.status_catalog = dict(zip(['S', 'I', 'H', 'R', 'D'], np.arange(5)))
  
        else:
            self.status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))
        self.n_status = len(self.status_catalog.keys())

        #array of status to observe
        self.obs_status_idx=np.array([self.status_catalog[status] for status in obs_status])

        #The fraction of states
        self.obs_frac = np.clip(obs_frac,0.0,1.0)
        #The minimum threshold (in [0,1]) to be considered for test (e.g 0.7)
        self.obs_min_threshold = np.clip(min_threshold,0.0,1.0)
        self.obs_max_threshold = np.clip(max_threshold,0.0,1.0)

        #default init observation
        self.obs_states = np.arange(int(self.obs_frac*self.N)*self.obs_status_idx.size)

    #updates the observation model when taking observation
    def find_observation_states(self, state):
        #Candidates for observations are those with a required state >= threshold
        onetoN=np.arange(self.N)
        candidate_states= np.hstack([self.N*self.obs_status_idx+i for i in onetoN])

        xmean = np.mean(state[:,candidate_states],axis=0)
        candidate_states_ens=candidate_states[(xmean>=self.obs_min_threshold) & \
                                                  (xmean<=self.obs_max_threshold)]

        M=candidate_states_ens.size
        if (int(self.obs_frac*M)>=1)&(self.obs_frac < 1.0) :
            # If there is at least one state to sample (...)>=1.0
            # If we don't just sample every state
            choice=np.random.choice(onetoM, size=int(self.obs_frac*M), replace=False)
            self.obs_states=candidate_states_ens[choice]
        elif (self.obs_frac == 1.0):
            self.obs_states=candidate_states_ens
        else: #The value is too small
            self.obs_states=np.array([],dtype=int)
            print("no observation was above the threshold")
    

#combine them together
class Observation(StateInformedObservation, TestMeasurement):

    def __init__(self,
                 N,
                 obs_frac,
                 obs_status,
                 obs_name,
                 min_threshold=0.0,
                 max_threshold=1.0,
                 reduced_system=True,
                 sensitivity = 0.80,
                 specificity = 0.99):
        
        self.name=obs_name
        
        StateInformedObservation.__init__(self,
                                          N,
                                          obs_frac,
                                          obs_status,
                                          min_threshold,
                                          max_threshold,
                                          reduced_system)

        TestMeasurement.__init__(self,
                                 obs_status,
                                 sensitivity,
                                 specificity,
                                 reduced_system)
        
    #State is a numpy array of size [self.N * n_status]    
    #def find_observation_states(self, state):
        # obtain where one should make an observation based on the
        # current state, and the contact network
    #   StateInformedObservation.find_observation_states(self,state)

    # data is a dictionary {node number : status} data[i] = contact_network.node(i)
    # status is 'I'
    def observe(self,
                contact_network,
                state,
                data,
                scale = 'log',
                noisy_measurement = False):

        #make a measurement of the data
        TestMeasurement.update_prevalence(self,
                                          state,
                                          scale)
        #mean, var np.arrays of size state
        observed_states = np.remainder(self.obs_states,self.N)
        #convert from np.array indexing to the node id in the (sub)graph
        observed_nodes = np.array(list(contact_network.nodes))[observed_states]
        observed_data = {node : data[node] for node in observed_nodes}

        mean,var =  TestMeasurement.take_measurements(self,
                                                      observed_data,
                                                      scale,
                                                      noisy_measurement)

        observed_mean     = np.array([mean[node] for node in observed_nodes])
        observed_variance = np.array([var[node] for node in observed_nodes])

        self.mean     = observed_mean
        self.variance = observed_variance