from epiforecast.risk_simulator import MasterEquationModelEnsemble
import numpy as np

population = 1000
# define the contact network [networkx.graph] - haven't done this yet
contact_network = []

# give the ensemble size (is this required)
ensemble_size = 10

# take the generated rates (typically)
# transition_rates: array[ensemble_size, number of rate types (always 6?), population]
# transmission_rate: array[ensemble_size]
state_transition_rates = np.random.uniform(0.01, 0.99, size=(ensemble_size,6,population))
transmission_rate = 0.06*np.ones(ensemble_size)



master_eqn_ensemble = MasterEquationModelEnsemble(contact_network=contact_network,
                                                  state_transition_rates=transition_rates,
                                                  transmission_rate=transmission_rate,
                                                  ensemble_size=ensemble_size)


# in practice when we update we will always update rates and network at the same time:
new_contact_network = [] 
new_transition_rates = np.random.uniform(0.01, 0.99, size=(ensemble_size,6,population))
new_transmission_rate = 0.08*np.ones(ensemble_size)
master_eqn_ensemble.update_ensemble(new_contact_network=new_contact_network,
                                    new_transition_rates=new_transition_rates,
                                    new_transmission_rate=new_transmission_rate)

# simulate
current_state = np.random.uniform(0.01, 0.2, size=(ensemble_size,5*population)) 
static_network_interval=0.25
new_state = master_eqn_ensemble.simulate(current_state, static_network_interval)
#can optionally 
#n_steps=number of steps(=100), t_0=initial_time(=0), and closure(='independent') 

#verification test?
