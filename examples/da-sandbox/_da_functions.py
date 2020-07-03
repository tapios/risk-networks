import numpy as np

def forward_DA(forward_start_time, forward_interval, 
               day, static_contact_interval,
               states_ensemble,
               master_eqn_ensemble,
               epidemic_data_storage, user_network, user_nodes,
               assimilator_imperfect_observations, assimilator_perfect_observations,
               transition_rates_to_update_imperf_str, rates_inflation,
               transition_rates_ensemble, community_transmission_rate_ensemble):
    states_ensemble_all = np.zeros([states_ensemble.shape[0],
                                    states_ensemble.shape[1],
                                    int(forward_interval/static_contact_interval)])
    master_eqn_ensemble.set_start_time(forward_start_time)
    print('#'*60)
    print('Run forward prediction for time window [%2.3f, %2.3f]' \
          %(forward_start_time, forward_start_time+forward_interval))
    print('#'*60)
    for j in range(int(forward_interval/static_contact_interval)):
    
    
        loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=forward_start_time)
        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())
        states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, n_steps = 25)

        forward_end_time = forward_start_time + static_contact_interval

        print('Assimilating infection data:')
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_imperfect_observations.update(ensemble_state = states_ensemble,
                                                                data = loaded_data.end_statuses,
                                      full_ensemble_transition_rates = transition_rates_ensemble,
                                     full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                        user_nodes = user_nodes,
                                                              time = forward_end_time)

    
        if (j+1)%int(day/static_contact_interval) == 0:
            print('Assimilating hospital and death records:')
            (states_ensemble,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_perfect_observations.update(ensemble_state = states_ensemble,
                                                                  data = loaded_data.end_statuses,
                                        full_ensemble_transition_rates = transition_rates_ensemble,
                                       full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                            user_nodes = user_nodes,
                                                                  time = forward_end_time)

        for rate in transition_rates_ensemble:
            rate.add_noise_to_clinical_parameters(transition_rates_to_update_imperf_str,
                                                  rates_inflation)
            
        lp = np.array([rate.latent_periods for rate in transition_rates_ensemble])
        cip = np.array([rate.community_infection_periods for rate in transition_rates_ensemble])
        hip = np.array([rate.hospital_infection_periods for rate in transition_rates_ensemble])
        print("latent_periods             : mean", np.mean(np.exp(lp)),  "var", np.var(np.exp(lp)))
        print("community_infection_periods: mean", np.mean(np.exp(cip)), "var", np.var(np.exp(cip)))
        print("hospital_infection_periods : mean", np.mean(np.exp(hip)), "var", np.var(np.exp(hip)))
    
        forward_start_time = forward_end_time
        master_eqn_ensemble.set_states_ensemble(states_ensemble)
        master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                           new_transmission_rate = community_transmission_rate_ensemble)

        states_ensemble_all[:,:,j] = states_ensemble

    return (forward_start_time, 
            states_ensemble,
            states_ensemble_all,
            master_eqn_ensemble,
            transition_rates_ensemble, community_transmission_rate_ensemble) 

def backward_DA(backward_start_time, backward_interval_effective,
                day, static_contact_interval,
                master_eqn_ensemble, 
                epidemic_data_storage, user_network, user_nodes,
                assimilator_imperfect_observations, assimilator_perfect_observations,
                transition_rates_to_update_imperf_str, rates_inflation,
                transition_rates_ensemble, community_transmission_rate_ensemble):
    master_eqn_ensemble.set_start_time(backward_start_time)
    print('#'*60)
    print('Run backward DA for time window [%2.3f, %2.3f]' \
          %(backward_start_time, backward_start_time-backward_interval_effective))
    print('#'*60)
    for i in range(int(backward_interval_effective/static_contact_interval)):
        loaded_data=epidemic_data_storage.get_network_from_end_time(end_time=backward_start_time)
        user_network.update_from(loaded_data.contact_network)
        master_eqn_ensemble.set_mean_contact_duration(
                user_network.get_edge_weights())
        states_ensemble = master_eqn_ensemble.simulate_backwards(static_contact_interval, n_steps = 25)
        
        #at the update the time
        backward_end_time = backward_start_time - static_contact_interval
        
        print('Assimilating infection data:')
        (states_ensemble,
         transition_rates_ensemble,
         community_transmission_rate_ensemble
        ) = assimilator_imperfect_observations.update(ensemble_state = states_ensemble,
                                                                data = loaded_data.start_statuses,
                                      full_ensemble_transition_rates = transition_rates_ensemble,
                                     full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                          user_nodes = user_nodes,
                                                                time = backward_end_time)

        if (i+1)%int(day/static_contact_interval) == 0:
            print('Assimilating hospital and death records:')
            (states_ensemble,
             transition_rates_ensemble,
             community_transmission_rate_ensemble
            ) = assimilator_perfect_observations.update(ensemble_state = states_ensemble,
                                                                  data = loaded_data.start_statuses,
                                        full_ensemble_transition_rates = transition_rates_ensemble,
                                       full_ensemble_transmission_rate = community_transmission_rate_ensemble,
                                                            user_nodes = user_nodes,
                                                                  time = backward_end_time)

        for rate in transition_rates_ensemble:
            rate.add_noise_to_clinical_parameters(transition_rates_to_update_imperf_str,
                                                  rates_inflation)
        #update model parameters (transition and transmission rates) of the master eqn model

        #at the update the time
        backward_start_time = backward_end_time
        master_eqn_ensemble.set_states_ensemble(states_ensemble)
        master_eqn_ensemble.update_ensemble(new_transition_rates = transition_rates_ensemble,
                                           new_transmission_rate = community_transmission_rate_ensemble)

    return (backward_start_time,
           states_ensemble,
           master_eqn_ensemble,
           transition_rates_ensemble, community_transmission_rate_ensemble)
