from epiforecast.intervention import Intervention

from _user_network_init import user_population
from _master_eqn_init import ensemble_size
from _argparse_init import arguments

compartment_index = { 'S' : 0, 'E' : 1, 'I' : 2, 'H' : 3, 'R' : 4, 'D' : 5}

intervention = Intervention(
        user_population,
        ensemble_size,
        compartment_index,
        E_thr = arguments.intervention_E_min_threshold,
        I_thr = arguments.intervention_I_min_threshold)

#intervention_frequency: 
#'none' (default),  - no interventions
#'single', - intervene at a fixed time
#'interval' - intervene on an interval (after a fixed time)
intervention_frequency = arguments.intervention_frequency
#intervention_nodes: 
#'all' (default), - intervene all nodes
#'sick', - intervene model determined 'sick' nodes
#'random', - intervene on a random selection (given by a fixed budget)
#'test_data_only', - intervene if node tests positive
#'test_data_neighbors', - intervene all neighbors of nodes testing positive
#'contact_tracing'- intervene neighbors of a positive node if they have spent > 15 mins with the node over a given time period (default 10 days)
intervention_nodes = arguments.intervention_nodes
#intervention_type: 
#'social_distance' (default), - reduce daytime contact rate
#'isolate', - make daytime contact rate = nighttime contact rate
#'nothing' - do not apply an intervention (just test and record which nodes are subject to it)
intervention_type = arguments.intervention_type

intervention_interval = arguments.intervention_interval
intervention_sick_isolate_time = arguments.intervention_sick_isolate_time

from _utilities import are_close, modulo_is_close_to_zero

def query_intervention(
        intervention_frequency,
        current_time,
        intervention_start_time,
        static_contact_interval):
    """
    Function to query whether we should apply an intervention at a current time
    based on the intervention frequency
    """
    if intervention_frequency == "none":
        intervene_now = False
    elif intervention_frequency == "single":
        intervene_now = are_close(current_time,intervention_start_time)
    elif intervention_frequency == "interval":
        if current_time > intervention_start_time - 0.1*static_contact_interval:
            intervene_now = modulo_is_close_to_zero(current_time,
                                                    intervention_interval,
                                                    eps=static_contact_interval)
        else:
            intervene_now = False
    else:
        raise ValueError("unknown 'intervention_frequency'; " +
                         "choose from 'none' (default), 'single' or 'interval'")

    return intervene_now


