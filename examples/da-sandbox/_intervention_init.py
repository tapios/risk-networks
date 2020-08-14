from epiforecast.intervention import Intervention

from _user_network_init import user_population
from _master_eqn_init import ensemble_size
from _argparse_init import arguments

compartment_index = { 'S' : 0, 'E' : -1, 'I' : 1, 'H' : 2, 'R' : 3, 'D' : 4}

intervention = Intervention(
        user_population,
        ensemble_size,
        compartment_index,
        E_thr = arguments.intervention_E_min_threshold,
        I_thr = arguments.intervention_I_min_threshold)

#intervention_frequency: 'none' (default), 'single', 'interval'
intervention_frequency = arguments.intervention_frequency
#intervention_nodes: 'all' (default), 'sick'
intervention_nodes = arguments.intervention_nodes
#intervention_type: 'social_distance' (default), 'isolate'
intervention_type = arguments.intervention_type
