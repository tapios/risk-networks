import os, sys; sys.path.append(os.path.join('..', '..'))
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

# Settings
EXP_NAME = 'u75_s0_d1_i0.03_exog' #'1e5_WRIO_1.0_1e-2_3.0_1e-8_recI_1e-2'
#EXP_PARAMS = [0, 4897, 9794, 24485, 97942]
#EXP_PARAMS = [0, 3672, 7347, 18364, 73353]
EXP_PARAMS = [0, 3672, 7347, 18364, 73456]

intervention_nodes = 'sick'
intervention_sick_isolate_time = 7.0
population = 97942

for param in EXP_PARAMS:

    OUTPUT_PATH = os.path.join('output', EXP_NAME+"_"+str(param))
    stored_nodes_to_intervene = np.load(os.path.join(OUTPUT_PATH, 'isolated_nodes.npy'),
                                        allow_pickle=True)
    stored_nodes_to_intervene = stored_nodes_to_intervene.item()
    
    times = []
    number_nodes_to_intervene = []
    for current_time in stored_nodes_to_intervene.keys():
        if intervention_nodes == 'random':
            nodes_to_intervene = stored_nodes_to_intervene[current_time]
        else:
            nodes_to_intervene = \
                                 np.unique( \
                                            np.concatenate([v \
                                                            for k, v in stored_nodes_to_intervene.items() \
                                                            if k > current_time - intervention_sick_isolate_time and k <= current_time]) \
                                        )
        times.append(current_time)
        number_nodes_to_intervene.append(nodes_to_intervene.size)

    #matplotlib.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
    matplotlib.rcParams.update({'font.size': 12})
    plt.figure()
    plt.plot(times, np.array(number_nodes_to_intervene)/population*100, lw=2)
    plt.xlabel("Current time (day)")
    plt.ylabel("Isolated population (%)")
    plt.xlim([times[0], times[-1]])
    plt.xticks(np.array(times)[::5])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'intervention_info.png'))
    plt.show()
