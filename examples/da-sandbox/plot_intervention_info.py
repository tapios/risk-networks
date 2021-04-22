import os, sys; sys.path.append(os.path.join('..', '..'))
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

# Settings
#EXP_NAME = 'test_and_isolate' 
#EXP_NAME = 'contact_trace_and_isolate' 
#EXP_NAME = 'u100_s0_d1_i0.01' 
#EXP_PARAMS = [0, 4897, 9794, 24485, 97942]

#EXP_NAME = 'u75rand_s0_d1_i0.01' 
#EXP_PARAMS = [0, 3672, 7347, 18364, 73353]

EXP_NAME = 'u75_s0_d1_i0.01' 
EXP_PARAMS = [0, 3672, 7347, 18364, 73456]
#EXP_PARAMS = [3672, 18364]

#EXP_NAME = 'u50rand_s0_d1_i0.005' 
#EXP_PARAMS = [0, 2448, 4897, 12242, 48371]

intervention_nodes = 'sick'
intervention_sick_isolate_time = 5.0
population = 97942

for param in EXP_PARAMS:

    OUTPUT_PATH = os.path.join('output', EXP_NAME+"_"+str(param))
    stored_nodes_to_intervene = np.load(os.path.join(OUTPUT_PATH, 'isolated_nodes.npy'),
                                        allow_pickle=True)
    stored_nodes_to_intervene = stored_nodes_to_intervene.item()

    intervened_duration={}
    duration_counter = np.zeros(100,dtype=int)
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
                
        for node in nodes_to_intervene:
            duration = int(intervened_duration.get(node, 0))
            
            if duration > 0:
                #then determine if this is part of an ongoing isolation, or a new isolation
                if node in yesterday_nodes_to_intervene:
                    intervened_duration[node] = int(duration + 1)
                else:
                    duration_counter[duration] += 1
                    intervened_duration[node] = int(1)
            else:
                intervened_duration[node] = int(1)
           

        yesterday_nodes_to_intervene = nodes_to_intervene

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


    #get hisogram of figure - NB includes figures from end time which will have histogram < 5.0
    for duration in intervened_duration.values():        
        duration_counter[int(duration)] += 1
        
    print(duration_counter)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(30)+1,duration_counter[1:31])
    fig.savefig(os.path.join(OUTPUT_PATH, 'intervention_durations.png'))
    
    
