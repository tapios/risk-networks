#load the initialization for the epidemic etc.
from _epidemic_initializer import *

from epiforecast.performance_metrics import *

import matplotlib
import matplotlib.pyplot as plt
import pickle

epidemic_statuses_all = pickle.load(open('data/epidemic_statuses_all.pkl', 'rb'))
states_trace_ensemble = pickle.load(open('data/states_trace_ensemble.pkl', 'rb'))

acc_calculator = ModelAccuracy()
acc_all = []

num_pts = len(epidemic_statuses_all)

for j in range(num_pts):
    acc = acc_calculator(epidemic_statuses_all[j], states_trace_ensemble[:,:,j])
    acc_all.append(acc)

matplotlib.rcParams.update({'font.size':16})
plt.plot(np.arange(num_pts)*static_contact_interval/24., np.array(acc_all))
plt.xlabel('Time (days)')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig('backward_forward_DA_accuracy.png')
