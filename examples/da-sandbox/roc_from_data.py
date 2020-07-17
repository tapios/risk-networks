import os
import numpy as np

EXP_NAME = 'sum_roc_itest_'
EXP_PARAM_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1.0]
THRESHOLD_NUM = 50
OUTDIR = 'output'


thresholds = 1.0/THRESHOLD_NUM*np.arange(THRESHOLD_NUM)
print(thresholds)
exp_run = [EXP_NAME + str(val) for val in EXP_PARAM_VALUES]
output_dirs = [os.path.join('.',OUTDIR, exp) for exp in exp_run]  

print(output_dirs)

for output_dir in output_dirs:
    #load in as performance_track
    true_positive_rate_ctrack_all = np.empty([THRESHOLD_NUM,0])
    true_negative_rate_track_all = np.empty([THRESHOLD_NUM,0])

    #load in performance from file
    
    fnames = [f for f in filter(lambda f: f.startswith('performance_track'), os.listdir(output_dir)) ]
    fnames.sort(key = lambda f: float(f.split('performance_track_')[1].split('.npy')[0])) #sort by performance_track_NUMBER.npy
    
    for fname in fnames:
        performance_track = np.load(os.path.join(output_dir,fname))
        
        #...
