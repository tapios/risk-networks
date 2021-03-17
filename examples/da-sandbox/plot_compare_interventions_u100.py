import os, sys; sys.path.append(os.path.join('..', '..'))
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import datetime as dt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks")


#Plot comparison script for interventions 100% user base
# Model - based intervention
# 

# Set plotting cases
population = 97942
OUTPUT_PATH = "output/"
OUTPUT_FIGURE_NAME = os.path.join(OUTPUT_PATH, 'compare_interventions_u100.pdf')

#
INTERVENTION_CASE_NAMES = [
    "u100_s0_d1_i0.03_4897",
    "u100_s0_d1_i0.03_24485",
    #"test_and_isolate_4897",
    #"test_and_isolate_24485",
    #"u100_s0_d1_randi0.1_0",
    #"u100_s0_d1_randi0.3_0",
    "u100_s0_d1_blanket_social_dist_0"]
#if there is no isolated_nodes.npy

#see the graph for types.
INTERVENTION_TYPE = [
    "daily", 
    "daily", 
    #"daily",
    #"daily",
#    "const",
#    "const", 
    None ]
MODEL_BASED = [
    True,
    True,   
#    False, 
#    False, 
#    False, 
#    False, 
    False]
USE_ISOLATED_NODES_NPY = [
    True,
    True, 
#    True,
#    True,  
#    True,
#    True,
    False]

INTERVENTION_LABELS = [
    'model (5\%)', 
    'model (25\%)',
#    'TI (5\%)',
#    'TI (25\%)',
#    'random isolate 10\%',
#    'random isolate 30\%',
    'blanket SD']

NO_INTERVENTION_CASE_NAMES = ["noda_1e5_parsd0.25_nosd_0"]
NO_INTERVENTION_LABELS = ['no intervention']
                             
USER_POPULATION = [population for x in np.arange(len(INTERVENTION_CASE_NAMES+NO_INTERVENTION_CASE_NAMES))]


#paths and files
INTERVENTION_CASE_PATH = [os.path.join(OUTPUT_PATH, name) for name in INTERVENTION_CASE_NAMES]
NO_INTERVENTION_CASE_PATH = [os.path.join(OUTPUT_PATH, name) for name in NO_INTERVENTION_CASE_NAMES]


statuses_sum_trace_list = [np.load(
    os.path.join(path,'trace_kinetic_statuses_sum.npy')) if USER_POPULATION[idx] == population
                           else np.load(
    os.path.join(path,'trace_full_kinetic_statuses_sum.npy'))
                           for (idx,path) in enumerate(INTERVENTION_CASE_PATH)]

statuses_sum_trace_nointervention_list = [np.load(
    os.path.join(path, 'trace_kinetic_statuses_sum.npy')) for path in NO_INTERVENTION_CASE_PATH]

#load isolation files where present
isolation_trace_list = [np.load(
    os.path.join(path,'isolated_nodes.npy'), allow_pickle=True) if USE_ISOLATED_NODES_NPY[idx] else None
                           for (idx,path) in enumerate(INTERVENTION_CASE_PATH)]
isolation_trace_list_nointervention = [ None for path in NO_INTERVENTION_CASE_PATH]


data_list = statuses_sum_trace_list + statuses_sum_trace_nointervention_list
model_based_list = MODEL_BASED + [False for name in NO_INTERVENTION_CASE_NAMES]
idata_list = isolation_trace_list + isolation_trace_list_nointervention
idata_type_list = INTERVENTION_TYPE + [None for name in NO_INTERVENTION_CASE_NAMES]

# Assemble list of cases for plotting - qualitative
#intervention_colors = [plt.cm.Pastel1(x) for x in np.arange(len(INTERVENTION_CASE_NAMES))]
#no_intervention_colors = [plt.cm.Dark2(x) for x in np.arange(len(NO_INTERVENTION_CASE_NAMES))]

colors_list = ['C'+ str(i) for i in np.arange(len(data_list))]
#[plt.cm.BuPu(x) for x in np.linspace(0.1,1.0,len(NO_INTERVENTION_CASE_NAMES)+len(INTERVENTION_CASE_NAMES))]
label_list = INTERVENTION_LABELS + NO_INTERVENTION_LABELS 
#['cornflowerblue',
#               '#EDBF64']
#data_list = [statuses_sum_trace,
#             statuses_sum_trace_nointervention]
#colors_list = intervention_colors + no_intervention_colors


# Set time range
base = dt.datetime(2020, 3, 5)
time_arr = [base + dt.timedelta(hours=3*i) for i in range(statuses_sum_trace_list[0].shape[0])]
days_per_tick=14
# Customized settings for plotting
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Helvetica',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True}
rcParams.update(params)

# Set figure sizes
fig_width_pt = 368*1.5    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.75*fig_width*ratio  # height in inches
fig_size = [1.5*fig_width, 2*fig_height]
rcParams.update({'figure.figsize': fig_size})

#%% Start Plotting 
fig, axs = plt.subplots(nrows = 2, ncols = 2)

# Cumulative death panel
ax00 = axs[0][0]
ax00.set_title(r'Deaths per 100,000')
ax00.set_ylabel("cumulative")
#ax00.set_ylim(0,800)
ax00.set_xlim([time_arr[0], time_arr[-1]])
ax00.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax00.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax00.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, color in zip(data_list, colors_list):
    #No construction as D accumulates naturally
    ax00.plot(time_arr, data[:,-1], 
              color, linewidth = 1.5, zorder = 100)
ax00.yaxis.grid()

# Cumulative infection panel
ax01 = axs[0][1]
ax01.set_title(r'Infections per 100,000')
ax01.set_ylabel("cumulative")
#ax01.set_ylim(0,60000)
ax01.set_xlim([time_arr[0], time_arr[-1]])
ax01.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax01.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax01.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, color,label in zip(data_list, colors_list, label_list):

    #Construct the cumulative I.
    Sout = -np.diff(data[:,0])
    Ein = Sout
    Eout = Ein - np.diff(data[:,1])
    Iin = Eout
    cumulative_Iin = [np.sum(Iin[0:i - 1]) + data[0,2] if i > 0 else data[0,2] for i in np.arange(Iin.shape[0] + 1)]     
    ax01.plot(time_arr, cumulative_Iin, 
              color, linewidth = 1.5, zorder = 100, label=label)
ax01.yaxis.grid(zorder=0)



#percent isolated panel
ax10 = axs[1][0]
ax10.set_title(r"Isolated nodes")
ax10.set_xlim([time_arr[0], time_arr[-1]])
ax10.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax10.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, data_type, color  in zip(idata_list, idata_type_list, colors_list):
    if data is not None:
        data = data.item()
        number_in_isolation = []
        days_pre_intervention = 8
        
        if data_type == "daily":
            for current_time in data.keys():
                intervened_nodes = np.unique(np.concatenate(
                    [v for k, v in data.items() 
                     if k > current_time - 7.0 and k <= current_time]))
                number_in_isolation.append(intervened_nodes.size)
                        
        elif data_type == "const":
            intervened_nodes = data[list(data.keys())[0]] 
            number_in_isolation = len(intervened_nodes) * np.ones(len(time_arr[::8]) - days_pre_intervention)
            
        number_in_isolation = np.hstack([np.zeros(days_pre_intervention),np.array(number_in_isolation)])

        #data frequency is only onces per day
        ax10.plot(time_arr[::8], number_in_isolation / population * 100, 
                  color, linewidth = 1.5, zorder = 100)


ax10.yaxis.grid()

# # Death cases panel
# ax10 = axs[1][0]
# ax10.set_ylabel("daily")
# #ax10.set_ylim(0,25)
# ax10.set_xlim([time_arr[0], time_arr[-1]])
# ax10.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
# ax10.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# ax10.get_yaxis().set_major_formatter(
#     ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# for data, color in zip(data_list, colors_list):
#     ax10.plot(time_arr[:-1], np.diff(data[:,-1]), 
#               color, linewidth = 1.5, zorder = 100)
# ax10.yaxis.grid()

# Daily new infection panel
ax11 = axs[1][1]
#ax01.set_ylim(0,60000)
ax11.set_ylabel("Infections per 100,000")
ax11.set_ylabel("Daily")

ax11.set_xlim([time_arr[0], time_arr[-1]])
ax11.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, color,label in zip(data_list, colors_list, label_list):

    #Construct the inflow to I.
    Sout = -np.diff(data[:,0])
    Ein = Sout
    Eout = Ein - np.diff(data[:,1])
    Iin = Eout #put a zero for day 1
    n_days = int((Iin.shape[0])/8)
    daily_cumulative_Iin = [np.sum(Iin[8*i : 8*(i+1) - 1]) for i in np.arange(n_days)]     
    ax11.plot(time_arr[7::8], daily_cumulative_Iin, 
              color, linewidth = 1.5, zorder = 100, label=label)
ax11.legend(loc='upper right')
ax11.yaxis.grid(zorder=0)

# # Current hospitalization pressure panel
# ax20 = axs[2][0]
# ax20.set_title(r'Hospital bed occupancy per 100,000')
# ax20.set_xlim([time_arr[0], time_arr[-1]])
# ax20.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
# ax20.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# ax20.get_yaxis().set_major_formatter(
#     ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# for data, color in zip(data_list, colors_list):
#     ax20.plot(time_arr, data[:,3], 
#               color, linewidth = 1.5, zorder = 100)
# ax20.yaxis.grid()



# Other settings for plotting
ax01.set_zorder(1)  
ax01.patch.set_visible(False)  
ax11.set_zorder(1)  
ax11.patch.set_visible(False)
plt.tight_layout()
plt.margins(0,0)
sns.despine(top=True, right=True, left=True)
plt.savefig(OUTPUT_FIGURE_NAME)
