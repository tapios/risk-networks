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


#Plot comparison script for interventions different user bases
# Model - based intervention
#
#cases '100', '75n', '75r', 50r, 50n
plot_case = '50r'


# Set plotting cases
population = 97942
OUTPUT_PATH = "output/"
if plot_case == '100':
    OUTPUT_FIGURE_NAME = os.path.join(OUTPUT_PATH, 'compare_interventions_u100.pdf')
elif plot_case == '75n':
    OUTPUT_FIGURE_NAME = os.path.join(OUTPUT_PATH, 'compare_interventions_u75nbhd.pdf')
elif plot_case == '75r':
    OUTPUT_FIGURE_NAME = os.path.join(OUTPUT_PATH, 'compare_interventions_u75rand_i0.0025.pdf')
elif plot_case == '50r':
    OUTPUT_FIGURE_NAME = os.path.join(OUTPUT_PATH, 'compare_interventions_u50rand_i0.0025.pdf')
elif plot_case == '50n':
    OUTPUT_FIGURE_NAME = os.path.join(OUTPUT_PATH, 'compare_interventions_u50nbhd.pdf')
else:
    raise ValueError("unknown plot_case")

#
days_pre_intervention = 9

if plot_case == '100':
    INTERVENTION_CASE_NAMES = [
        "u100_s0_d1_i0.01_4897",
        "u100_s0_d1_i0.01_24485",
        "contact_trace_and_isolate_4897",
        "contact_trace_and_isolate_24485",
        "blanket_social_dist_0"]

elif plot_case == '75n':
    INTERVENTION_CASE_NAMES = [
        "u75_s0_d1_i0.01_3672",
        "u75_s0_d1_i0.01_18364",
        "u75_contact_trace_and_isolate_3672",
        "u75_contact_trace_and_isolate_18364",
        "blanket_social_dist_0"]
    
elif plot_case == '75r':
    INTERVENTION_CASE_NAMES = [
        "u75rand_s0_d1_i0.0025_3672",
        "u75rand_s0_d1_i0.0025_18364",
        "u75rand_contact_trace_and_isolate_3672",
        "u75rand_contact_trace_and_isolate_18364",
        "blanket_social_dist_0"]
elif plot_case == '50r':
    INTERVENTION_CASE_NAMES = [
        "u50rand_s0_d1_i0.0025_2448",
        "u50rand_s0_d1_i0.0025_12242",
        "u50rand_contact_trace_and_isolate_2448",
        "u50rand_contact_trace_and_isolate_12242",
        "blanket_social_dist_0"]
elif plot_case == '50n':
    INTERVENTION_CASE_NAMES = [
        "u50_s0_d1_i0.005_2448",
        "u50_s0_d1_i0.005_12242",
        "u50_contact_trace_and_isolate_2448",
        "u50_contact_trace_and_isolate_12242",
        "blanket_social_dist_0"]
    
#if there is no isolated_nodes.npy

#see _intervention_init for types.
INTERVENTION_TYPE = [
    "daily", 
    "daily", 
    "daily",     
    "daily",     
    None ]
INTERVENTION_DURATIONS=[
    5.0,
    5.0,
    14.0,
    14.0,
    None]

MODEL_BASED = [
    True,
    True,   
    False,
    False,
    False]
USE_ISOLATED_NODES_NPY = [
    True,
    True, 
    True,
    True,
    False]

INTERVENTION_LABELS = [
    'Network DA (5\%)', 
    'Network DA (25\%)',
    'TTI (5\%)',
    'TTI (25\%)',
    'Lockdown']



NO_INTERVENTION_CASE_NAMES = ["noda_u100_prior_0"]
NO_INTERVENTION_LABELS = ['No intervention']


model_intervention_colors = [plt.cm.Greens(0.3), plt.cm.Greens(0.9)]
other_intervention_colors = [plt.cm.Purples(0.4), plt.cm.Purples(0.8), '#6394EB'] #Lockdown is blue
#other_intervention_colors = ['#6394EB'] #Lockdown is blue
no_intervention_colors    = ['#EBBD63'] #orange


colors_list = model_intervention_colors + other_intervention_colors + no_intervention_colors

USER_POPULATION = [population for x in np.arange(len(INTERVENTION_CASE_NAMES+NO_INTERVENTION_CASE_NAMES))]

if plot_case == '100':
    pass
elif plot_case == '75n':
    USER_POPULATION[0] = 73456
    USER_POPULATION[1] = 73456 
    USER_POPULATION[2] = 73456
    USER_POPULATION[3] = 73456 

elif plot_case == '75r':
    USER_POPULATION[0] = 73353
    USER_POPULATION[1] = 73353 
    USER_POPULATION[2] = 73353
    USER_POPULATION[3] = 73353 

elif plot_case == '50r':
    USER_POPULATION[0] = 48371
    USER_POPULATION[1] = 48371 
    USER_POPULATION[2] = 48371
    USER_POPULATION[3] = 48371 
    
elif plot_case == '50n':
    USER_POPULATION[0] = 48971
    USER_POPULATION[1] = 48971 
    USER_POPULATION[2] = 48971
    USER_POPULATION[3] = 48971 

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
intervention_durations = INTERVENTION_DURATIONS + [None for name in NO_INTERVENTION_CASE_NAMES]
# Assemble list of cases for plotting - qualitative

#colors_list = ['C'+ str(i) for i in np.arange(len(data_list))]
label_list = INTERVENTION_LABELS + NO_INTERVENTION_LABELS 

# Set time range
base = dt.datetime(2020, 3, 5)
time_arr = [base + dt.timedelta(hours=3*i) for i in range(statuses_sum_trace_list[0].shape[0])]
days_per_tick=14
# Customized settings for plotting
params = {  # 'backend': 'ps',
#    'font.family': 'sans-serif',
#    'font.sans-serif': 'Helvetica',
    'font.size': 11,
    'axes.labelsize': 'large',
    'axes.titlesize': 'large',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
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

labelpad=10.0

# Cumulative infection panel
ax00 = axs[0][0]
ax00.set_title(r'Infections per 100,000')
ax00.set_ylabel("Cumulative",labelpad=labelpad)
ax00.set_xlim([time_arr[0], time_arr[-1]])
ax00.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax00.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax00.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, color,label in zip(data_list, colors_list, label_list):

    #Construct the cumulative I.
    Sout = -np.diff(data[:,0])
    Ein = Sout
    Eout = Ein - np.diff(data[:,1])
    Iin = Eout
    cumulative_Iin = [np.sum(Iin[0:i - 1]) + data[0,2] if i > 0 else data[0,2] for i in np.arange(Iin.shape[0] + 1)]     
    ax00.plot(time_arr, cumulative_Iin, 
              color=color, linewidth = 1.5, zorder = 100, label=label)

ax00.set_ylim([0,None])
ax00.yaxis.grid()

# Cumulative death panel
ax01 = axs[0][1]
ax01.set_title(r'Deaths per 100,000')
ax01.set_ylabel("Cumulative",labelpad=labelpad)
ax01.set_xlim([time_arr[0], time_arr[-1]])
ax01.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax01.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax01.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, color in zip(data_list, colors_list):
    #No construction as D accumulates naturally
    ax01.plot(time_arr, data[:,-1], 
              color=color, linewidth = 1.5, zorder = 100)

ax01.set_ylim([0, None])
ax01.yaxis.grid()

# Daily new infection panel
ax10 = axs[1][0]
ax10.set_ylabel("Daily",labelpad=labelpad)
ax10.set_xlim([time_arr[0], time_arr[-1]])
ax10.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax10.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, color,label in zip(data_list, colors_list, label_list):

    #Construct the inflow to I.
    Sout = -np.diff(data[:,0])
    Ein = Sout
    Eout = Ein - np.diff(data[:,1])
    Iin = Eout #put a zero for day 1
    n_days = int((Iin.shape[0])/8)
    daily_cumulative_Iin = [np.sum(Iin[8*i : 8*(i+1) - 1]) for i in np.arange(n_days)]     
    ax10.plot(time_arr[7::8], daily_cumulative_Iin, 
              color=color, linewidth = 1.5, zorder = 100, label=label)

ax10.set_ylim([0, None])
ax10.legend(loc='upper right')
ax10.yaxis.grid()

#percent isolated panel
ax11 = axs[1][1]
ax11.set_title("Isolated Fraction")
ax11.set_ylabel("Percentage of population",labelpad=labelpad)
ax11.set_xlim([time_arr[0], time_arr[-1]])
ax11.xaxis.set_major_locator(ticker.MultipleLocator(days_per_tick))
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
for data, data_type, duration, color  in zip(idata_list, idata_type_list, intervention_durations, colors_list):
    
    if data is not None:
        data = data.item()
        number_in_isolation = []
        
        if data_type == "daily":
            for current_time in data.keys():
                intervened_nodes = np.unique(np.concatenate(
                    [v for k, v in data.items() 
                     if k > current_time - duration and k <= current_time]))
                number_in_isolation.append(intervened_nodes.size)
                        
        elif data_type == "const":
            intervened_nodes = data[list(data.keys())[0]] 
            number_in_isolation = len(intervened_nodes) * np.ones(len(time_arr[::8]) - days_pre_intervention)
            
        number_in_isolation = np.hstack([np.zeros(days_pre_intervention),np.array(number_in_isolation)])

        #data frequency is only onces per day
        ax11.plot(time_arr[::8], number_in_isolation / population * 100, 
                  color=color, linewidth = 1.5, zorder = 100)

ax11.yaxis.grid()


# Other settings for plotting
#ax00.set_zorder(1)  
#ax00.patch.set_visible(False)  
#ax10.set_zorder(1)  
#ax10.patch.set_visible(False)
plt.margins(0,0)
plt.tight_layout()
sns.despine(top=True, right=True, left=True)
plt.savefig(OUTPUT_FIGURE_NAME)
