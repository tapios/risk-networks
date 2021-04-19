import os, sys; sys.path.append(os.path.join('..', '..'))
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import datetime as dt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks")

def plot_ensemble_states(t, states_perc, axes, color, a_min=0.0, a_max=None):
    axes.fill_between(t, np.clip(states_perc[0], a_min, a_max), np.clip(states_perc[-1], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    axes.fill_between(t, np.clip(states_perc[1], a_min, a_max), np.clip(states_perc[-2], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    axes.fill_between(t, np.clip(states_perc[2], a_min, a_max), np.clip(states_perc[-3], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    axes.plot(t, states_perc[3], color = color)

# Set plotting cases
user_population = 1e5 
x_ticker_freq = 14 
total_days = 126
OUTPUT_PATH = "output/"
OUTPUT_FIGURE_NAME = os.path.join(OUTPUT_PATH, 'closure_without_correction.pdf')
CASE_NAME = "mean_field_closure"

CASE_PATH = os.path.join(OUTPUT_PATH,
                         CASE_NAME)
statuses_sum_trace = np.load(
                        os.path.join(CASE_PATH, 
                            'trace_kinetic_statuses_sum.npy'))
master_states_sum_timeseries_container = np.load(
                                            os.path.join(CASE_PATH, 
                                           'trace_master_states_sum.npy'))

master_states_perc = np.percentile(master_states_sum_timeseries_container * user_population, 
                            q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

# Assemble list of cases for plotting
colors_list = ['cornflowerblue',
               '#EDBF64']
data_list = [statuses_sum_trace,
             master_states_perc]

# Set time range
base = dt.datetime(2020, 3, 5)
time_arr = [base + dt.timedelta(hours=3*i) for i in range(statuses_sum_trace.shape[0])]

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
fig_width_pt = 368    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.75*fig_width*ratio  # height in inches
fig_size = [1.5*fig_width, 2*fig_height]
rcParams.update({'figure.figsize': fig_size})

#%% Start Plotting 
fig, axs = plt.subplots(nrows = 2, ncols = 3)
legend_elements = [Line2D([0], [0], color=colors_list[0], lw=1.5, label='Kinetic simulation'),
                   Line2D([0], [0], color=colors_list[1], lw=1.5, label='Master equation')]

# Cumulative susceptible panel
ax00 = axs[0][0]
ax00.set_title(r'Susceptible')
ax00.set_ylabel("Number of people")
#ax00.set_ylim(0,800)
#ax00.set_xlim([time_arr[0], time_arr[-1]])
ax00.set_xlim([time_arr[0], time_arr[0] + dt.timedelta(total_days)])
ax00.xaxis.set_minor_locator(ticker.MultipleLocator(x_ticker_freq))
ax00.set_xticks([time_arr[0] + dt.timedelta(42*i) for i in range(4)])
ax00.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax00.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax00.plot(time_arr, data_list[0][:,0], 
          colors_list[0], linewidth = 1.5, zorder = 100)
plot_ensemble_states(time_arr, data_list[1][:,0,:], ax00, colors_list[1])
ax00.yaxis.grid()

# Cumulative exposed panel
ax01 = axs[0][1]
ax01.set_title(r'Exposed')
#ax01.set_ylim(0,60000)
#ax01.set_xlim([time_arr[0], time_arr[-1]])
ax01.set_xlim([time_arr[0], time_arr[0] + dt.timedelta(total_days)])
ax01.xaxis.set_minor_locator(ticker.MultipleLocator(x_ticker_freq))
ax01.set_xticks([time_arr[0] + dt.timedelta(42*i) for i in range(4)])
ax01.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax01.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax01.plot(time_arr, data_list[0][:,1], 
          colors_list[0], linewidth = 1.5, zorder = 100)
plot_ensemble_states(time_arr, data_list[1][:,1,:], ax01, colors_list[1])
ax01.yaxis.grid(zorder=0)

# Cumulative infectious panel
ax02 = axs[0][2]
ax02.set_title(r'Infectious')
ax02.set_ylim([0,10000])
#ax02.set_xlim([time_arr[0], time_arr[-1]])
ax02.set_xlim([time_arr[0], time_arr[0] + dt.timedelta(total_days)])
ax02.xaxis.set_minor_locator(ticker.MultipleLocator(x_ticker_freq))
ax02.set_xticks([time_arr[0] + dt.timedelta(42*i) for i in range(4)])
ax02.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax02.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax02.plot(time_arr, data_list[0][:,2], 
          colors_list[0], linewidth = 1.5, zorder = 100)
plot_ensemble_states(time_arr, data_list[1][:,2,:], ax02, colors_list[1])
ax02.yaxis.grid(zorder=0)

# Cumulative hospitalized panel
ax10 = axs[1][0]
ax10.set_title(r'Hospitalized')
ax10.set_ylabel("Number of people")
#ax10.set_xlim([time_arr[0], time_arr[-1]])
ax10.set_xlim([time_arr[0], time_arr[0] + dt.timedelta(total_days)])
ax10.xaxis.set_minor_locator(ticker.MultipleLocator(x_ticker_freq))
ax10.set_xticks([time_arr[0] + dt.timedelta(42*i) for i in range(4)])
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax10.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax10.plot(time_arr, data_list[0][:,3], 
          colors_list[0], linewidth = 1.5, zorder = 100)
plot_ensemble_states(time_arr, data_list[1][:,3,:], ax10, colors_list[1])
ax10.yaxis.grid(zorder=0)

# Cumulative resistant panel
ax11 = axs[1][1]
ax11.set_title(r'Resistant')
ax11.set_ylim([0,80000])
#ax11.set_xlim([time_arr[0], time_arr[-1]])
ax11.set_xlim([time_arr[0], time_arr[0] + dt.timedelta(total_days)])
ax11.xaxis.set_minor_locator(ticker.MultipleLocator(x_ticker_freq))
ax11.set_xticks([time_arr[0] + dt.timedelta(42*i) for i in range(4)])
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax11.plot(time_arr, data_list[0][:,4], 
          colors_list[0], linewidth = 1.5, zorder = 100)
plot_ensemble_states(time_arr, data_list[1][:,4,:], ax11, colors_list[1])
ax11.yaxis.grid(zorder=0)

# Cumulative death panel
ax12 = axs[1][2]
ax12.set_title(r'Deceased')
#ax12.set_xlim([time_arr[0], time_arr[-1]])
ax12.set_xlim([time_arr[0], time_arr[0] + dt.timedelta(total_days)])
ax12.xaxis.set_minor_locator(ticker.MultipleLocator(x_ticker_freq))
ax12.set_xticks([time_arr[0] + dt.timedelta(42*i) for i in range(4)])
ax12.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax12.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax12.plot(time_arr, data_list[0][:,-1], 
          colors_list[0], linewidth = 1.5, zorder = 100)
plot_ensemble_states(time_arr, data_list[1][:,-1,:], ax12, colors_list[1])
ax12.yaxis.grid(zorder=0)

# Other settings for plotting
ax01.set_zorder(1)  
ax01.patch.set_visible(False)  
ax11.set_zorder(1)  
ax11.patch.set_visible(False)
plt.tight_layout()
sns.despine(top=True, right=True, left=True)
plt.subplots_adjust(top=0.88)
fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 1.0),
           ncol=2)
plt.savefig(OUTPUT_FIGURE_NAME)
