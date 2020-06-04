import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import random 
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from matplotlib import rcParams

# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Latin Modern Roman',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True}
# tell matplotlib about your params
rcParams.update(params)

# set nice figure sizes
fig_width_pt = 368    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.75*fig_width*ratio  # height in inches
fig_size = [fig_width, fig_height]
rcParams.update({'figure.figsize': fig_size})

from epiforecast.scenarios import load_edges

def simulation_average(model_data, times, sampling_time = 1):
    """
    Returns daily averages of simulation data.
    """
    
    simulation_data_average = {}
    daily_average = {}

    for key in model_data.keys():
        simulation_data_average[key] = []
        daily_average[key] = []
    
    tav = 0

    for i in range(len(times)):
        for key in model_data.keys():
            simulation_data_average[key].append(model_data[key][i])

        if times[i] >= tav:
            for key in model_data.keys():
                daily_average[key].append(np.mean(simulation_data_average[key]))
                simulation_data_average[key] = []

            tav += sampling_time

    return daily_average

#%% load network data to get the population size
edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e4_nobeds.txt')) 

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)
#%%

#%% load simulation data (with and without interventions)
simulation_data = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'simulation_data_NYC_1e4_3.txt'))
simulation_data_nointervention = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'simulation_data_nointervention.txt'))

times = simulation_data[:,0]
times_nointervention = simulation_data_nointervention[:,0]

kinetic_model = {'S': simulation_data[:,1], 'E': simulation_data[:,2], 'I': simulation_data[:,3], 'R': simulation_data[:,4], 'H': simulation_data[:,5], 'D': simulation_data[:,6]}
kinetic_model_nointervention = {'S': simulation_data_nointervention[:,1], 'E': simulation_data_nointervention[:,2], 'I': simulation_data_nointervention[:,3], 'R': simulation_data_nointervention[:,4], 'H': simulation_data_nointervention[:,5], 'D': simulation_data_nointervention[:,6]}

#%% load NYC data
NYC_data = pd.read_csv(os.path.join('..', 'data', 'NYC_COVID_CASES', 'data_new_york.csv'))
NYC_cases = np.asarray([float(x) for x in NYC_data['Cases'].tolist()])
NYC_date_of_interest_cases = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_data['DATE_OF_INTEREST'].tolist()])

NYC_death_data = pd.read_csv(os.path.join('..', 'data', 'NYC_COVID_CASES', 'death_data_new_york.csv'))
NYC_data_date_of_interest_deaths = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_death_data.columns.values[1:]])
NYC_death_data = np.asarray(NYC_death_data.iloc[-1].tolist()[1:])+np.asarray(NYC_death_data.iloc[-2].tolist()[1:])

# population of NYC
NYC_population = 8.399e6

# fraction reported cases
fraction_reported = 0.13

# cumulative cases NYC
cumulative_reported_cases_NYC = 1/fraction_reported*np.cumsum(NYC_cases)/NYC_population*1e5
cumulative_deaths_NYC = np.cumsum(NYC_death_data)/NYC_population*1e5

#%% determine averages of simulation data
# daily averages of simulation data
# sampling_time = 1 means that we average over 1-day intervals
sampling_time = 1
daily_average = simulation_average(kinetic_model, times, sampling_time)
cumulative_cases_simulation = (1-np.asarray(daily_average['S'])/population)*1e5
cumulative_deaths_simulation = np.asarray(daily_average['D'])/population*1e5

daily_average_nointervention = simulation_average(kinetic_model_nointervention, times_nointervention, sampling_time)
cumulative_cases_simulation_nointervention = (1-np.asarray(daily_average_nointervention['S'])/population)*1e5
cumulative_deaths_simulation_nointervention = np.asarray(daily_average_nointervention['D'])/population*1e5

simulation_dates = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time) for i in range(len(cumulative_cases_simulation))])
simulation_dates_nointervention = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time) for i in range(len(cumulative_deaths_simulation_nointervention))])

#%% plot results

fig, ax = plt.subplots()

ax2 = ax.twinx()

ax.text(dt.date(2020, 3, 10), 0.9*60000, r'(a)')

#ax.plot(NYC_date_of_interest_cases[::3], cumulative_reported_cases_NYC[::3], marker = 'o', markersize = 3, color = 'k', ls = 'None', label = r'total cases (NYC)')
ax.plot(simulation_dates+dt.timedelta(days = 17), cumulative_cases_simulation, 'k', label = 'total cases (simulation)')
ax.plot(simulation_dates_nointervention+dt.timedelta(days = 17), cumulative_cases_simulation_nointervention, 'k', ls = '--')
ax.bar(NYC_date_of_interest_cases[::3], cumulative_reported_cases_NYC[::3], color = 'k', alpha = 0.5, width = 2)


#ax2.plot(NYC_data_date_of_interest_deaths[::3], cumulative_deaths_NYC[::3], marker = 's', markersize = 4, color = 'darkred', markeredgecolor = 'Grey', ls = 'None', label = r'death cases (NYC)')
ax2.plot(simulation_dates+dt.timedelta(days = 17), cumulative_deaths_simulation, 'darkred', label = 'death cases (simulation)')
ax2.plot(simulation_dates_nointervention+dt.timedelta(days = 17), cumulative_deaths_simulation_nointervention, 'darkred', ls = '--')
ax2.bar(NYC_data_date_of_interest_deaths[::3], cumulative_deaths_NYC[::3], color = 'darkred', alpha = 0.7, width = 2)

ax.text(dt.date(2020, 5, 14), 41700, r'no SD')

ax.annotate('', xy=(dt.date(2020, 5, 13), 43000), xytext=(dt.date(2020, 4, 21), 50000),
            arrowprops=dict(arrowstyle="<-, head_width = 0.1",
                            connectionstyle="arc3"),
            )
            
ax.annotate('', xy=(dt.date(2020, 5, 13), 43000), xytext=(dt.date(2020, 5, 1), 27000),
            arrowprops=dict(arrowstyle="<-, head_width = 0.1",
                            connectionstyle="arc3"),
            )

ax.text(dt.date(2020, 7, 5), 15000, r'loosening SD')

ax.fill_between((simulation_dates+dt.timedelta(days = 17))[106:], cumulative_cases_simulation[106:], 0, facecolor = 'Grey', alpha = 0.5)
         
ax.vlines(dt.date(2020, 7, 1), 0, cumulative_cases_simulation[107], linestyles = '--', linewidth = 1.0, alpha = 0.7)         
         
#ax.text(dt.date(2020, 3, 10), 0.13, 'no SD')
#ax.text(dt.date(2020, 4, 19), 0.03, 'SD intervention')
#ax.text(dt.date(2020, 6, 21), 0.13, 'loosening SD')

#ax.fill_between([dt.date(2020, 3, 1), dt.date(2020, 3, 26)], 1, 0, color = 'Salmon', alpha = 0.2)
#ax.fill_between([dt.date(2020, 3, 26), dt.date(2020, 6, 15)], 1, 0, color = 'orange', alpha = 0.2)
#ax.fill_between([dt.date(2020, 6, 15), dt.date(2020, 7, 26)], 1, 0, color = 'Salmon', alpha = 0.2)

ax.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 8, 9)])
ax.set_xticklabels(NYC_date_of_interest_cases[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylim(1,60000)
#ax.set_yscale('log')
ax.set_ylabel(r'total cases/100,000', labelpad = 3)

ax2.set_ylim(1,2000)
#ax2.set_yscale('log')
ax2.set_ylabel(r'total deaths/100,000', color = 'darkred')
ax2.tick_params(axis='y', labelcolor = 'darkred')   

#ax.legend(frameon = True, loc = 2, fontsize = 7)
#ax2.legend(frameon = True, loc = 1, fontsize = 7)

plt.tight_layout()
plt.margins(0,0)
plt.savefig('new_york_cases.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)

fig, ax = plt.subplots()

ax2 = ax.twinx()

ax.plot(NYC_date_of_interest[::3], reported_cases_NYC[::3], marker = 'o', markersize = 3, color = 'k', ls = 'None', label = r'cases (NYC)')
ax.plot(simulation_dates[:-1]+dt.timedelta(days = 17), reported_cases_simulation, 'k', label = 'cases (simulation)')

ax2.plot(NYC_date_of_interest[::3], reported_deaths_NYC[::3], marker = 's', markersize = 4, color = 'darkred', markeredgecolor = 'Grey', ls = 'None', label = r'deaths (NYC)')
ax2.plot(simulation_dates[:-1]+dt.timedelta(days = 17), reported_deaths_simulation, 'darkred', label = 'deaths (simulation)')

#ax.text(dt.date(2020, 3, 10), 0.13, 'no SD')
#ax.text(dt.date(2020, 4, 19), 0.03, 'SD intervention')
#ax.text(dt.date(2020, 6, 21), 0.13, 'loosening SD')
#
#
#ax.fill_between([dt.date(2020, 3, 1), dt.date(2020, 3, 26)], 1, 0, color = 'Salmon', alpha = 0.2)
#ax.fill_between([dt.date(2020, 3, 26), dt.date(2020, 6, 15)], 1, 0, color = 'orange', alpha = 0.2)
#ax.fill_between([dt.date(2020, 6, 15), dt.date(2020, 7, 26)], 1, 0, color = 'Salmon', alpha = 0.2)

ax.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 7, 26)])
ax.set_xticklabels(NYC_date_of_interest[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylim(0,0.05)
ax.set_ylabel(r'proportion of cases', labelpad = 3)

ax2.set_ylim(0,100)
ax2.set_ylabel(r'deaths/100,000', color = 'darkred')
ax2.tick_params(axis='y', labelcolor = 'darkred')   

ax.legend(frameon = True, loc = 1, fontsize = 7)
ax2.legend(frameon = True, loc = 4, fontsize = 7)

plt.tight_layout()
plt.margins(0,0)
plt.savefig('new_york_cases2.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)
