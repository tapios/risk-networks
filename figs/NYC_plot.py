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

import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks")

from matplotlib import rcParams

# customized settings
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
# tell matplotlib about your params
rcParams.update(params)

# set nice figure sizes
fig_width_pt = 368    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.75*fig_width*ratio  # height in inches
fig_size = [1.5*fig_width, 2*fig_height]
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
simulation_data = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'simulation_data_NYC_1e4_6.txt'))
simulation_data_nointervention = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'simulation_data_nointervention2.txt'))

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
fraction_reported = 1

# cumulative cases NYC
reported_cases_NYC = 1/fraction_reported*NYC_cases/NYC_population*1e5
reported_deaths_NYC = NYC_death_data/NYC_population*1e5
cumulative_reported_cases_NYC = np.cumsum(reported_cases_NYC)
cumulative_deaths_NYC = np.cumsum(reported_deaths_NYC)

NYC_death_data_weekly = np.mean(np.append(reported_deaths_NYC, (7-len(reported_deaths_NYC)%7)*[reported_deaths_NYC[-1]]).reshape(-1, 7), axis=1)

NYC_cases_weekly = np.mean(np.append(reported_cases_NYC, (7-len(reported_cases_NYC)%7)*[reported_cases_NYC[-1]]).reshape(-1, 7), axis=1)

#%% determine averages of simulation data
# daily averages of simulation data
# sampling_time = 1 means that we average over 1-day intervals
sampling_time = 7
daily_average = simulation_average(kinetic_model, times, sampling_time)
cumulative_cases_simulation = (1-np.asarray(daily_average['S'])/population)*1e5
daily_average = simulation_average(kinetic_model, times, sampling_time)
cumulative_deaths_simulation = np.asarray(daily_average['D'])/population*1e5

daily_average_nointervention = simulation_average(kinetic_model_nointervention, times_nointervention, sampling_time)
cumulative_cases_simulation_nointervention = (1-np.asarray(daily_average_nointervention['S'])/population)*1e5
cumulative_deaths_simulation_nointervention = np.asarray(daily_average_nointervention['D'])/population*1e5

cases_simulation = np.ediff1d(cumulative_cases_simulation)/sampling_time
cases_simulation_nointervention = np.ediff1d(cumulative_cases_simulation_nointervention)/sampling_time

simulation_dates = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time) for i in range(len(cumulative_cases_simulation))])
simulation_dates_nointervention = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time) for i in range(len(cumulative_deaths_simulation_nointervention))])

sampling_time2 = 7
daily_average2 = simulation_average(kinetic_model, times, sampling_time2)
cumulative_cases_simulation2 = (1-np.asarray(daily_average2['S'])/population)*1e5
daily_average2 = simulation_average(kinetic_model, times, sampling_time2)
cumulative_deaths_simulation2 = np.asarray(daily_average2['D'])/population*1e5
daily_average_nointervention2 = simulation_average(kinetic_model_nointervention, times_nointervention, sampling_time2)
cumulative_cases_simulation_nointervention2 = (1-np.asarray(daily_average_nointervention2['S'])/population)*1e5
cumulative_deaths_simulation_nointervention2 = np.asarray(daily_average_nointervention2['D'])/population*1e5

deaths_simulation = np.ediff1d(cumulative_deaths_simulation2)/sampling_time2
deaths_simulation_nointervention = np.ediff1d(cumulative_deaths_simulation_nointervention2)/sampling_time2

simulation_dates2 = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time2) for i in range(len(deaths_simulation))])
simulation_dates_nointervention2 = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time2) for i in range(len(deaths_simulation_nointervention))])

#%% plot results

fig, axs = plt.subplots(nrows = 2, ncols = 2)


# cumulative death panel
ax00 = axs[0][0]

ax00_2 = axs[0][0].twinx()

ax00.set_title(r'Deaths per 100,000')

ax00.text(dt.date(2020, 3, 12), 0.9*1000, r'(a)')

ax00.plot(simulation_dates+dt.timedelta(days = 13), cumulative_deaths_simulation, 'cornflowerblue', alpha = 1)
ax00.plot(simulation_dates_nointervention+dt.timedelta(days = 13), cumulative_deaths_simulation_nointervention, 'Grey')
ax00.bar(NYC_data_date_of_interest_deaths, cumulative_deaths_NYC, facecolor='Grey', edgecolor='Grey', alpha = 0.6, width = 0.00001, align = 'center')

ax00.text(dt.date(2020, 4, 29), 75, r'data')
ax00.text(dt.date(2020, 6, 14), 300, r'model')
ax00.text(dt.date(2020, 4, 20), 900, r'no SD')

ax00.set_ylabel("cumulative")

ax00.set_ylim(0,1000)
ax00.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 9, 7)])
ax00.set_xticklabels(NYC_date_of_interest_cases[::7], rotation = 45)
ax00.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax00.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax00.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax00_2.set_yticks([])

ax00.yaxis.grid()

# cumulative infection panel
ax01 = axs[0][1]
ax01.set_title(r'Infections per 100,000')

ax01_2 = axs[0][1].twinx()

ax01.text(dt.date(2020, 3, 12), 0.9*60000, r'(b)')

ax01.plot(simulation_dates+dt.timedelta(days = 13), cumulative_cases_simulation, 'cornflowerblue')
ax01.plot(simulation_dates_nointervention+dt.timedelta(days = 13), cumulative_cases_simulation_nointervention, 'Grey')
ax01_2.bar(NYC_date_of_interest_cases, cumulative_reported_cases_NYC, facecolor='red', edgecolor='red', alpha = 0.4, width = 0.00001, align = 'center')

ax01.text(dt.date(2020, 4, 29), 6000, r'data')
ax01.text(dt.date(2020, 6, 11), 18000, r'model')
ax01.text(dt.date(2020, 4, 5), 55000, r'no SD')
                           
ax01.set_ylim(0,60000)
ax01.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 9, 7)])
ax01.set_xticklabels(NYC_date_of_interest_cases[::7], rotation = 45)
ax01.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax01.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax01.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))


ax01_2.set_ylim(0,9000)
ax01_2.tick_params(axis='y', colors='indianred')
ax01_2.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
ax01.yaxis.grid()
#ax01_2.grid("off")

# death cases panel
ax10 = axs[1][0]

ax10_2 = axs[1][0].twinx()

ax10.text(dt.date(2020, 3, 12), 0.9*30, r'(c)')

ax10.plot(simulation_dates2+dt.timedelta(days = 13), deaths_simulation, 'cornflowerblue')
ax10.plot(simulation_dates_nointervention2+dt.timedelta(days = 13), deaths_simulation_nointervention, 'Grey')
ax10.fill_between(NYC_data_date_of_interest_deaths[::7]+dt.timedelta(days = 3.5), NYC_death_data_weekly,  edgecolor = 'k', facecolor = 'Grey', alpha = 0.4, linewidth = 1.)
ax10.plot(NYC_data_date_of_interest_deaths[::7]+dt.timedelta(days = 3.5), NYC_death_data_weekly,  color = 'k',  linewidth = 1.)

ax10.bar(NYC_data_date_of_interest_deaths, reported_deaths_NYC, facecolor='Grey', edgecolor='Grey', alpha = 0.5, width = 0.0001)

ax10.text(dt.date(2020, 4, 3), 10.5, r'new deaths', fontsize = 7)
ax10.text(dt.date(2020, 5, 18), 8, r'7-day average', fontsize = 7)
ax10.plot([dt.date(2020, 5, 16), dt.date(2020, 4, 28)], [8.0, 3.2], color = 'k', linewidth = 0.5)
                 
ax10.set_ylabel("daily")

ax10.set_ylim(0,30)
ax10.set_yticks([0,10,20,30])
ax10.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 9, 7)])
ax10.set_xticklabels(NYC_date_of_interest_cases[::7], rotation = 45)
ax10.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax10.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax10_2.set_yticks([])

ax10.yaxis.grid()

# cases panel
ax11 = axs[1][1]

ax11_2 = axs[1][1].twinx()

ax11.text(dt.date(2020, 3, 12), 0.9*2000, r'(d)')

ax11.plot(simulation_dates2+dt.timedelta(days = 13), cases_simulation, 'cornflowerblue')
ax11.plot(simulation_dates_nointervention2+dt.timedelta(days = 13), cases_simulation_nointervention, 'Grey')
ax11_2.fill_between(NYC_date_of_interest_cases[::7]+dt.timedelta(days = 3.5), NYC_cases_weekly,  edgecolor = 'red', facecolor = 'red', alpha = 0.15, linewidth = 1.)
ax11_2.plot(NYC_date_of_interest_cases[::7]+dt.timedelta(days = 3.5), NYC_cases_weekly,  color = 'red',  linewidth = 1.)

ax11_2.bar(NYC_date_of_interest_cases, reported_cases_NYC, facecolor='red', edgecolor='red', alpha = 0.4, width = 0.0001)

ax11.text(dt.date(2020, 3, 26), 550, r'new cases', fontsize = 7)
ax11.text(dt.date(2020, 5, 17), 400, r'7-day average', fontsize = 7)
ax11.plot([dt.date(2020, 5, 16), dt.date(2020, 4, 23)], [420, 240], color = 'k', linewidth = 0.5)

ax11.set_ylim(0,2050)
ax11.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 9, 7)])
ax11.set_xticklabels(NYC_date_of_interest_cases[::7], rotation = 45)
ax11.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
ax11_2.set_ylim(0,300)
ax11_2.set_yticks([0,100,200,300])
ax11_2.tick_params(axis='y', colors='indianred')
ax11_2.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
ax11.yaxis.grid()

plt.tight_layout()
plt.margins(0,0)
sns.despine(top=True, right=True, left=True)
plt.savefig('new_york_cases.pdf', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)
