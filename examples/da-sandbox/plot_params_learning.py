import numpy as np
import datetime as dt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os, sys; sys.path.append(os.path.join('..', '..'))
import matplotlib
import matplotlib.pyplot as plt

from _constants import (community_transmission_rate,
                        age_indep_transition_rates_true,
                        age_dep_transition_rates_true)

from _constants import time_span

def plot_transmission_rate(transmission_rate_timeseries,
                           t,
                           truth,
                           xticks,
                           params,
                           color='b',
                           a_min=None,
                           a_max=None,
                           output_path='.',
                           output_name='transmission_rate'):

    matplotlib.rcParams.update(params)
    rate_perc = np.percentile(transmission_rate_timeseries,
            q = [5, 10, 25, 50, 75, 90, 95], axis = 0)

    fig, ax = plt.subplots()
    ax.fill_between(t, np.clip(rate_perc[0,0], a_min, a_max), np.clip(rate_perc[-1,0], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    ax.fill_between(t, np.clip(rate_perc[1,0], a_min, a_max), np.clip(rate_perc[-2,0], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    ax.fill_between(t, np.clip(rate_perc[2,0], a_min, a_max), np.clip(rate_perc[-3,0], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    ax.plot(t, rate_perc[3,0], color = color)
    ax.axhline(truth, color = 'red', ls = '--')
    ax.set_ylabel('Transmission rate')
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(14))
    ax.set_xlim([t[0], t[-1]])
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,output_name+'.pdf'))
    plt.close()

def plot_ensemble_averaged_clinical_parameters(
        transition_rates_timeseries,
        t,
        xticks,
        params,
        age_indep_rates_true = None,
        age_dep_rates_true = None,
        color='b',
        a_min=None,
        a_max=None,
        num_rates=6,
        num_ages=5,
        age_dep_rates=[3,4,5],
        output_path='.',
        output_name=''):

    matplotlib.rcParams.update(params)
    rate_timeseries = transition_rates_timeseries
    rate_perc = np.percentile(rate_timeseries,
            q = [5, 10, 25, 50, 75, 90, 95], axis = 0)

    ylabel_list = ['latent_periods',
            'community_infection_periods',
            'hospital_infection_periods',
            'hospitalization_fraction',
            'community_mortality_fraction',
            'hospital_mortality_fraction']
    ylabel_name_list = ['latent_period',
            'community_infection_period',
            'hospital_infection_period',
            'hospitalization_fraction',
            'community_mortality_fraction',
            'hospital_mortality_fraction']
    age_indep_rates = [i for i in range(num_rates) if i not in age_dep_rates]

    for k in age_indep_rates:
        fig, ax = plt.subplots()
        ax.fill_between(t, np.clip(rate_perc[0,k], a_min, a_max), np.clip(rate_perc[-1,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        ax.fill_between(t, np.clip(rate_perc[1,k], a_min, a_max), np.clip(rate_perc[-2,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        ax.fill_between(t, np.clip(rate_perc[2,k], a_min, a_max), np.clip(rate_perc[-3,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        ax.plot(t, rate_perc[3,k], color = color)
        if age_indep_rates_true is not None:
            ax.axhline(age_indep_rates_true[ylabel_list[k]], color = 'red', ls = '--')

        ylabel_names = ylabel_name_list[k].split('_')
        ylabel_names[0] = ylabel_names[0].capitalize()
        ax.set_ylabel(' '.join(ylabel_names))
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(14))
        ax.set_xlim([t[0], t[-1]])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,ylabel_list[k]+output_name+'.pdf'))
        plt.close()

params = { 'font.family': 'sans-serif',
           'font.size': 16,
           'axes.labelsize': 'large',
           'axes.titlesize': 'large',
           'legend.fontsize': 'large',
           'xtick.labelsize': 'medium',
           'ytick.labelsize': 'medium',
           'savefig.dpi': 150}
   

OUTPUT_PATH = 'output/CASE_NAME'
network_transition_rates_timeseries = np.load(os.path.join(OUTPUT_PATH, 
                                              'network_mean_transition_rates.npy'))

network_transmission_rate_timeseries = np.load(os.path.join(OUTPUT_PATH,
                                              'network_mean_transmission_rate.npy'))

base = dt.datetime(2020, 3, 5)
time_span = [base + dt.timedelta(hours=3*i) for i in range(network_transition_rates_timeseries.shape[2])]
xticks = [base + dt.timedelta(42*i) for i in range(4)]

plot_ensemble_averaged_clinical_parameters(
    np.swapaxes(network_transition_rates_timeseries, 0, 1),
    time_span,
    xticks,
    params,
    age_indep_rates_true = age_indep_transition_rates_true,
    age_dep_rates_true = age_dep_transition_rates_true,
    a_min=0.0,
    output_path=OUTPUT_PATH,
    output_name='network')

plot_transmission_rate(np.swapaxes(network_transmission_rate_timeseries,0,1),
                       time_span,
                       community_transmission_rate,
                       xticks,
                       params,
                       a_min=0.0,
                       output_path=OUTPUT_PATH,
                       output_name='networktransmission_rate')
