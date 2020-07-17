import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


COLORS_OF_STATUSES = {
        'S': 'C0',
        'E': 'C3',
        'I': 'C1',
        'H': 'C2',
        'R': 'C4',
        'D': 'C6'
}


def plot_master_eqns(
        states,
        t,
        axes=None,
        xlims=None,
        leave=False,
        figsize=(15, 4),
        **kwargs):

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)
    N_eqns = 5
    S, I, H, R, D = np.arange(N_eqns)

    for mm in tqdm(range(states.shape[0]), desc = 'Plotting ODE', leave = leave):

        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[S], color = 'C0', linestyle = '--', linewidth = 2)
        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[R], color = 'C4', linestyle = '--')
        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[I], color = 'C1', linestyle = '--')

        axes[1].plot(t, (1 - states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'C3', linestyle = '--')
        axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[I], color = 'C1', linestyle = '--')
        axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[H], color = 'C2', linestyle = '--')
        axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[D], color = 'C6', linestyle = '--')

    axes[0].legend(['Susceptible', 'Resistant', 'Infected'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode="expand", borderaxespad=0.);
    axes[1].legend(['Exposed', 'Infected', 'Hospitalized', 'Death'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=4, mode="expand", borderaxespad=0.);

    for kk, ax in enumerate(axes):
        ax.set_xlim(xlims)

    lg = axes[0].get_legend()
    hl = {handle.get_label(): handle for handle in lg.legendHandles}
    hl['_line0'].set_color('C0')
    hl['_line1'].set_color('C4')
    hl['_line2'].set_color('C1')

    lg = axes[1].get_legend()
    hl = {handle.get_label(): handle for handle in lg.legendHandles}
    hl['_line0'].set_color('C3')
    hl['_line1'].set_color('C1')
    hl['_line2'].set_color('C2')
    hl['_line3'].set_color('C6')

    plt.tight_layout()

    return axes

def plot_ensemble_states(
        population,
        states,
        t,
        axes=None,
        xlims=None,
        leave=False,
        figsize=(15, 4),
        a_min=None,
        a_max=None):

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)

    ensemble_size = states.shape[0]
    N_eqns = 5
    statuses = np.arange(N_eqns)
    statuses_colors = ['C0', 'C1', 'C2', 'C4', 'C6']
    user_population = int(states.shape[1]/N_eqns)

    states_sum  = (states.reshape(ensemble_size, N_eqns, -1, len(t)).sum(axis = 2))/population
    states_perc = np.percentile(states_sum, q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

    for status in statuses:
        if status in [0, 3]:
            axes[0].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].plot(t, states_perc[3,status], color = statuses_colors[status])

        if status in [1]:
            axes[1].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].plot(t, states_perc[3,status], color = statuses_colors[status])

        if status in [2, 4]:
            axes[2].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].plot(t, states_perc[3,status], color = statuses_colors[status])

    residual_state = user_population/population - states_sum.sum(axis = 1)
    residual_state = np.percentile(residual_state, q = [1, 10, 25, 50, 75, 90, 99], axis = 0)
    axes[0].fill_between(t, np.clip(residual_state[0], a_min, a_max), np.clip(residual_state[-1], a_min, a_max), alpha = .2, color = 'C3', linewidth = 0.)
    axes[0].fill_between(t, np.clip(residual_state[1], a_min, a_max), np.clip(residual_state[-2], a_min, a_max), alpha = .2, color = 'C3', linewidth = 0.)
    axes[0].fill_between(t, np.clip(residual_state[2], a_min, a_max), np.clip(residual_state[-3], a_min, a_max), alpha = .2, color = 'C3', linewidth = 0.)
    axes[0].plot(t, np.clip(residual_state[3], a_min, a_max), color = 'C3')

    axes[0].legend(['Susceptible', 'Resistant', 'Exposed'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode="expand", borderaxespad=0.);
    axes[1].legend(['Infected'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=2, mode="expand", borderaxespad=0.);
    axes[2].legend(['Hospitalized', 'Death'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=4, mode="expand", borderaxespad=0.);

    for kk, ax in enumerate(axes):
        ax.set_xlim(xlims)

    plt.tight_layout()

    return axes

def plot_epidemic_data(
        population,
        statuses_list,
        axes,
        plot_times):
    """
    Plot cumulative kinetic model states vs time

    Input:
        population (int): total population, used to normalize states to [0,1]
        statuses_list (list): timeseries of cumulative states; each element is a
                              6-tuple: (n_S, n_E, n_I, n_H, n_R, n_D)
        axes (np.array or list): (3,) array for plotting (S,E,R), (I) and (H,D)
        plot_times (np.array): (len(statuses_list),) array of times to plot
                               against
    Output:
        None
    """
    global COLORS_OF_STATUSES

    Sdata = [statuses_list[i][0]/population for i in range(len(plot_times))]
    Edata = [statuses_list[i][1]/population for i in range(len(plot_times))]
    Idata = [statuses_list[i][2]/population for i in range(len(plot_times))]
    Hdata = [statuses_list[i][3]/population for i in range(len(plot_times))]
    Rdata = [statuses_list[i][4]/population for i in range(len(plot_times))]
    Ddata = [statuses_list[i][5]/population for i in range(len(plot_times))]
    
    axes[0].scatter(plot_times, Sdata, c=COLORS_OF_STATUSES['S'], marker='x')
    axes[0].scatter(plot_times, Edata, c=COLORS_OF_STATUSES['E'], marker='x')
    axes[0].scatter(plot_times, Rdata, c=COLORS_OF_STATUSES['R'], marker='x')

    axes[1].scatter(plot_times, Idata, c=COLORS_OF_STATUSES['I'], marker='x')

    axes[2].scatter(plot_times, Hdata, c=COLORS_OF_STATUSES['H'], marker='x')
    axes[2].scatter(plot_times, Ddata, c=COLORS_OF_STATUSES['D'], marker='x')

    return axes

def plot_ensemble_transmission_latent_fraction(
        community_transmission_rate_trace,
        latent_periods_trace, time_horizon):

    transmission_perc = np.percentile(community_transmission_rate_trace, q = [1, 25, 50, 75, 99], axis = 0)
    latent_periods_perc = np.percentile(latent_periods_trace, q = [1, 25, 50, 75, 99], axis = 0)

    fig, axes = plt.subplots(1, 2, figsize = (12, 4))

    axes[0].fill_between(time_horizon, transmission_perc[0], transmission_perc[-1], alpha = .2, color = 'C0')
    axes[0].fill_between(time_horizon, transmission_perc[1], transmission_perc[-2], alpha = .2, color = 'C0')
    axes[0].plot(time_horizon, transmission_perc[2])
    axes[0].set_title(r'Transmission rate: $\beta$');

    axes[1].fill_between(time_horizon, latent_periods_perc[0], latent_periods_perc[-1], alpha = .2, color = 'C0')
    axes[1].fill_between(time_horizon, latent_periods_perc[1], latent_periods_perc[-2], alpha = .2, color = 'C0')
    axes[1].plot(time_horizon, latent_periods_perc[2])
    axes[1].set_title(r'Latent period: $\gamma$');

    return axes

def plot_scalar_parameters(parameters, time_horizon, names):
    percentiles = {}
    fig, axes = plt.subplots(1, len(parameters), figsize = (4 * len(parameters), 4))

    for kk, parameter in enumerate(names):
        percentiles[parameter] = np.percentile(parameters[kk], q = [1, 25, 50, 75, 99], axis = 0)

        axes[kk].fill_between(time_horizon, percentiles[parameter][0], percentiles[parameter][-1], alpha = .2, color = 'C0')
        axes[kk].fill_between(time_horizon, percentiles[parameter][1], percentiles[parameter][-2], alpha = .2, color = 'C0')
        axes[kk].plot(time_horizon, percentiles[parameter][2])
        axes[kk].set_title(names[kk]);

    return axes


def plot_roc_curve(true_negative_rates,
                   true_positive_rates,
                   labels = None,
                   show = True,
                   fig_size=(10, 5)):
    """
    Plots an ROC (Receiver Operating Characteristics) curve. This requires many experiments to
    as each experiments will produce one TNR, TPR pair.
    The x-axis is the False Positive Rate = 1 - TNR = 1 - TN / (TN + FP) 
    The y-axis is the True Positive Rate = TPR = TP / (TP + FN) 

    One can obtain these quantities through the PerformanceMetrics object
    
    Args
    ----
    true_negative_rates(np.array): array of true_negative_rates
    true_positive_rates(np.array): array of true_positive_rates of the same dimensions
    show                   (bool): bool to display graph
    labels                 (list): list of labels for the line plots
    """
    if true_negative_rates.ndim == 1:
        fpr = 1 -  np.array([true_negative_rates])
    else:
        fpr = 1 - true_negative_rates
        
    if true_positive_rates.ndim == 1:
        tpr = np.array([true_positive_rates])
    else:
        tpr = true_positive_rates

    # fpr,tpr size num_line_plots x num_samples_per_plot 
    colors = ['C'+str(i) for i in range(tpr.shape[0])]

    if labels is None:
        labels = ['ROC_' + str(i) for i in range(tpr.shape[0])]
        
    fig, ax = plt.subplots(figsize=fig_size)
    for xrate,yrate,clr,lbl in zip(fpr,tpr,colors,labels):
        plt.plot(xrate, yrate, color=clr, label=lbl )
            
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    if show:
        plt.show()
        
    return fig, ax
