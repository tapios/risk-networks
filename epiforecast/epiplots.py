import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

def plot_master_eqns(states, t, axes = None, xlims = None, reduced_system = True, leave = False, figsize = (15, 4), **kwargs):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)
    if reduced_system:
        N_eqns = 5
        S, I, H, R, D = np.arange(N_eqns)
    else:
        N_eqns = 6
        S, E, I, H, R, D = np.arange(N_eqns)

    for mm in tqdm(range(states.shape[0]), desc = 'Plotting ODE', leave = leave):
        # axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[S], color = 'white', linewidth = 4)
        # axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[R], color = 'white', linewidth = 4)
        # axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[I], color = 'white', linewidth = 4)

        # if reduced_system:
        #     axes[1].plot(t, (1 - states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'white', linewidth = 4)
        # else:
        #     axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[E], color = 'white', linewidth = 4)
        # axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[I], color = 'white', linewidth = 4)
        # axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[H], color = 'white', linewidth = 4)
        # axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[D], color = 'white', linewidth = 4)
        #
        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[S], color = 'C0', linestyle = '--', linewidth = 2)
        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[R], color = 'C4', linestyle = '--')
        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[I], color = 'C1', linestyle = '--')

        if reduced_system:
            axes[1].plot(t, (1 - states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'C3', linestyle = '--')
        else:
            axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[E], color = 'C3', linestyle = '--')
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

def plot_ensemble_states(states, t, axes = None,
                                    xlims = None,
                                    reduced_system = True,
                                    leave = False,
                                    figsize = (15, 4),
                                    a_min = None,
                                    a_max = None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)

    ensemble_size = states.shape[0]
    if reduced_system:
        N_eqns = 5
        statuses = np.arange(N_eqns)
        statuses_colors = ['C0', 'C1', 'C2', 'C4', 'C6']
    else:
        N_eqns = 6
        statuses = np.arange(N_eqns)
        statuses_colors = ['C0', 'C3', 'C1', 'C2', 'C4', 'C6']
    population   = states.shape[1]/N_eqns

    states_sum  = states.reshape(ensemble_size, N_eqns, -1, len(t)).sum(axis = 2)
    states_perc = np.percentile(states_sum, q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

    for status in statuses:
        if (reduced_system and status in [0, 3]) or (not reduced_system and status in [0, 2, 4]):
            axes[0].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].plot(t, states_perc[3,status], color = statuses_colors[status])

        if (reduced_system and status in [1]) or (not reduced_system and status in [2, 3, 5]):
            axes[1].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].plot(t, states_perc[3,status], color = statuses_colors[status])

        if (reduced_system and status in [2, 4]) or (not reduced_system and status in [3, 5]):
            axes[2].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].plot(t, states_perc[3,status], color = statuses_colors[status])

    if reduced_system:
        residual_state = population - states_sum.sum(axis = 1)
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

def plot_kinetic_model_data(kinetic_model, axes):
    statuses_name   = kinetic_model.return_statuses
    statuses_colors = ['C0', 'C3', 'C1', 'C2', 'C4', 'C6']
    colors_dict = dict(zip(statuses_name, statuses_colors))

    data = kinetic_model.statuses
    axes[1].scatter(kinetic_model.current_time, data['I'][-1], c = colors_dict['I'], marker = 'x')
    axes[2].scatter(kinetic_model.current_time, data['H'][-1], c = colors_dict['H'], marker = 'x')
    axes[2].scatter(kinetic_model.current_time, data['D'][-1], c = colors_dict['D'], marker = 'x')

    # axes[2].set_ylim(-.5, 10)

    return axes

def plot_ensemble_transmission_latent_fraction(community_transmission_rate_trace, latent_periods_trace, time_horizon):
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
