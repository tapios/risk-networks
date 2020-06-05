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

def plot_ensemble_states(states, t, axes = None, xlims = None, reduced_system = True, leave = False, figsize = (15, 4)):
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
        if (reduced_system and status in [0, 1, 3]) or (not reduced_system and status in [0, 2, 4]):
            axes[0].fill_between(t, states_perc[0,status], states_perc[-1,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].fill_between(t, states_perc[1,status], states_perc[-2,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].fill_between(t, states_perc[2,status], states_perc[-3,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[0].plot(t, states_perc[3,status], color = statuses_colors[status])

        if (reduced_system and status in [1, 2, 4]) or (not reduced_system and status in [2, 3, 5]):
            axes[1].fill_between(t, states_perc[0,status], states_perc[-1,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].fill_between(t, states_perc[1,status], states_perc[-2,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].fill_between(t, states_perc[2,status], states_perc[-3,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[1].plot(t, states_perc[3,status], color = statuses_colors[status])

        if (reduced_system and status in [2, 4]) or (not reduced_system and status in [3, 5]):
            axes[2].fill_between(t, states_perc[0,status], states_perc[-1,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].fill_between(t, states_perc[1,status], states_perc[-2,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].fill_between(t, states_perc[2,status], states_perc[-3,status], alpha = .2, color = statuses_colors[status], linewidth = 0.)
            axes[2].plot(t, states_perc[3,status], color = statuses_colors[status])

    if reduced_system:
        residual_state = population - states_sum.sum(axis = 1)
        residual_state = np.percentile(residual_state, q = [1, 10, 25, 50, 75, 90, 99], axis = 0)
        axes[1].fill_between(t, residual_state[0], residual_state[-1], alpha = .2, color = 'C3', linewidth = 0.)
        axes[1].fill_between(t, residual_state[1], residual_state[-2], alpha = .2, color = 'C3', linewidth = 0.)
        axes[1].fill_between(t, residual_state[2], residual_state[-3], alpha = .2, color = 'C3', linewidth = 0.)
        axes[1].plot(t, residual_state[3], color = 'C3')

    axes[0].legend(['Susceptible', 'Resistant', 'Infected'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode="expand", borderaxespad=0.);
    axes[1].legend(['Exposed', 'Infected', 'Hospitalized', 'Death'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=2, mode="expand", borderaxespad=0.);
    axes[2].legend(['Hospitalized', 'Death'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=4, mode="expand", borderaxespad=0.);

    for kk, ax in enumerate(axes):
        ax.set_xlim(xlims)
    #
    # lg = axes[0].get_legend()
    # hl = {handle.get_label(): handle for handle in lg.legendHandles}
    # hl['_line0'].set_color('C0')
    # hl['_line1'].set_color('C4')
    # hl['_line2'].set_color('C1')
    #
    # lg = axes[1].get_legend()
    # hl = {handle.get_label(): handle for handle in lg.legendHandles}
    # hl['_line0'].set_color('C3')
    # hl['_line1'].set_color('C1')
    # hl['_line2'].set_color('C2')
    # hl['_line3'].set_color('C6')
    #
    plt.tight_layout()

    return axes
