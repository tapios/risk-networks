import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import os

def ensemble_average(states, num_status):
    obs = np.mean(states.reshape(states.shape[0], states.shape[1], num_status, -1), axis=3)
    return obs

def plot_range(ax, steps, mean, vmin, vmax, cl):
    ax.plot(steps, mean, '--', color=cl)
    ax.fill_between(steps, vmin, vmax,
                     color=cl, alpha=0.2)

def plot_states(states_t, states, t_t, t_states, num_status, N, fig_name):
    ## states_t: observations 
    ## states: all states from backward model evaluations 

    fig, axes = plt.subplots(1, 2, figsize = (15, 4))

    states_avg = ensemble_average(states, num_status)
    mean_H = np.mean(states_avg, 0)
    min_H = np.min(states_avg, 0)
    max_H = np.max(states_avg, 0)

    alpha = .5
    
    axes[0].plot(t_t, N * states_t.reshape(6, N, -1).mean(axis = 1)[0], '*', color = 'C0', linewidth = 2)
    axes[0].plot(t_t, N * states_t.reshape(6, N, -1).mean(axis = 1)[-2], '*', color = 'C4')
    axes[0].plot(t_t, N * states_t.reshape(6, N, -1).mean(axis = 1)[2], '*', color = 'C1')
    plot_range(axes[0], t_states, N * mean_H[:,0], N * min_H[:,0], N * max_H[:,0], 'C0')
    plot_range(axes[0], t_states, N * mean_H[:,-2], N * min_H[:,-2], N * max_H[:,-2], 'C4')
    plot_range(axes[0], t_states, N * mean_H[:,2], N * min_H[:,2], N * max_H[:,2], 'C1')
    axes[0].set_ylim([0,400])
    
    axes[1].plot(t_t, N * states_t.reshape(6, N, -1).mean(axis = 1)[1], '*', color = 'C3')
    axes[1].plot(t_t, N * states_t.reshape(6, N, -1).mean(axis = 1)[2], '*', color = 'C1')
    axes[1].plot(t_t, N * states_t.reshape(6, N, -1).mean(axis = 1)[-3], '*', color = 'C2')
    axes[1].plot(t_t, N * states_t.reshape(6, N, -1).mean(axis = 1)[-1], '*', color = 'C6')
    plot_range(axes[1], t_states, N * mean_H[:,1], N * min_H[:,1], N * max_H[:,1], 'C3')
    plot_range(axes[1], t_states, N * mean_H[:,2], N * min_H[:,2], N * max_H[:,2], 'C1')
    plot_range(axes[1], t_states, N * mean_H[:,-3], N * min_H[:,-3], N * max_H[:,-3], 'C2')
    plot_range(axes[1], t_states, N * mean_H[:,-1], N * min_H[:,-1], N * max_H[:,-1], 'C6')
    axes[1].set_ylim([-50,200])
    
    axes[0].legend(['Susceptible', 'Resistant', 'Infected'],
                   bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.);
    axes[1].legend(['Exposed', 'Infected', 'Hospitalized', 'Death'],
                  bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=4, mode="expand", borderaxespad=0.);
    
    for ax in axes:
        ax.set_xlim(-1, np.max(t_states)*1.1)
    
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

def plot_params(steps, mean, vmin, vmax, truth, ylabel, fig_name):
    plt.figure()
    plt.plot(steps, mean, '-', color='b', label='Ensemble mean')
    plt.fill_between(steps, vmin, vmax,
                     color='gray', alpha=0.2, label=r'Range of ensemble')
    plt.hlines(truth, np.min(steps), np.max(steps), 'r', 'dashed', label='Truth')
    lg = plt.legend(loc=0)
    lg.draw_frame(False)
    plt.xticks(np.arange(min(steps), max(steps)+1, (max(steps) - min(steps)) / 10.))
    plt.xlabel('EAKF Steps')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

if __name__ == "__main__":

    matplotlib.rcParams.update({'font.size':16})

    directory = 'figs/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    steps_DA = 5
    N = 327
    num_status = 6
    t_t = np.arange(0.0, 25., 1.0)
    t_range = np.arange(0.0, 25., 1.0)
    obs_interval = 1

    x_all = pickle.load(open('data/x_back.pkl', 'rb'))
    x_all = x_all[:,::-1,:]
    data = pickle.load(open('data/states_truth_beta_0p04.pkl', 'rb'))
    print(data.shape)
    print(t_t[::obs_interval])

    plot_states(data[:,:25], x_all, \
                t_t[::obs_interval], t_range, \
                num_status, N, 'figs/EAKF-master-Eqn-backward.pdf')

    theta = pickle.load(open('data/u_back.pkl', 'rb'))
    theta = np.exp(theta)
    mean = np.mean(theta, 1)
    vmin = np.min(theta, 1)
    vmax = np.max(theta, 1)

    truth = [0.04]
    ylabels = [r'$\beta$']

    for iterN in range(theta.shape[-1]):
        plot_params(np.arange(steps_DA+1), mean[:,iterN], vmin[:,iterN], vmax[:,iterN], truth[iterN], \
                  ylabels[iterN], 'figs/params_'+str(iterN)+'_backward.pdf')
