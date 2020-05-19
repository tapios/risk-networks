import pandas as pd
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


def plot_eon_simulations(sims, **kwargs):
	fig, axes = plt.subplots(1, 2, figsize = (15, 4))

	for sim in tqdm(sims, desc = 'Plotting simulations'):
		times, states = sim.summary()
		axes[0].plot(times, states['S'], color = 'C0', **kwargs)
		axes[0].plot(times, states['R'], color = 'C4', **kwargs)
		axes[0].plot(times, states['I'], color = 'C1', **kwargs)

		axes[1].plot(times, states['E'], color = 'C3', **kwargs)
		axes[1].plot(times, states['I'], color = 'C1', **kwargs)
		axes[1].plot(times, states['H'], color = 'C2', **kwargs)
		axes[1].plot(times, states['D'], color = 'C6', **kwargs)

	axes[0].legend(['Susceptible', 'Resistant', 'Infected'],
		bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		ncol=3, mode="expand", borderaxespad=0.);
	axes[1].legend(['Exposed', 'Infected', 'Hospitalized', 'Death'],
		bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		ncol=4, mode="expand", borderaxespad=0.);

	for ax in axes:
		ax.set_xlim(-1, 40)

	plt.tight_layout()

	return axes

def eon_sims_to_pandas(ens, sims, times_plot = [5, 10, 15, 40]):
	df = pd.DataFrame(columns = ['node_name', 'node_number', 'time', 'S', 'E', 'I', 'H', 'R', 'D'])
	G = ens.ensemble[0].G

	for kk, node in tqdm(enumerate(G.nodes()), desc = 'Processing kMC', total = len(G)):
		data = defaultdict()

		for tn in times_plot:
			states, counts = np.unique([sim.node_status(node, tn) for sim in sims], return_counts = True)
			rel_freq = dict(zip(states, counts/ens.M))
			data[tn] = rel_freq

		df_node = pd.DataFrame.from_dict(data, orient = 'index')
		df_node['time'] = df_node.index
		df_node['node_name'] = node
		df_node['node_number'] = kk

		df = df.append(df_node)
		df.fillna(0.0, inplace = True)
	# df.sort_values(by=['node', 'time'], inplace=True)
	return df

def plot_eon_ode(sims, ke_euler, t, xlims = (-.25, 40), reduced = True, leave = False, figsize = (15, 4), **kwargs):
	fig, axes = plt.subplots(1, 2, figsize = figsize)

	for sim in tqdm(sims, desc = 'Plotting kMC', leave = leave):
		times, states = sim.summary()
		axes[0].plot(times, states['S'], color = 'C0', **kwargs)
		axes[0].plot(times, states['R'], color = 'C4', **kwargs)
		axes[0].plot(times, states['I'], color = 'C1', **kwargs)

		axes[1].plot(times, states['E'], color = 'C3', **kwargs)
		axes[1].plot(times, states['I'], color = 'C1', **kwargs)
		axes[1].plot(times, states['H'], color = 'C2', **kwargs)
		axes[1].plot(times, states['D'], color = 'C6', **kwargs)

	for mm in tqdm(range(ke_euler.shape[0]), desc = 'Plotting ODE', leave = leave):
		axes[0].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[0], color = 'white', linewidth = 4)
		axes[0].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[-2], color = 'white', linewidth = 4)
		axes[0].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'white', linewidth = 4)

		axes[1].plot(t, (1 - ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'white', linewidth = 4)
		axes[1].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'white', linewidth = 4)
		axes[1].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[-3], color = 'white', linewidth = 4)
		axes[1].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[-1], color = 'white', linewidth = 4)

		axes[0].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[0], color = 'C0', linestyle = '--', linewidth = 2)
		axes[0].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[-2], color = 'C4', linestyle = '--')
		axes[0].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'C1', linestyle = '--')

		axes[1].plot(t, (1 - ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'C3', linestyle = '--')
		axes[1].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'C1', linestyle = '--')
		axes[1].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[-3], color = 'C2', linestyle = '--')
		axes[1].plot(t, ke_euler[mm].reshape(5, -1, len(t)).sum(axis = 1)[-1], color = 'C6', linestyle = '--')

	axes[0].legend(['Susceptible', 'Resistant', 'Infected'],
		bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		ncol=3, mode="expand", borderaxespad=0.);
	axes[1].legend(['Exposed', 'Infected', 'Hospitalized', 'Death'],
		bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		ncol=4, mode="expand", borderaxespad=0.);

	for kk, ax in enumerate(axes):
		ax.set_xlim(xlims)

	plt.tight_layout()

def plot_node_probabilities(df, ke_euler, t, figsize = (12, 10), sharex = False, sharey = False):
	state_keys   = ['S', 'I', 'H', 'R', 'D']
	state_names  = ['Susceptible', 'Infected', 'Hospitalized','Resistant',  'Death']
	state_colors = ['C0', 'C1', 'C2', 'C4', 'C6']
	times_plot   = np.unique(df.time)
	M            = ke_euler.shape[0]

	fig, axes = plt.subplots(len(times_plot), len(state_keys), figsize = figsize, sharex = sharex, sharey = sharey)

	for ax in axes.flatten():
		ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle = '--', color = 'black')

	for jj, tn in tqdm(enumerate(times_plot), desc = 'Snapshot', total = len(times_plot)):
		for kk, ax in enumerate(axes[jj]):
			ax.scatter(
				df[df.time == tn][state_keys[kk]],
				ke_euler[:,:,t == tn].reshape(M, 5, -1)[:,kk].mean(axis = 0),
				color = state_colors[kk]
			)
			axes[-1, kk].set_xlabel(state_names[kk],
				labelpad = 15,
				bbox=dict(facecolor='gray', alpha=0.25,boxstyle="round"))

		axes[jj, 0].set_ylabel(r't = %2.0f'%tn,
			labelpad=15,
			bbox=dict(facecolor='gray', alpha=0.25,boxstyle="round"))

	fig.tight_layout(h_pad = 2.2, w_pad = .2)

	return fig, axes

def plot_node_state(df, ke_euler, t, ax, tn = 10, state = 'S'):
	state_keys   = ['S', 'I', 'H', 'R', 'D']
	state_names  = ['Susceptible', 'Infected', 'Hospitalized','Resistant',  'Death']
	state_colors = ['C0', 'C1', 'C2', 'C4', 'C6']

	state_to_plot=np.arange(5)[np.array([key == state for key in state_keys])][0]
	times_plot   = np.unique(df.time)
	M            = ke_euler.shape[0]

	ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle = '--', color = 'black')

	ax.scatter(
		df[df.time == tn][state_keys[state_to_plot]],
		ke_euler[:,:,t == tn].reshape(M, 5, -1)[:,state_to_plot].mean(axis = 0),
		color = state_colors[state_to_plot]
	)

	return ax
