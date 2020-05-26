import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

def plot_master_eqns(states, t, xlims = None, reduced = True, leave = False, figsize = (15, 4), **kwargs):
	fig, axes = plt.subplots(1, 2, figsize = figsize)

	for mm in tqdm(range(states.shape[0]), desc = 'Plotting ODE', leave = leave):
		axes[0].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[0], color = 'white', linewidth = 4)
		axes[0].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[-2], color = 'white', linewidth = 4)
		axes[0].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'white', linewidth = 4)

		axes[1].plot(t, (1 - states[mm].reshape(5, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'white', linewidth = 4)
		axes[1].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'white', linewidth = 4)
		axes[1].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[-3], color = 'white', linewidth = 4)
		axes[1].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[-1], color = 'white', linewidth = 4)

		axes[0].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[0], color = 'C0', linestyle = '--', linewidth = 2)
		axes[0].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[-2], color = 'C4', linestyle = '--')
		axes[0].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'C1', linestyle = '--')

		axes[1].plot(t, (1 - states[mm].reshape(5, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'C3', linestyle = '--')
		axes[1].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[1], color = 'C1', linestyle = '--')
		axes[1].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[-3], color = 'C2', linestyle = '--')
		axes[1].plot(t, states[mm].reshape(5, -1, len(t)).sum(axis = 1)[-1], color = 'C6', linestyle = '--')

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

	return fig, axes
