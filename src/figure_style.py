import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches

FIGSIZE_WIDE = (7.16, 3.35)
FIGSIZE_TWO_PANEL = (7.16, 5.50)
FIGSIZE_COMPACT = (3.20, 3.60)
FIGSIZE_13TestCase = (4.65, 2.0)
FIGSIZE_123TestCase = (6.25, 2.0)
FIGSIZE_13TestCaseCompact = (3.5, 2.0)
FIGSIZE_13TestCaseHCompact = (2.0, 2.0)

PHASE_STYLES = {
	'Phase A': {'color': '#0072B2', 'marker': 'o'},
	'Phase B': {'color': '#D55E00', 'marker': 'x'},
	'Phase C': {'color': '#009E73', 'marker': 's'},
}
VOLTAGE_LIMIT_COLOR = '#6F4E7C'

plt.rcParams.update({
	'font.family': 'serif',
	# 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
	# 'mathtext.fontset': 'stix',
	'font.size': 8.0,
	'axes.labelsize': 9.0,
	'axes.titlesize': 9.0,
	'axes.labelweight': 'normal',
	'axes.titleweight': 'normal',
	'axes.linewidth': 0.75,
	'xtick.labelsize': 8.0,
	'ytick.labelsize': 8.0,
	'legend.fontsize': 8.0,
	'legend.frameon': False,
	'lines.linewidth': 1.05,
	'lines.markersize': 3.2,
	'xtick.direction': 'in',
	'ytick.direction': 'in',
	'xtick.major.size': 3.0,
	'ytick.major.size': 3.0,
	'xtick.minor.size': 1.5,
	'ytick.minor.size': 1.5,
	'xtick.major.width': 0.65,
	'ytick.major.width': 0.65,
	'pdf.fonttype': 42,
	'ps.fonttype': 42,
	'savefig.dpi': 600,
	'savefig.bbox': 'tight',
})
plt.rcParams['text.usetex'] = True

def apply_style(ax, grid_axis='y', inset=False):
	
	tick_size = 7.0 if inset else 8.0
	label_size = 8.0 if inset else 9.0
	title_size = 8.0 if inset else 9.0

	ax.set_axisbelow(True)
	ax.grid(False, which='both')
	if grid_axis is not None:
		ax.grid(
			axis=grid_axis,
			which='major',
			color='0.88',
			linestyle='-',
			linewidth=0.45,
		)

	for spine in ax.spines.values():
		spine.set_linewidth(0.75 if not inset else 0.65)
		spine.set_color('0.20')

	ax.tick_params(
		axis='both',
		which='major',
		direction='in',
		labelsize=tick_size,
		width=0.65,
		length=3.0 if not inset else 2.4,
	)
	ax.tick_params(
		axis='both',
		which='minor',
		direction='in',
		width=0.50,
		length=1.5,
	)

	if ax.get_yscale() == 'linear':
		ax.yaxis.set_minor_locator(AutoMinorLocator(2))

	ax.xaxis.label.set_fontsize(label_size)
	ax.yaxis.label.set_fontsize(label_size)
	ax.xaxis.label.set_fontweight('normal')
	ax.yaxis.label.set_fontweight('normal')
	ax.title.set_fontsize(title_size)
	ax.title.set_fontweight('normal')

	legend = ax.get_legend()
	if legend is not None:
		legend.set_frame_on(False)
		for text in legend.get_texts():
			text.set_fontsize(7.5 if inset else 8.0)

	return ax


def style_phase_plot(ax, inset=False, markevery=None):
	for line in ax.get_lines():
		label = line.get_label()

		if label in PHASE_STYLES:
			style = PHASE_STYLES[label]
			line.set_color(style['color'])
			line.set_marker(style['marker'])
			line.set_linewidth(0.90 if inset else 1.05)
			line.set_markersize(2.7 if inset else 3.2)
			line.set_markeredgewidth(0.65)
			if style['marker'] != 'x':
				line.set_markerfacecolor('white')
				line.set_markeredgecolor(style['color'])

			if markevery is not None:
				line.set_markevery(markevery)
			elif not inset and len(line.get_xdata()) > 50:
				line.set_markevery(2)

		# Identify horizontal voltage-limit lines, including unlabeled ones.
		y_data = np.asarray(line.get_ydata(), dtype=float)
		is_horizontal = len(y_data) >= 2 and np.allclose(y_data, y_data[0])
		if is_horizontal and (
			np.isclose(y_data[0], 0.95) or np.isclose(y_data[0], 1.05)
		):
			line.set_color(VOLTAGE_LIMIT_COLOR)
			line.set_linestyle((0, (4, 2)))
			line.set_linewidth(0.90 if inset else 1.0)

	apply_style(ax, grid_axis='y', inset=inset)
	return ax

