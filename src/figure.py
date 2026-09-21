import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,          
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6
})

plt.rcParams['text.usetex'] = True

S_max = 1.0  
P_target = 0.9  


theta = np.linspace(-np.pi/2, np.pi/2, 500)
P_boundary = S_max * np.cos(theta)
Q_boundary = S_max * np.sin(theta)


Q_max = np.sqrt(S_max**2 - P_target**2)
Q_traj = np.linspace(-Q_max, Q_max, 200)
P_traj = np.full_like(Q_traj, P_target)


angle_pf09 = np.arccos(0.9)
slope_pf09 = np.tan(angle_pf09)

fig = plt.figure(figsize=(2.35, 2.35))
grid = fig.add_gridspec(
    1, 2,
    width_ratios=(1.0, 1.5),
    left=0.20, right=0.98, bottom=0.20, top=0.97,
    wspace=0.08
)
ax = fig.add_subplot(grid[0, 0])
info_ax = fig.add_subplot(grid[0, 1])

lagging_color = "#350072"
leading_color = "#FFB411"
target_color = '#D55E00'
unity_color = "#B40000"

ax.plot(P_boundary, Q_boundary, color='black', linewidth=1.3,
        label=r'$S_\mathrm{max}$ limit')

ax.fill_betweenx(Q_boundary, 0, P_boundary, color='gray', alpha=0.1)

ax.plot(P_traj, Q_traj, color=target_color, linewidth=2.5,
        label='Target locus')

ax.plot([0, P_target], [0, P_target * slope_pf09],
        color=lagging_color, linestyle=(0, (5, 2)), linewidth=1.0,
        label=r'$\mathrm{pf} = 0.9$ (lagging)')
ax.plot([0, P_target], [0, -P_target * slope_pf09],
        color=leading_color, linestyle=(0, (1.5, 1.5)), linewidth=1.2,
        label=r'$\mathrm{pf} = 0.9$ (leading)')
ax.plot([0, 1.0], [0, 0],
        color=unity_color, linestyle='-.', linewidth=1.0,
        label=r'$\mathrm{pf} = 1$ (unity)')

ax.scatter([P_target, P_target, P_target], [Q_max, 0, -Q_max], 
           color=[lagging_color, unity_color, leading_color],
           edgecolor='white', linewidth=0.35, zorder=5, s=26)
# ax.scatter([P_target], [Q_max], color=lagging_color, edgecolor='white', linewidth=0.35, zorder=5, s=26, label ='Original operating point\n($\\mathrm{pf} = 0.9$, lagging)')

ax.set_xlim(0, 1.1)
ax.set_ylim(-1.1, 1.1)
guide_style = dict(color='black', linestyle=':', linewidth=0.8, zorder=1)
ax.vlines(P_target, ax.get_ylim()[0], Q_max, **guide_style)
ax.hlines(Q_max, ax.get_xlim()[0], P_target, **guide_style)

x_ticks = [0.0, 0.5, P_target, 1.0]
ax.set_xticks(x_ticks)
ax.set_xticklabels(['0', '0.5', f'{P_target:.1f}', '1.0'])

for tick_value, tick_label in zip(x_ticks, ax.get_xticklabels()):
    if np.isclose(tick_value, P_target):
        offset = -4.2 / 72
    elif np.isclose(tick_value, 1.0):
        offset = 4.2 / 72
    else:
        continue
    tick_label.set_transform(
        tick_label.get_transform()
        + matplotlib.transforms.ScaledTranslation(
            offset, 0, fig.dpi_scale_trans
        )
    )

y_ticks = [-1.0, -0.5, 0.0, Q_max, 1.0]
ax.set_yticks(y_ticks)
ax.set_yticklabels(['-1.0', '-0.5', '0', f'{Q_max:.2f}', '1.0'])

ax.set_xlabel(r'DER Active power, $p(\psi)$ (p.u.)')
ax.set_ylabel(r'DER Reactive power, $q(\psi)$ (p.u.)')
ax.set_aspect('equal', adjustable='box') 

handles, labels = ax.get_legend_handles_labels()
info_ax.axis('off')
info_ax.legend(
    handles, labels,
    loc='center left',
    bbox_to_anchor=(0.0, 0.5),
    borderaxespad=0,
    frameon=True,
    edgecolor='black',
    framealpha=1,
    handlelength=2.6,
    labelspacing=0.45,
    borderpad=0.55
)

# info_ax.scatter(
#     0.035, 0.39,
#     s=26, color=lagging_color,
#     edgecolor='white', linewidth=0.35,
#     transform=info_ax.transAxes, clip_on=False
# )
# info_ax.text(
#     0.10, 0.39,
#     'Original point\n'
#     '($\\mathrm{pf} = 0.9$, lagging)',
#     transform=info_ax.transAxes,
#     fontsize=8, va='center', linespacing=1.2
# )

# info_ax.plot(
#     [0.015, 0.065], [0.15, 0.15],
#     color=target_color, linewidth=3,
#     transform=info_ax.transAxes, clip_on=False
# )
# info_ax.text(
#     0.10, 0.15,
#     "IDSO target locus\n"
#     "($0.9 \\leq \\mathrm{pf} \\leq 1$)\n"
#     "for $Q$-support\n"
#     "remains within the\n"
#     "$S_\\mathrm{max}$ limit.",
#     transform=info_ax.transAxes,
#     fontsize=8, va='center', linespacing=1.2
# )
fig.savefig('PQCapability.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
fig.savefig('PQCapability.png', format='png', bbox_inches='tight', pad_inches=0.01)

