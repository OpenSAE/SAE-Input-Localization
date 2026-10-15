"""Regenerate the exact-construction scope-curve figure (figures/frozen_curves.*).

Publication styling for the ICLR 2027 template (\\textwidth = 5.5 in, 10 pt Times
body). The figure is drawn at its printed width, 0.92\\linewidth = 5.06 in, so LaTeX
places it without rescaling. Text content is unchanged; the long y-axis label is
only wrapped onto two lines. Paths are resolved relative to this file, so the
script can be run from any directory.
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
VERIFICATION = Path(__file__).resolve().parent / 'verification_results.json'
FIGURES = ROOT / 'assets' / 'figures'

TEXTWIDTH = 5.5
FONT, TICK, SMALL = 9, 8, 7.5
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'font.size': FONT, 'axes.labelsize': FONT, 'axes.titlesize': FONT,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
    'legend.fontsize': SMALL, 'legend.title_fontsize': SMALL,
    'axes.linewidth': 0.6, 'axes.edgecolor': '#333333',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'axes.axisbelow': True,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
    'grid.linewidth': 0.5, 'grid.color': '#dddddd', 'grid.alpha': 1.0,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'legend.handletextpad': 0.5,
    'legend.labelspacing': 0.3, 'legend.columnspacing': 1.2,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# One colour, marker and dash pattern per order; the marker takes the colour of
# its own step line (the original drew markers with the next colour in the cycle).
# Where curves coincide, the lower orders are drawn on top with thinner lines so
# every order stays visible.
STYLE = {1: ('#0072B2', 'o', '-', 1.1),
         2: ('#009E73', 's', (0, (5, 2)), 1.3),
         3: ('#E69F00', '^', (0, (1.5, 1.5)), 1.5),
         4: ('#CC79A7', 'D', (0, (5, 1.5, 1.5, 1.5)), 1.7)}

with open(VERIFICATION, 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(0.92 * TEXTWIDTH, 2.4))
for record in sorted(data['family']['records'], key=lambda rec: -rec['order']):
    r = record['order']
    curve = record['curve']
    total = curve[0]
    x = list(range(len(curve)))
    y = [value / total if total else 0.0 for value in curve]
    color, marker, ls, lw = STYLE[r]
    ax.step(x, y, where='post', color=color, linestyle=ls, linewidth=lw, zorder=6 - r,
            label=fr'$r={r}$', marker=marker, markersize=3.2, markevery=None,
            markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.4)

ax.set_xlabel('Input-feature correspondence budget $B$')
ax.set_ylabel(textwrap.fill('Minimum omitted activation variance / total variance', 28))
ax.set_ylim(-0.03, 1.05)
ax.set_xlim(-0.2, 16.2)
ax.set_xticks(range(0, 17, 2))
handles, labels = ax.get_legend_handles_labels()
order = sorted(range(len(labels)), key=lambda i: labels[i])
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          title='Interaction order', ncol=2, loc='upper right')
ax.grid(True, axis='y')
fig.savefig(FIGURES / 'frozen_curves.pdf',
            metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
fig.savefig(FIGURES / 'frozen_curves.png', dpi=300, facecolor='white')
