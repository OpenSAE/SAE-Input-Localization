"""Create the paper figures from the learned-SAE validation results.

Publication styling for the ICLR 2027 template (\\textwidth = 5.5 in, 10 pt Times
body). Figures are drawn at their printed width so LaTeX places them without
rescaling: learned_sae_validation at 0.98\\linewidth = 5.39 in and
learned_sae_certificates at 0.72\\linewidth = 3.96 in. Text content is unchanged.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--results', type=Path, default=ROOT / 'results')
PARSER.add_argument('--figures', type=Path, default=ROOT / 'figures')
OPTIONS = PARSER.parse_args()
RESULTS = OPTIONS.results.resolve()
FIGURES = OPTIONS.figures.resolve()
FIGURES.mkdir(exist_ok=True)

TEXTWIDTH = 5.5
FONT, TICK, SMALL = 9, 8, 7.5
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
    'figure.constrained_layout.wspace': 0.06,
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'font.size': FONT, 'axes.labelsize': FONT, 'axes.titlesize': FONT,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'legend.fontsize': SMALL,
    'axes.linewidth': 0.6, 'axes.edgecolor': '#333333',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlepad': 4.0, 'axes.labelpad': 2.5, 'axes.axisbelow': True,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.4, 'ytick.minor.size': 1.4,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
    'grid.linewidth': 0.5, 'grid.color': '#dddddd', 'grid.alpha': 1.0,
    'legend.frameon': False, 'legend.handlelength': 2.0, 'legend.handletextpad': 0.5,
    'legend.labelspacing': 0.3,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
BLUE, INK = '#0072B2', '#222222'
META = {'Creator': None, 'Producer': None, 'CreationDate': None}


def save(fig, stem):
    fig.savefig(FIGURES / f'{stem}.pdf', metadata=META)
    fig.savefig(FIGURES / f'{stem}.png', dpi=300, facecolor='white')
    plt.close(fig)


# Main two-panel figure.
fig, axes = plt.subplots(1, 2, figsize=(0.98 * TEXTWIDTH, 2.25))
ax = axes[0]
# Arity is ordered, so one hue at three depths, with the original dash patterns.
arity_color = {1: '#08306B', 3: '#2171B5', 5: '#6BAED6'}
for arity, linestyle in zip([1, 3, 5], ['-', '--', ':']):
    curves = []
    for seed in range(5):
        data = np.load(RESULTS / f'seed_{seed}.npz')
        W = data['z'].var(axis=0).sum()
        curves.append(data[f'curve_{arity}'] / W)
    curves = np.asarray(curves)
    x = np.arange(curves.shape[1])
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1)
    c = arity_color[arity]
    ax.fill_between(x, np.maximum(mean - sd, 0), np.minimum(mean + sd, 1), color=c,
                    alpha=.16, linewidth=0, zorder=1)
    ax.plot(x, mean, linestyle=linestyle, linewidth=1.4, color=c, label=f'Arity {arity}', zorder=3)
ax.set_xlim(0, 384); ax.set_ylim(0, 1.01)
ax.set_xticks([0, 64, 128, 192, 256, 320, 384])
ax.set_xlabel('Dependency budget $B$')
ax.set_ylabel(r'$\mathcal{L}_z(B)/W_z$')
ax.set_title('(a) Learned-SAE scope curves', loc='left')
ax.grid(True, axis='y'); ax.legend(loc='upper right')

ax = axes[1]
samples = pd.read_csv(RESULTS / 'sampling.csv')
Ns = sorted(samples.pairs_per_scope.unique())
means = []; lows = []; highs = []
for N in Ns:
    vals = samples.loc[samples.pairs_per_scope == N, 'max_curve_error_over_W'].to_numpy()
    means.append(vals.mean()); lows.append(np.quantile(vals, .1)); highs.append(np.quantile(vals, .9))
ax.fill_between(Ns, lows, highs, color=BLUE, alpha=.16, linewidth=0, zorder=1)
ax.plot(Ns, means, marker='o', linewidth=1.4, color=BLUE, markersize=4.2,
        markeredgecolor='white', markeredgewidth=0.5, zorder=3)
ax.set_xscale('log', base=2)
ax.set_xticks(Ns, [str(n) for n in Ns])
ax.minorticks_off()
ax.set_ylim(0, 0.18)
ax.set_yticks([0, 0.04, 0.08, 0.12, 0.16], ['0', '0.04', '0.08', '0.12', '0.16'])
ax.set_xlabel('Pairs per candidate scope')
ax.set_ylabel(r'$\max_B |\widehat{\mathcal{L}}_z(B)-\mathcal{L}_z(B)|/W_z$')
ax.set_title('(b) Finite-sample recovery', loc='left')
ax.grid(True, axis='y')
save(fig, 'learned_sae_validation')

# Appendix certificate diagnostic.
models = pd.read_csv(RESULTS / 'models.csv').groupby('map_arity').agg(['mean', 'std'])
code = pd.read_csv(RESULTS / 'code_order_bounds.csv').groupby('map_arity').agg(['mean', 'std'])
fig, ax = plt.subplots(figsize=(0.72 * TEXTWIDTH, 2.5))
x = np.arange(3); labels = ['1', '3', '5']; width = .26
actual = models[('B10', 'mean')].to_numpy(); spectral = models[('spectral_lower_B10', 'mean')].to_numpy(); order = code[('code_order_lower_B10', 'mean')].to_numpy()
colors = {'Exact learned budget': '#3A3A3A', 'Representation spectral certificate': '#BDBDBD',
          'Feature-order relaxation': BLUE}
for offset, values, label in [(-width, actual, 'Exact learned budget'), (0, spectral, 'Representation spectral certificate'), (width, order, 'Feature-order relaxation')]:
    bars = ax.bar(x + offset, values, width=width, label=label, color=colors[label], edgecolor='none', zorder=2)
    ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=7, color=INK)
ax.set_xticks(x, labels); ax.set_xlabel('Input-map arity'); ax.set_ylabel(r'Budget at $0.1 W_z$ omission')
ax.tick_params(axis='x', length=0)
ax.set_ylim(0, 240); ax.grid(True, axis='y')
# legend above the axes so it cannot cover bars or their value labels
fig.legend(loc='outside upper center', ncol=2, columnspacing=1.2, handlelength=1.2)
save(fig, 'learned_sae_certificates')
