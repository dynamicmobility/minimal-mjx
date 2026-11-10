"""Plot a run's progress.csv as the learning curve shown in the docs.

Usage: python docs/scripts/plot_progress.py <progress.csv> <out.png>
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

GREY = '#8a8f98'  # readable on both the light and dark docs themes


def main(csv_path, out_path):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.plot(df.x / 1e6, df.y, color='#2f7ed8', lw=2, marker='o', ms=4)
    ax.set_xlabel('environment steps (millions)', color=GREY)
    ax.set_ylabel('eval episode reward', color=GREY)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=GREY)
    ax.grid(alpha=0.25, color=GREY)
    fig.tight_layout()
    fig.savefig(out_path, transparent=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
