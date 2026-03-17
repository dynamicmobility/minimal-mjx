"""Plotting utilities for MuJoCo simulation data, reward metrics, and video export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from datetime import datetime

import cv2
import h5py
import pandas as pd
import time
import wandb
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import mediapy as media
import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Matplotlib / rendering helpers
# ---------------------------------------------------------------------------

def set_mpl_params(label_size: int = 16, title_size: int = 18,
                   tick_size: int = 14, legend_size: int = 14) -> None:
    """Configure matplotlib for publication-quality LaTeX-style figures."""
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": label_size,
        "axes.titlesize": title_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "figure.titlesize": title_size,
    })


def get_subplot_grid(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for a roughly square grid that fits *n* subplots."""
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def get_mj_scene_option(contacts: bool = False, perts: bool = False, com: bool = False,
                        geomgroup2: bool = True, geomgroup3: bool = False) -> mujoco.MjvOption:
    """Create an MjvOption with common visualization flags toggled."""
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = geomgroup2
    scene_option.geomgroup[3] = geomgroup3
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contacts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = perts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = com
    return scene_option


def add_text_to_frame(pixels: np.ndarray, text: str, org: tuple[int, int],
                      size: int = 1, thickness: int = 2,
                      color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Overlay white text on a frame (in-place) and return it."""
    cv2.putText(
        pixels, text, org, cv2.FONT_HERSHEY_SIMPLEX,
        size, color, thickness, cv2.LINE_AA,
    )
    return pixels

def infer_frame_dim(
        mj_model, width, height
    ):
    if width is None:
        width = mj_model.vis.global_.offwidth
    if height is None:
        height = mj_model.vis.global_.offheight
    
    return width, height


# ---------------------------------------------------------------------------
# Data plotters
# ---------------------------------------------------------------------------

class InfoPlotter:
    """Collects arbitrary info-dict fields over time and exports to HDF5."""

    DEFAULT_PLOTKEY: list[str] = []

    def __init__(self, plotkey: list[str] | None = None) -> None:
        self.plotkey: list[str] = plotkey if plotkey is not None else self.DEFAULT_PLOTKEY
        self.data: dict[str, list[Any]] = {key: [] for key in self.plotkey}
        self.data['time'] = []

    def add_row(self, time: float, info: dict[str, Any]) -> None:
        """Append one timestep of data from an info dict."""
        for key in self.plotkey:
            self.data[key].append(info[key].copy())
        self.data['time'].append(time)

    def to_numpy(self) -> None:
        """Convert all stored lists to numpy arrays in-place."""
        for key in self.data:
            self.data[key] = np.array(self.data[key])

    def save_to_h5(self, filename: str | Path) -> None:
        """Write all data to an HDF5 file."""
        with h5py.File(filename, 'w') as f:
            for key in self.data:
                f.create_dataset(key, data=np.array(self.data[key]))


class MujocoPlotter:
    """Collects MuJoCo mjData fields over time and exports to HDF5."""

    DEFAULT_PLOTKEY: list[str] = ['qpos', 'qvel', 'ctrl', 'sensordata', 'qfrc_actuator']

    def __init__(self, plotkey: list[str] | None = None, record_time: bool = True) -> None:
        self.plotkey: list[str] = plotkey if plotkey is not None else self.DEFAULT_PLOTKEY
        self.record_time: bool = record_time
        self.data: dict[str, list[Any]] = {key: [] for key in self.plotkey}
        if record_time:
            self.data['time'] = []

    def add_row(self, data: mujoco.MjData) -> None:
        """Append one timestep of data from an mjData-like object."""
        if self.record_time:
            self.data['time'].append(getattr(data, 'time'))
        for key in self.plotkey:
            self.data[key].append(getattr(data, key).copy())

    def to_numpy(self) -> None:
        """Convert all stored lists to numpy arrays in-place."""
        for key in self.plotkey:
            self.data[key] = np.array(self.data[key])

    def save_to_h5(self, filename: str | Path) -> None:
        """Write all data to an HDF5 file."""
        with h5py.File(filename, 'w') as f:
            for key in self.data:
                f.create_dataset(key, data=np.array(self.data[key]))

    @classmethod
    def time_idx(cls, time: float, data: dict[str, list[Any]]) -> int:
        """Return the index of the last entry at or before *time*."""
        if 'time' not in data:
            raise ValueError("Time data is required for indexing.")
        time_array = np.array(data['time'])
        idx = np.searchsorted(time_array, time)
        if idx == 0 or idx == len(time_array):
            raise ValueError("Time index out of bounds.")
        return idx - 1


class RewardPlotter:
    """Accumulates per-step reward components and produces a summary figure."""

    def __init__(self, metrics: Sequence[str]) -> None:
        self.axkey: dict[str, int] = {}
        self.axdata: dict[str, list[float]] = {}
        self.rewards: list[float] = []
        for i, metric in enumerate(metrics):
            self.axkey[metric] = i
            self.axdata[metric] = []

    def add_row(self, metrics: dict[str, float], reward: float) -> None:
        """Record one timestep of metric values and total reward."""
        self.rewards.append(reward)
        for metric in metrics:
            self.axdata[metric].append(metrics[metric])

    def plot(self, figsize: tuple[int, int] = (10, 10),
             title: str = 'Metrics') -> tuple[mfig.Figure, np.ndarray]:
        """Create a multi-subplot figure of all metrics plus total reward."""
        nrows, ncols = get_subplot_grid(len(self.axdata) + 1)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        axs = axs.flatten()
        fig.suptitle(title)

        for metric in self.axdata:
            axs[self.axkey[metric]].set_title(metric)
            axs[self.axkey[metric]].plot(self.axdata[metric])

        axs[len(self.axdata)].set_title('Total Reward')
        axs[len(self.axdata)].plot(self.rewards)
        return fig, axs

def plot_progress(
    num_steps,
    metrics, 
    times, 
    x_data, 
    y_data, 
    y_dataerr, 
    ppo_params, 
    save_dir,
    run=None
):
    print('=== TRAINING EPOCH ===')
    print('time', time.time())
    print('num_steps', num_steps)
    print('total steps', ppo_params["num_timesteps"])
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    pd.DataFrame(
        {
            'times': times,
            'x': x_data,
            'y': y_data,
            'yerr': y_dataerr
        }
    ).to_csv(
        save_dir / 'progress.csv',
        index=False
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim([0, ppo_params["num_timesteps"] * 1.25])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    y_data = np.array(y_data)
    y_dataerr = np.array(y_dataerr)
    if np.nan in y_data or np.nan in y_dataerr:
        raise Exception(f'NaN found... \n\n{y_data}\n\n{y_dataerr}')

    ax.errorbar(x_data, y_data, yerr=y_dataerr)
    ax.scatter(x_data, y_data)

    save_dir = save_dir / 'progress.svg'
    plt.savefig(save_dir)
    if run:
        with open(save_dir, "r") as f:
            svg = f.read()
        run.log(
            {"reward_plot": wandb.Html(svg)},
            step=num_steps,
        )
        
# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def check_directory_exists(path):
    path = Path(path)
    if path.suffix:
        raise Exception('path is a file, not a directory', path.resolve())
    
    if path.exists() and path.is_dir():
        return True
    else:
        response = input(f"Directory '{path}' does not exist. Create it? [y/N]: ").strip().lower()
        if response == 'y':
            path.mkdir(parents=True, exist_ok=True)
            return True
        else:
            return False


def save_metrics(plotter: RewardPlotter, path: Path = Path('visualization/metrics.pdf')) -> None:
    """Plot metrics from a RewardPlotter and save the figure to *path*."""
    print(f'Saving metrics to {path}')
    path = Path(path)
    if not check_directory_exists(path.parent):
        return
    fig, _ = plotter.plot()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_video(frames: list[np.ndarray], dt: float,
               path: Path = Path('visualization/policy_rollout.mp4')) -> None:
    """Write a list of frames to an MP4 video at the environment control rate."""
    print(f'Saving video to {path}')
    path = Path(path)
    if not check_directory_exists(path.parent):
        return
    media.write_video(path, frames, fps=round(1 / dt))


def load_dict_from_hdf5(filename: str | Path) -> dict[str, np.ndarray]:
    """Load all datasets from an HDF5 file into a {key: ndarray} dict."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            out[key] = np.array(f[key][:])
    return out