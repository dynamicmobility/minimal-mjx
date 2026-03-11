import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mujoco
import h5py
import cv2
from pathlib import Path
import mediapy as media

def set_mpl_params():
    # Make fonts LaTeX-like (good for papers)
    mpl.rcParams.update({
        "text.usetex": True,  # Enable LaTeX
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",   # Computer Modern for math
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18
    })

def get_subplot_grid(n):
    ncols = np.ceil(np.sqrt(n)).astype(int)
    nrows = np.ceil(n / ncols).astype(int)
    return nrows, ncols

def get_mj_scene_option(contacts=True, perts=True, com=True):
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contacts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = perts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = com
    return scene_option

class InfoPlotter:
    
    DEFAULT_PLOTKEY = []
    
    def __init__(
        self,
        plotkey=None
    ):
        self.plotkey = plotkey if plotkey is not None else self.DEFAULT_PLOTKEY
        self.data = {}
        for key in self.plotkey:
            self.data[key] = []
        
        self.data['time'] = []
        
    def add_row(self, time, info):
        for key in self.plotkey:
            self.data[key].append(info[key].copy())
            
        self.data['time'].append(time)
    
    def to_numpy(self):
        for key in self.data:
            self.data[key] = np.array(self.data[key])
                
    def save_to_h5(self, filename):
        with h5py.File(filename, 'w') as f:
            for key in self.data:
                f.create_dataset(key, data=np.array(self.data[key]))
        

class MujocoPlotter:

    DEFAULT_PLOTKEY = ['qpos', 'qvel', 'ctrl', 'sensordata', 'qfrc_actuator']

    def __init__(
        self,
        plotkey=None,
        record_time=True
    ):
        if plotkey is None:
            plotkey = MujocoPlotter.DEFAULT_PLOTKEY
        self.plotkey = plotkey
        self.record_time = record_time
        
        self.data = {}
        for key in self.plotkey:
            self.data[key] = []
        
        if record_time:
            self.data['time'] = []

    def add_row(self, data):
        if self.record_time:
            self.data['time'].append(getattr(data, 'time'))
        
        for key in self.plotkey:
            self.data[key].append(getattr(data, key).copy())

    def to_numpy(self):
        for key in self.plotkey:
            self.data[key] = np.array(self.data[key])

    def save_to_h5(self, filename):
        with h5py.File(filename, 'w') as f:
            for key in self.data:
                f.create_dataset(key, data=np.array(self.data[key]))

    @classmethod
    def time_idx(cls, time, data):
        if 'time' not in data:
            raise ValueError("Time data is required for indexing.")
        
        time_array = np.array(data['time'])
        idx = np.searchsorted(time_array, time)
        if idx == 0 or idx == len(time_array):
            raise ValueError("Time index out of bounds.")
        return idx - 1

def add_text_to_frame(pixels, text, org, size=1, thickness=2, color=(255, 255, 255)):
    # Text settings
    # org = (50, 100) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = size

    # Add text to the image
    cv2.putText(
        pixels, text, org, font, fontScale, color, thickness, cv2.LINE_AA
    )
    return pixels

def get_subplot_grid(n):
    ncols = np.ceil(np.sqrt(n)).astype(int)
    nrows = np.ceil(n / ncols).astype(int)
    return nrows, ncols

class RewardPlotter:

    def __init__(self, metrics):
        self.axkey = {}
        self.axdata = {}
        self.rewards = []
        for i, metric in enumerate(metrics):
            self.axkey[metric] = i
            self.axdata[metric] = []

    def add_row(self, metrics, reward):
        self.rewards.append(reward)
        for metric in metrics:
            self.axdata[metric].append(metrics[metric])

    def plot(self):
        ncols, nrows = get_subplot_grid(len(self.axdata) + 1)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
        axs = axs.flatten()
        fig.suptitle('Metrics')

        for metric in self.axdata:
            axs[self.axkey[metric]].set_title(metric)
            axs[self.axkey[metric]].plot(self.axdata[metric])

        axs[len(self.axdata)].set_title('Total Reward')
        axs[len(self.axdata)].plot(self.rewards)
        return fig, axs
    
def ensure_dir_exists(path):
    response = input(f"Create directory {path}? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        return False
    Path(path).mkdir(parents=True, exist_ok=True)
    return True
    

def save_metrics(plotter, path=Path('visualization/metrics.pdf')):
    print(f'Saving metrics to {path}')
    ans = ensure_dir_exists(path)
    if not ans:
        return
    fig, axs = plotter.plot()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_video(frames, dt, path=Path('visualization/policy_rollout.mp4')):
    print(f'Saving video to {path}')
    ans = ensure_dir_exists(path)
    if not ans:
        return False
    media.write_video(path, frames, fps=round(1 / dt))


def load_dict_from_hdf5(filename):
    out = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            out[key] = np.array(f[key][:])
    return out