import yaml
from ml_collections import config_dict
from pathlib import Path
import os

class FlowSeqDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        # Force all sequences (lists) to use flow style
        return super().represent_sequence(tag, sequence, flow_style=True)

def read_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        raise Exception(f"Error reading {yaml_file}: {e}") from e
    return data

def read_config(path):
    """Reads the YAML config file at ``path`` into a ConfigDict."""
    data = read_yaml(path)
    return config_dict.ConfigDict(data)



def create_config_dict(config: dict) -> config_dict.ConfigDict:
    """Converts a dictionary to a ConfigDict."""
    config_dict_obj = config_dict.ConfigDict()
    for key, value in config.items():
        if isinstance(value, dict):
            config_dict_obj[key] = create_config_dict(value)
        else:
            config_dict_obj[key] = value
    return config_dict_obj

def save_config(config, path):
    """Saves a ConfigDict (or dict) to a YAML file."""
    if isinstance(config, config_dict.ConfigDict):
        config = config.to_dict()
    with open(path, 'w') as file:
        yaml.dump(config, file, Dumper=FlowSeqDumper, default_flow_style=False)


def get_commit_hash(warn=True):
    import subprocess

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()

        # Check for unadded / uncommitted changes
        status_output = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).decode('utf-8').strip()

        if status_output and warn:
            input("⚠️ Warning: There are unadded or uncommitted changes in the repository. Press ENTER to continue...")

        return commit_hash

    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hash: {e}")
        return None
    
    
class Config:
    
    def __init__(self, config: Path):
        self.cfg = read_config(config)
        
    def load(self, config):
        self.cfg = read_config(config)
        return self.cfg
    
    def save(self, path: Path, make_dir=True, rewrite_test=True, warn_github_changes=True):
        if make_dir:
            os.makedirs(self.run_dir, exist_ok=self.cfg.name == 'test')
        
        if self.cfg.name != 'test' and rewrite_test:
            self.cfg.git_hash = get_commit_hash(warn=warn_github_changes)
    
        save_config(self.cfg, path)
    
    @property
    def algorithm(self) -> str:
        """Learning algorithm, like ``MORLAX``"""
        return self.cfg.algorithm

    @property
    def env_name(self) -> str:
        """Environment class name, e.g. ``'MOCheetah'``."""
        return self.cfg.env

    @property
    def backend(self) -> str:
        """``'jnp'`` when building an env for training, else the configured backend."""
        return self.cfg.backend

    @property
    def env_params(self) -> config_dict.ConfigDict:
        """The ``env_config`` section, given to env classes as ``env_params``."""
        return create_config_dict(self.cfg.env_config.to_dict())

    @property
    def ppo_params(self) -> dict:
        """PPO settings shared by both algorithms."""
        return self.cfg.learning_params.ppo_params.to_dict()

    @property
    def network_params(self) -> dict:
        """Network settings for the configured algorithm."""
        return self.cfg.learning_params.network_params.to_dict()

    @property
    def normalize_observations(self) -> bool:
        """Whether training keeps a running observation normalizer."""
        return self.cfg.learning_params.ppo_params.normalize_observations

    @property
    def name(self) -> str:
        """Run name."""
        return self.cfg.name

    @property
    def save_dir(self) -> Path:
        """Parent directory of all runs for this config."""
        return Path(self.cfg.save_dir)

    @property
    def run_dir(self) -> Path:
        """Run directory: ``save_dir / name``."""
        return self.save_dir / self.name
        