import yaml
import sys
from ml_collections import config_dict

class FlowSeqDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        # Force all sequences (lists) to use flow style
        return super().represent_sequence(tag, sequence, flow_style=True)

def read_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        sys.exit(1) 
    return data

def read_config(path=None):
    """Reads the YAML config file"""
    if len(sys.argv) != 2 and path is None:
        print("Usage: python script.py <yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1] if path is None else path
    data = read_yaml(yaml_file)
    
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

def get_commit_hash():
    import subprocess

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()

        # Check for unadded / uncommitted changes
        status_output = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).decode('utf-8').strip()

        if status_output:
            input("⚠️ Warning: There are unadded or uncommitted changes in the repository. Press ENTER to continue...")

        return commit_hash

    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hash: {e}")
        return None