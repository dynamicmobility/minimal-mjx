import yaml

class FlowSeqDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        # Force all sequences (lists) to use flow style
        return super().represent_sequence(tag, sequence, flow_style=True)

def open_yaml(path):
    # Read YAML from a file
    with open(path, "r") as f:
        gait = yaml.safe_load(f)
    
    return gait