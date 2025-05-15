import yaml

def load_config(path="config/config.yml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config