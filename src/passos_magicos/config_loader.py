import yaml
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")
