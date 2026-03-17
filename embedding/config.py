"""Load setup.config content."""

import configparser
from pathlib import Path
from typing import Union

CONFIG_FILE = 'setup.config'


def read_config(file_path: str) -> configparser.RawConfigParser:
    """Read setup config file.
    
    Args:
        file_path: Path to the configuration file.
        
    Returns:
        ConfigParser object with parsed configuration.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_path = Path(str(Path(file_path).resolve()))
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {file_path} not found.")
    
    config_obj = configparser.RawConfigParser()
    config_obj.read(file_path)
    return config_obj


def get_config() -> configparser.RawConfigParser:
    """Returns the parsed configuration object.
    
    Returns:
        ConfigParser object with parsed configuration.
        
    Raises:
        FileNotFoundError: If setup.config doesn't exist.
    """
    return read_config(CONFIG_FILE)
