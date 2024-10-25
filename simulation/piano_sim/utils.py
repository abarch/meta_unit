"""Utility functions.

Colletions of utility functions that are used in the service.

- Load configuration
- Recursively search for directory


"""

import logging
import sys
from pathlib import Path

import yaml

# from pydantic import ValidationError
from yaml.loader import SafeLoader

# from service.schemas.config_schema import Config

logger = logging.getLogger(__name__)


def find_dir(target_dir: str = "config", path: Path = Path("./")) -> Path:
    """Recursively search for the target directory.

    Function starts at path and recursively searches for the target directory
    upwards.

    Args:
        target_dir (str): Name of the target directory
        path (Path): starting path

    Returns:
        path: Path to the target directory, starting from `path`
    """
    if (path / target_dir).is_dir():
        return path / target_dir
    elif path == path.parent:
        return None
    return find_dir(target_dir, path.parent)


def load_config() -> dict:
    """Load configuration file.

    Loads the configuration file from the config directory.
    The configuration file is named `config.yaml`.

    Returns:
        dict: Dictionary with config
    """
    config_dir = find_dir("config")
    config_path = Path.joinpath(config_dir, "config.yaml")
    try:
        # Open the file and load the file
        with config_path.open() as f:
            return yaml.load(f, Loader=SafeLoader)
    except yaml.YAMLError as exc:
        logger.error(exc)
        logger.error("Could not load %s!", config_path)
        sys.exit(1)


# def get_config() -> Config:
#     """Get configuration class.
#
#     Get the configuration class with the loaded configuration file.
#     The configuration file is validated with the pydantic model.
#
#     Returns:
#         Config: Configuration class
#     """
#     try:
#         return Config.model_validate(load_config())
#     except ValidationError as e:
#         logger.error("Error in configuration file: %s", e)
#         sys.exit(1)
