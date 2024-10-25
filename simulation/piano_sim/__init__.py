"""Main module.

The init file is used to load the logging configuration and
provides some functionality to get the package version.
"""

import importlib.metadata
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

import toml
from dotenv import find_dotenv, load_dotenv
from piano_sim.utils import find_dir

LOG_DIR = "logs"
LOG_CONFIG_DIR = "logging_config"


def __get_package_version() -> str:
    """Find the version of this package.

    Try to get the version of the current package if
    it is running from a distribution.
    Fall back on getting it from a local pyproject.toml.
    This works in a development environment where the
    package has not been installed from a distribution.

    Returns:
        str: The version of the package.
    """
    package_version = "unknown"
    try:
        # Try to get the version of the current package if
        # it is running from a distribution.
        package_version = importlib.metadata.version("service")
    except importlib.metadata.PackageNotFoundError:
        # Fall back on getting it from a local pyproject.toml.
        # This works in a development environment where the
        # package has not been installed from a distribution.

        pyproject_toml_file = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
            package_version = toml.load(pyproject_toml_file)["tool"]["poetry"][
                "version"
            ]
            # Indicate it might be locally modified or unreleased.
            package_version = package_version + "-dev"

    return package_version


def __getattr__(name: str) -> str | int:
    """Get package attributes (version only).

    Get the package version if the attribute is `version` or `__version__`.

    Args:
        name (str): The name of the attribute to get (i.e. version).

    Raises:
        AttributeError: Name of the attribute is not `version` or `__version__`.

    Returns:
        str | int: The version of the package.
    """
    if name in ("version", "__version__"):
        return __get_package_version()
    else:
        raise AttributeError(f"No attribute {name} in module {__name__}.")


def setup_logging() -> None:
    """Load and setup logging configuration.

    Loads the logging configuration file based on the
    environment variable STAGE. If STAGE is not set, it
    defaults to "dev". If the STAGE is set to "prod", it
    loads the production logging configuration file.

    The logging configuration file is loaded from the
    log_config directory and is named based on the STAGE
    environment variable. If the file does not exist, it
    defaults to "logging.dev.ini".
    """
    config = {"dev": "logging.dev.ini", "prod": "logging.prod.ini"}.get(
        os.getenv("STAGE", "dev"), "logging.dev.ini"
    )

    try:
        config_dir = find_dir(LOG_CONFIG_DIR)
        config_path = Path.joinpath(config_dir, config)
    except IOError as e:
        logging.error(e)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    if not Path(LOG_DIR).exists():
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": f"{LOG_DIR}/{timestamp}.log"},
    )


# find .env file in parent directory
env_file = find_dotenv()
# Load the users .env file into environment variables
load_dotenv(env_file, verbose=True, override=True)

del load_dotenv
del env_file

setup_logging()
