"""Load project configuration from YAML files.

Supports a base config.yaml with overrides from config.local.yaml,
so each developer can set their own Google Drive paths without
modifying the tracked config file.
"""

from pathlib import Path

import yaml


# Project root is two levels up from this file (src/utils/config_loader.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(config_name: str = "config.yaml") -> dict:
    """Load configuration from YAML, with local overrides.

    Loads config/<config_name> first, then merges any overrides from
    config/<name>.local.yaml (e.g. config.local.yaml).

    Args:
        config_name: Name of the base config file.

    Returns:
        Merged configuration dictionary.
    """
    base_path = CONFIG_DIR / config_name
    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")

    with open(base_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Look for local override file
    stem = base_path.stem
    local_path = CONFIG_DIR / f"{stem}.local.yaml"
    if local_path.exists():
        with open(local_path, "r") as f:
            local_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, local_config)
        print(f"Loaded local config overrides from {local_path.name}")

    return config


def get_data_dir(config: dict) -> Path:
    """Get the resolved data directory path from config.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to the data directory.
    """
    return Path(config["paths"]["data_dir"])


def get_output_dir(config: dict) -> Path:
    """Get the resolved output directory path from config.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to the output directory.
    """
    return Path(config["paths"]["output_dir"])


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary (base is modified in-place and returned).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
