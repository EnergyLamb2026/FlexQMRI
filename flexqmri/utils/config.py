'''Functions to load configuration files.'''

import argparse
import yaml
from pathlib import Path
from copy import deepcopy
import json

def load_yaml(filepath: str) -> dict:
    """Load a YAML configuration file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary parsed from the YAML file.

    Raises:
        FileNotFoundError: If `filepath` does not point to an existing file.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Merge multiple configuration dictionaries recursively. Later configs override earlier ones.

    Recursively merges nested dicts so that deeply nested keys are preserved
    rather than overwritten by a shallow update.

    Args:
        *configs (dict): Variable number of configuration dictionaries to merge.

    Returns:
        dict: Merged configuration dictionary.
    """
    result = {}
    for config in configs:
        if config:
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = deepcopy(value)
    return result


def load_config(model_config_path: str, data_config_path: str = 'config/ivim_data.yaml') -> dict:
    """Load and merge configuration from base files and a model-specific file.

    Loads `general.yaml` (which includes default paths) from the same
    directory as `model_config_path`, then merges all configs in priority
    order: data < general < model-specific.

    Args:
        model_config_path (str): Path to the model configuration YAML file
            (e.g., ``config/ncde.yaml``, ``config/mlp.yaml``).
        data_config_path (str, optional): Path to the data configuration YAML
            file. Defaults to ``'config/ivim_data.yaml'``.

    Returns:
        dict: Complete merged configuration dictionary.

    Raises:
        FileNotFoundError: If any of the expected YAML files do not exist.
    """
    # extract config directory from model_config_path
    config_dir = Path(model_config_path).parent

    # Load base configurations
    data_config = load_yaml(str(Path(data_config_path)))
    model_config = load_yaml(str(Path(model_config_path)))

    general_config = load_yaml(str(config_dir / 'general.yaml'))

    # Merge in order: data, general defaults (incl. paths), model-specific
    config = merge_configs(data_config, general_config, model_config)

    return config


def load_and_validate_config(args: argparse.Namespace) -> tuple:
    """Load and validate the configuration from parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed arguments from `parse_args`, expected
            to contain `model_config_path` and `data_config_path`.

    Returns:
        tuple: A three-element tuple ``(config, seed_nbr, base_seed)`` where
            ``config`` is the merged configuration dict, ``seed_nbr`` is the
            number of seed runs, and ``base_seed`` is the base random seed.

    Raises:
        ValueError: If `model_config_path` or `data_config_path` is not set.
    """
    
    config = load_config(args.model_config_path, args.data_config_path)
    
    seed_nbr = config["train"]["seed_nbr"]
    base_seed = config["train"]["seed"]
    
    print("Configuration loaded successfully")
    print(f"Number of seed runs: {seed_nbr}")
    
    return config, seed_nbr, base_seed

def determine_data_type(config: dict) -> str:
    """Determine the data type from the configuration.

    Args:
        config (dict): Merged configuration dictionary. Reads
            ``config['data']['simulation']['n_samples']`` to decide the type.

    Returns:
        str: Data type string. Currently always ``'simulation'`` when
            ``n_samples > 0``.

    Raises:
        ValueError: If no valid data type can be determined (e.g., `n_samples`
            is zero or missing).
    """

    if config['data']['simulation']['n_samples'] > 0: 
        data_type  = 'simulation'
    else: 
        raise ValueError("Invalid data configuration: no samples available.")
    
    return data_type


def load_config_for_model(model_path: str) -> dict:
    """Load the ``config.json`` saved alongside a model file.

    Args:
        model_path (str): Path to the model ``.pth`` file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If ``config.json`` is not found.
    """
    config_path = Path(model_path).parent / 'config.json'

    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")