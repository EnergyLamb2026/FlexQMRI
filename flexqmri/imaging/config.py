'''Functions for loading a config file (.yaml)'''

import yaml
import numpy as np 
from copy import deepcopy
import os

def _deep_merge_dicts(base: dict, override: dict) -> dict:
    '''Recursively merge override dict into base dict.
    
    override values take precedence over base values.
    Nested dicts are merged recursively.
    
    Parameters:
    base (dict): base configuration dictionary
    override (dict): override configuration dictionary
    
    Returns:
    dict: merged configuration dictionary
    '''
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # Override with new value
            result[key] = deepcopy(value)
    
    return result


def load_config(config_path: str) -> dict:
    '''Load a yaml config file and return the content as a dictionary.

    Parameters:
    config_path (str): path to the yaml config file

    Returns:
    dict: content of the yaml config file
    '''

    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def load_config_hierarchical(modality: str, common_config_path: str) -> dict:
    '''Load configuration hierarchically: common -> modality-specific.

    Loads the common config, then merges the modality file
    (``<paths.config>/<modality>.yaml``) on top of it.

    Parameters:
    modality (str): modality name ('ivim', 't2star')
    common_config_path (str): path to the common config file

    Returns:
    dict: merged configuration dictionary
    '''
    config = load_config(common_config_path)
    if config is None:
        raise FileNotFoundError(f"Could not load common config: {common_config_path}")

    # Load modality-specific configuration
    modality_path = os.path.join(config['paths']['config'], f'{modality}.yaml')
    modality_config = load_config(modality_path)
    
    if modality_config is not None:
        config = _deep_merge_dicts(config, modality_config)
    else:
        raise FileNotFoundError(f"Could not load modality config: {modality_path}")
    
    return config        


def read_offsets_from_txt(offset_path: str) -> list:
    '''Read the offset values for qMRI from the given file.

    Parameters:
    offset_path (str): path to the file containing the offset values

    Returns:
    list: list of offset values
    '''
    with open(offset_path, 'r') as f:
        return np.array([float(line.strip()) for line in f])
    
