'''Functions to parse command line arguments.'''

import argparse

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training and evaluation scripts.

    Returns:
        argparse.Namespace: Parsed arguments containing `model_config_path`
            and `data_config_path`.
    """
    parser = argparse.ArgumentParser(description='Train or evaluate regression models on synthetic MRI data.')
    
    # Common arguments
    parser.add_argument('--model_config_path', type=str, required=True, help='Path to the model configuration YAML file.')
    parser.add_argument('--data_config_path', type=str, required=True, help='Path to the data configuration YAML file (default: config/ivim_data.yaml).')
    parser.add_argument('--global_run_id', type=str, default=None, help='Fixed run ID for grouping results (auto-generated if not set).')

    return parser.parse_args()


def parse_variable_length_args() -> argparse.Namespace:
    """Parse command line arguments for the variable-length robustness experiment.

    Returns:
        argparse.Namespace: Parsed arguments containing `experiment_config_path`.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate NCDE robustness under undersampling/interpolation vs fixed-length models.'
    )
    parser.add_argument(
        '--experiment_config_path', type=str, required=True,
        help='Path to the variable-length experiment YAML file.'
    )
    return parser.parse_args()


def parse_compare_run_args() -> argparse.Namespace:
    """Parse command line arguments for the compare_experiments script.

    Returns:
        argparse.Namespace: Parsed arguments containing `global_run_ids`, `name`,
            `model_config_path`, and `data_config_path`.
    """
    parser = argparse.ArgumentParser(description='Compare results from multiple fitting/training experiments.')
    parser.add_argument('-c', '--model_config_path', type=str, required=True, help='Path to the model configuration YAML file.')
    parser.add_argument('--data_config_path', type=str, required=True, help='Path to the data configuration YAML file.')
    parser.add_argument('--global_run_ids', type=str, nargs='+', required=True, help='List of global run IDs to compare.')
    parser.add_argument('--name', type=str, required=True, help='Name for this comparison run, used as a subdirectory and filename prefix.')
    return parser.parse_args()
