"""I/O utilities for model saving, loading, and path management."""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


def get_model_path(
    model_type: str,
    modality: str,
    run_id: str,
    fixed_length: Optional[int] = None,
    filename: str = 'model.pth',
    global_run_id: Optional[str] = None
) -> Path:
    """Get the organized path for a trained model without saving.

    Creates hierarchical path structure:
    ``results/{modality}/{model_type}/{global_run_id}/{run_id}/{filename}``

    Args:
        model_type (str): Type of model (e.g., 'ncde', 'mlp').
        modality (str): Data modality (e.g., 'ivim', 't2star').
        run_id (str): Specific run identifier (unique for each seed).
        fixed_length (int, optional): If provided, appends to filename.
        filename (str): Base filename (default: 'model.pth').
        global_run_id (str, optional): Parent run ID for grouping runs.

    Returns:
        Path: Full path to the model file (directory is created).

    Raises:
        ValueError: If *global_run_id* is ``None``.
    """
    if global_run_id is None:
        raise ValueError("global_run_id must be provided (use experiment name, timestamp, or experiment ID)")

    model_dir = Path('results') / modality / model_type / global_run_id / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    if fixed_length is not None:
        name_parts = filename.split('.')
        final_filename = f"{name_parts[0]}_{fixed_length}.{name_parts[1]}"
    else:
        final_filename = filename

    return model_dir / final_filename


def discover_model_paths(
    global_run_id: str,
    model_type: str,
    modality: str = 'ivim'
) -> List[Tuple[int, str, str]]:
    """Discover all model paths for a given global_run_id.

    Args:
        global_run_id (str): Global run ID.
        model_type (str): Type of model (e.g., 'ncde', 'mlp').
        modality (str): Modality (default: 'ivim').

    Returns:
        List[tuple]: List of (seed_index, spec_run_id, model_path) tuples.

    Raises:
        FileNotFoundError: If the global run directory does not exist.
    """
    base_path = Path('results') / modality / model_type / global_run_id

    if not base_path.exists():
        raise FileNotFoundError(f"Global run directory not found: {base_path}")

    model_paths = []

    for spec_run_dir in sorted(base_path.iterdir()):
        if not spec_run_dir.is_dir():
            continue

        spec_run_id = spec_run_dir.name

        model_file = spec_run_dir / 'model.pth'
        if not model_file.exists():
            pth_files = list(spec_run_dir.glob('model_*.pth'))
            if pth_files:
                model_file = pth_files[0]
            else:
                print(f"Warning: No model file found in {spec_run_dir}")
                continue

        seed_index = len(model_paths)
        model_paths.append((seed_index, spec_run_id, str(model_file)))

    return model_paths

