"""Evaluate the variable-length NCDE against fixed-length MLP/Transformer
baselines under irregular acquisition sampling.

Two data-generation modes are selected by the experiment YAML:

Fixed x-values (IVIM): ``test_x_values`` lists the shared acquisition points.
  All samples use those exact values.  ``x_values_5`` / ``x_values_10``
  specify the subset/superset for undersample/interpolate.

Variable-length (T2*): ``test_min_x_length`` / ``test_max_x_length`` define
  the per-sample length range; x-values are drawn from the training
  distribution.  Undersample takes the first ``fixed_length`` TEs.
  Interpolate uses ``n_x_interpolate`` equally-spaced points spanning each
  sample's own TE range (no extrapolation).

The NCDE takes the raw signal directly (``adapt: none``) since it handles
irregular lengths natively. The first such entry's config.json is used as the
base configuration for data generation.
"""

import copy
from pathlib import Path

import numpy as np
import torch

from flexqmri.dataset.synthetic import SynthIVIM, SynthT2Star
from flexqmri.evaluation import utils as eval_utils
from flexqmri.evaluation.tables import build_summary_table
from flexqmri.evaluation.variable_length import (
    interpolate_signals,
    make_test_loader,
    undersample_signals,
)
from flexqmri.networks import utils as net_utils
from flexqmri.utils import parse as parse_utils
from flexqmri.utils.config import load_config_for_model, load_yaml
from flexqmri.utils.utils import set_seed

_SYNTH_CLASS = {
    "ivim": SynthIVIM,
    "t2star": SynthT2Star,
}


def _build_data_gen_config(base_config: dict, exp: dict) -> dict:
    """Override a model config to generate test data.

    Two modes depending on the experiment YAML:
    - ``test_x_values`` present: fixed acquisition points, all samples identical.
    - ``test_min_x_length`` / ``test_max_x_length`` present: variable-length,
      x-values drawn from the training distribution.

    Args:
        base_config: Full config loaded from the NCDE model's config.json.
        exp: The 'experiment' section of the experiment YAML.

    Returns:
        Config dict ready for synthetic data generation.
    """
    config = copy.deepcopy(base_config)
    sim = config["data"]["simulation"]
    sim["n_samples"] = exp["n_test_samples"]
    sim["train_val_test_split"] = [0.0, 0.0, 1.0]
    sim["load"] = False
    sim["save"] = False

    if "test_x_values" in exp:
        n = len(exp["test_x_values"])
        sim["x"] = exp["test_x_values"]
        sim["min_x_length"] = n
        sim["max_x_length"] = n
        config["data"]["fixed_length"] = n
    else:
        sim["min_x_length"] = exp["test_min_x_length"]
        sim["max_x_length"] = exp["test_max_x_length"]
        config["data"]["fixed_length"] = 0

    return config


if __name__ == "__main__":
    args = parse_utils.parse_variable_length_args()
    exp_yaml = load_yaml(args.experiment_config_path)
    exp = exp_yaml["experiment"]

    test_x_values = exp.get("test_x_values")
    x_values_5 = exp.get("x_values_5")
    x_values_10 = exp.get("x_values_10")
    modality = exp.get("modality", "ivim")
    SynthClass = _SYNTH_CLASS[modality]

    # Use the first variable-length model's (adapt: none) config.json as data generation base
    var_length_entry = next(m for m in exp["models"] if m.get("adapt", "none") == "none")
    base_config = load_config_for_model(var_length_entry["model_path"])

    # Generate shared test data
    data_gen_config = _build_data_gen_config(base_config, exp)
    generator = torch.Generator()
    generator.manual_seed(exp.get("seed", 42))
    set_seed(exp.get("seed", 42))

    if test_x_values is not None:
        print(f"Generating {exp['n_test_samples']} test samples at x-values: {test_x_values}")
    else:
        print(f"Generating {exp['n_test_samples']} test samples with "
              f"{exp['test_min_x_length']}-{exp['test_max_x_length']} random TEs "
              f"from training distribution")
    X, y, noise = SynthClass(data_gen_config).generate_data(generator)

    print("\n" + "=" * 70)
    print("VARIABLE-LENGTH ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    all_results = {}

    for entry in exp["models"]:
        label = entry["label"]
        adapt = entry.get("adapt", "none")
        model_path = entry["model_path"]

        model_config = load_config_for_model(model_path)
        if "fixed_length" in entry:
            model_config["data"]["fixed_length"] = entry["fixed_length"]
        model = net_utils.load_model(model_path, model_config)

        if adapt == "none":
            X_adapted = X
        elif adapt == "undersample":
            if x_values_5 is not None:
                X_adapted = undersample_signals(X, test_x_values, x_values_5)
            else:
                X_adapted = X[:, :entry["fixed_length"], :]
        elif adapt == "interpolate":
            if x_values_10 is not None:
                X_adapted = interpolate_signals(X, target_x_values=x_values_10)
            else:
                X_adapted = interpolate_signals(X, target_n=exp.get("n_x_interpolate", 10))
        else:
            raise ValueError(f"Unknown adapt value: '{adapt}'. Use 'none', 'undersample', or 'interpolate'.")

        test_loader = make_test_loader(X_adapted, y, noise, model_config, generator)
        results = eval_utils.test_network(model_config, test_loader, model, modality)

        all_results[label] = results

        print(f"\n--- {label} ---")
        eval_utils.print_fit_results(results, modality)

    parameters = eval_utils.get_modality_parameters(modality)
    seed = exp.get("seed", 42)
    header = f"{'Model':<35}" + "".join(f"{p:>12}" for p in parameters)
    sep = "-" * len(header)

    print("\n" + "=" * 70)
    print("SUMMARY (NRMSE per parameter)")
    print("=" * 70)
    print(header)
    print(sep)
    for label, results in all_results.items():
        row = f"{label:<35}"
        for p in parameters:
            nrmse = float(np.sqrt(np.mean(torch.stack(results[f'{p}_re']).cpu().numpy())))
            row += f"{nrmse:>12.4e}"
        print(row)
        ci = eval_utils.compute_results_bootstrap_ci(results, modality, seed=seed)
        ci_str = "  ".join(f"{p}: [{ci[p]['ci_lower']:.2e}, {ci[p]['ci_upper']:.2e}]" for p in parameters)
        print(f"  95% CI  {ci_str}")

    name = Path(args.experiment_config_path).stem
    output_dir = Path(base_config["paths"]["output_dir"]) / modality / "variable_length" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_summary_table(all_results, modality, metric="nrmse")

    csv_path = output_dir / f"{name}_nrmse_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

