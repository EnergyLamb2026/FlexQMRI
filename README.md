# FlexQMRI

**Neural CDEs for flexible-length quantitative MRI: IVIM and R2\* estimation in the pancreas.**

FlexQMRI estimates quantitative MRI (qMRI) parameters — intravoxel incoherent
motion (IVIM: `f`, `D`, `D*`) and the transverse relaxation rate (R2\*) — from
diffusion- and multi-echo-weighted signals. It compares a Neural Controlled
Differential Equation (NCDE) that natively handles variable-length acquisitions
against fixed-size deep-learning baselines (MLP, Transformer) and a least-squares
(LSQ) fit. This repository reproduces the results from the MICCAI 2026 Workshop OffGrid Submission 
*Neural CDEs for flexible-length quantitative MRI: IVIM and R2\* estimation in the pancreas*.

## Installation

FlexQMRI uses [uv](https://github.com/astral-sh/uv) for environment management.
From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

`uv pip install -e .` reads `pyproject.toml` and installs all dependencies. 

Run every command below from the repository root so the relative config and
results paths resolve correctly.

## Configuration

Configuration is split by stage under [flexqmri/config/](flexqmri/config/):

- [flexqmri/config/training/](flexqmri/config/training/) — synthetic training and
  reproduction. A run merges, in priority order, a model config (`ncde.yaml`,
  `mlp.yaml`, `transformer.yaml`, `lsq.yaml`) over `general.yaml` and a data
  config (`ivim_data.yaml` or `t2star_data.yaml`). 
- [flexqmri/config/inference/](flexqmri/config/inference/) — the in-vivo patient
  pipeline. `common.yaml` holds shared paths and settings and points
  (`paths.config`) to the directory containing the modality files `ivim.yaml`
  and `t2star.yaml`. Each modality file sets the trained model used for
  inference via `paths.*_model_path` and `paths.*_deepmr_config_path`.

All bundled config files are the ones used to generate the results reported in the paper — adjust them to your own data and setup as needed.

## Training and evaluation on simulations

### Train a model on synthetic signals. 
Outputs are saved under `{output_dir}/{modality}/{model}/{global_run_id}/`, where `output_dir` is set
under `paths` in [flexqmri/config/training/general.yaml](flexqmri/config/training/general.yaml).

Train the Neural CDE on IVIM. `train_networks.py` also accepts `mlp.yaml` or
`transformer.yaml`, but then trains only for the single acquisition length set
in the data config:

```bash
python scripts/train/train_networks.py \
    --model_config_path flexqmri/config/training/ncde.yaml \
    --data_config_path flexqmri/config/training/ivim_data.yaml \
    --global_run_id ncde_ivim
```

### Train the fixed-size baselines on every possible length 
One model per acquisition length:

```bash
python scripts/train/train_multi_fixed_models.py \
    --model_config_path flexqmri/config/training/mlp.yaml \
    --data_config_path flexqmri/config/training/ivim_data.yaml
```

### Run the least-squares baseline
No training, just LSQ fitting on the synthetic test set:

```bash
python scripts/train/run_lsq.py \
    --model_config_path flexqmri/config/training/lsq.yaml \
    --data_config_path flexqmri/config/training/ivim_data.yaml
```

Training tracks runs with MLflow when `train.ml_flow_tracking` is enabled in
`general.yaml` — this is the only setting required; the SQLite backend, the
experiment name, and the artifact location are configured automatically. Runs
are logged to a local SQLite store (`mlruns.db` in the repository root), so view
them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db   # run from the repository root
```

## Inference on real data

The patient pipeline loads DICOM/NIfTI data, normalizes and (optionally)
denoises it, segments organs (pancreas, liver or all abdominal organs) with TotalSegmentator, runs the trained
model voxel-wise, and saves the parameter maps. It reports the inference time
per patient. \
`paths.*_model_path` and `paths.*_deepmr_config_path` in the
modality config (`ivim.yaml` / `t2star.yaml`) point at the trained model used
for inference. We set them by default at the bundled NCDE checkpoints that you can download (see
[Data and trained models](#data-and-trained-models)). Point them at another
checkpoint's `.pth`/`config.json` to use a different trained model. 

### Real data layout

Patient data is **not** part of the repository. No real data is released for testing, and the `data/` folder is never committed (see
[.gitignore](.gitignore), which blocks `data/`). Create the `data/` folder
locally and place your input there; the output subfolders are created
automatically. The paths are configured in
[flexqmri/config/inference/common.yaml](flexqmri/config/inference/common.yaml):

```
data/
├── raw/dicom_data/<patient_id>/              # input DICOM series
└── processed/
    ├── nifti_data/<patient_id>/<study>/      # NIfTI (auto-converted from DICOM if absent)
    └── outputs/<patient_id>/<study>/...      # computed maps + ROI stats
```

`<patient_id>` and `<study>` are the `-p` and `-s` arguments below. If the NIfTI
study folder is missing, the pipeline converts the patient's DICOMs into
`nifti_data/` automatically. Adjust `paths.dicom_data` / `paths.nifti_data` /
`paths.output_data` in `common.yaml` to relocate these.

```bash
# IVIM maps (f, D, D*) for one patient and session
python scripts/inference/ivim_patient.py \
    -p P001 -s Study1 -c flexqmri/config/inference/common.yaml

# R2* map for one patient and session
python scripts/inference/r2star_patient.py \
    -p P001 -s Study1 -c flexqmri/config/inference/common.yaml
```

Study1 is given by default by the dicom to nifti conversion script, but you can change it to any string. 

## Reproducing the results
### Model robustness to variable-length acquisitions
Quantify how fixed-size models degrade under a size-adjustment strategy
(undersampling or interpolation) while the NCDE stays robust (Table 1):

```bash
python scripts/reproduce/test_variable_length.py \
    --experiment_config_path flexqmri/config/training/variable_length_ivim_experiment.yaml
```

### Compare trained models 
Compare per-parameter error and computational cost across trained runs (Tables 2–3):

```bash
python scripts/reproduce/compare_trained_models.py \
    -c flexqmri/config/training/ncde.yaml \
    --data_config_path flexqmri/config/training/ivim_data.yaml \
    --global_run_ids trf_ivim mlp_ivim transformer_ivim ncde_ivim \
    --name ivim_comparison
```

If a discovered run folder holds a trained model (or an LSQ config) but no
`metrics.pt`, the script regenerates it automatically: it recreates the test set
from the folder's `config.json` and re-runs the evaluation before comparing. This
lets you compare downloaded checkpoints that ship without saved metrics.

### Compare model predictions on real data

```bash
# Distributions and clinical summary table across models (Fig. 3)
python scripts/inference/compare_real_values.py \
    -p P001 P002 P003 -s1 Study1 -s2 Study2 \
    -c flexqmri/config/inference/common.yaml
```

### Assess intra-patient repeatability across two sessions

```bash
python scripts/inference/intra_repeat.py \
    -p P001 P002 P003 --maps_name ivim --model_name ncde \
    -c flexqmri/config/inference/common.yaml --run_pipeline
```

`--run_pipeline` runs `ivim_patient.py` (or `r2star_patient.py`, depending on
`--maps_name`) for each patient and session before computing the ICC.

## Data and trained models

The full set of trained checkpoints used in the paper (all models and
acquisition lengths) and the acquisition offset files
(`resources/IVIMbvalues.txt`, `resources/T2starTEvalues.txt`) are hosted on
Zenodo:

> **Zenodo:** https://doi.org/10.5281/zenodo.21075181

After downloading:

- unpack the checkpoints under `results/`, then point the relevant config paths
  at them — `paths.*_model_path` in the inference configs, or the `model_path`
  entries in the reproduction experiment YAML. The bundled configs expect one
  descriptive run-id folder per model under `results/{modality}/{model}/`
  (e.g. `results/ivim/ncde/ncde_ivim/`) — rename the unpacked run folders to
  match, or edit the paths to match the archive's folder names instead;
- place the offset files under `resources/` (referenced by `offset_path` in the
  inference configs).

## Repository structure

```
FlexQMRI/
├── flexqmri/                 # installable package
│   ├── networks/             # NCDE, MLP, Transformer
│   ├── dataset/              # synthetic data generation + DataLoaders
│   ├── models/               # supervised training loop, LSQ fitting
│   ├── evaluation/           # metrics, variable-length eval, tables, and all plots
│   ├── imaging/              # patient DICOM/NIfTI loading, structures, saving
│   ├── preprocessing/        # normalization, denoising, alignment
│   ├── compute_maps/         # IVIM / R2* map computation
│   ├── analysis/             # ROI statistics, repeatability (ICC)
│   ├── convert/              # DICOM → NIfTI
│   ├── utils/                # config parsing, biophysical models, I/O
│   └── config/
│       ├── training/         # synthetic training + reproduction configs
│       └── inference/        # patient pipeline configs
├── scripts/
│   ├── train/                # train_networks, train_multi_fixed_models, run_lsq
│   ├── reproduce/            # test_variable_length, compare_trained_models
│   └── inference/            # ivim_patient, r2star_patient, compare_real_values, intra_repeat
├── data/                     # data for inference (git-ignored)
├── resources/                # acquisition offset files (b-values, echo times) (git-ignored)
├── results/                  # outputs + model checkpoints (git-ignored)
├── pyproject.toml
└── LICENSE
```

Patient data, large outputs, and full checkpoints are not tracked by git for security reasons; see
[.gitignore](.gitignore).

## License

FlexQMRI is released under the [MIT License](LICENSE).
