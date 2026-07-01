"""
Supervised training module for deep regression models.

Uses PyTorch Lightning to handle the training loop, checkpointing,
early stopping, logging, and device management.
"""
import copy
import datetime
import json
import torch
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

try:
    from lightning.pytorch.loggers import MLFlowLogger
    MLFLOW_LOGGER_AVAILABLE = True
except ImportError:
    MLFLOW_LOGGER_AVAILABLE = False

from flexqmri.networks import utils as net_utils
from flexqmri.evaluation import training_plots as analysis_plots
from flexqmri.evaluation import training_plots as analysis_gradients
from flexqmri.evaluation import utils as eval_utils
from flexqmri.evaluation import evaluate
from flexqmri.utils import biophysical_model
from flexqmri.utils import parse as parse_utils
from flexqmri.utils import config as config_utils
from flexqmri.utils.utils import make_serializable, set_seed
from flexqmri.utils.io import get_model_path
from flexqmri.dataset import get_dataset_loaders
from flexqmri.dataset.datamodule import MRIDataModule

# Legacy alias -- some scripts (e.g. train_cval_networks) toggle this directly
MLFLOW_AVAILABLE = MLFLOW_LOGGER_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class BestModelCheckpoint(L.Callback):
    """Save the best model ``state_dict`` as a ``.pth`` file.

    Unlike Lightning's built-in ``ModelCheckpoint`` (which saves ``.ckpt``),
    this writes only the raw ``state_dict`` so that existing model-loading
    code keeps working unchanged.
    """

    def __init__(self, save_path: str, monitor: str = 'val_loss'):
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.best_score = float('inf')
        self.best_epoch = -1
        self._ready = False          # skip sanity-check validation

    def on_train_start(self, trainer, pl_module):
        self._ready = True

    def _check_and_save(self, trainer, pl_module):
        if not self._ready:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is not None and current < self.best_score:
            self.best_score = current.item()
            self.best_epoch = trainer.current_epoch
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pl_module.model.state_dict(), self.save_path)
            logger.info(
                f'Saved best model (epoch {self.best_epoch}, '
                f'{self.monitor}={self.best_score:.6g}) to {self.save_path}'
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.monitor != 'train_loss':
            self._check_and_save(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.monitor == 'train_loss':
            self._check_and_save(trainer, pl_module)


class GradientHistogramCallback(L.Callback):
    """Log a gradient histogram to MLflow at the end of training.

    Stores the last training batch, then on ``on_train_end`` runs a real
    forward+backward pass to obtain actual parameter gradients and logs
    the histogram PNG as an MLflow artifact.  Only active when
    ``pil`` is enabled and the trainer logger is an ``MLFlowLogger``.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._last_batch = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Keep a reference to the most recent training batch."""
        self._last_batch = batch

    def on_train_end(self, trainer, pl_module):
        if self._last_batch is None:
            return
        if not (MLFLOW_LOGGER_AVAILABLE and isinstance(trainer.logger, MLFlowLogger)):
            return
        try:
            batch = self._last_batch
            device = pl_module.device
            y_pred_coeffs, y_true_coeffs, y_pred_params, y_true_params, b_values, _ = (
                eval_utils.get_net_outputs(
                    pl_module.config, batch, pl_module.model, device,
                    atol=pl_module.atol, rtol=pl_module.rtol,
                )
            )
            supervised_loss = pl_module.criterion(y_pred_coeffs, y_true_coeffs)
            phys_inf_loss = fast_physics_informed_loss(
                y_pred_params, y_true_params, b_values,
                pl_module.var_length, pl_module.param_model,
            )
            optimizer = pl_module.optimizers()
            grad_plot_path = self.model_dir / 'grad_histogram.png'
            analysis_plots.plot_grad_histogram(
                pl_module.model,
                optimizer,
                supervised_loss,
                phys_inf_loss,
                save_path=str(grad_plot_path),
            )
            trainer.logger.experiment.log_artifact(
                trainer.logger.run_id, str(grad_plot_path),
            )
            logger.info(f'Logged gradient histogram to MLflow from {grad_plot_path}')
        except Exception as exc:
            logger.warning(f'Gradient histogram callback failed: {exc}')


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class SupervisedRegressionModule(L.LightningModule):
    """Lightning module for supervised deep MRI regression.

    Wraps any network from the ``networks`` package and handles:
    - Training with supervised + optional physics-informed loss
    - Validation
    - Optimizer / LR-scheduler configuration
    - Adaptive ODE tolerance scheduling (for NCDE)
    - Loss-balancing lambda updates
    """

    def __init__(self, config: dict, modality: str = 'ivim'):
        """
        Args:
            config: Full experiment configuration dictionary.
            modality: MRI modality (``'ivim'`` or ``'t2star'``).
        """
        super().__init__()
        self.config = config
        self.modality = modality
        self.param_model = config['data']['param_model']
        self.model_name = config['train']['model']
        self.var_length = self.model_name in ('ncde',)

        # ---- Network (Lightning handles device placement) ----
        self.model = net_utils.net_factory(config)

        # ---- Loss ----
        if config['train']['uncertainty_weighted']:
            n_params = biophysical_model.get_model(config['data']['param_model']).n_params
            self.criterion = MultiParamUncertaintyLoss(n_params)
        else:
            self.criterion = torch.nn.MSELoss()

        # Physics-informed loss settings
        self.use_pil = config['train']['pil']
        self.lambda_sup = 1.0
        self.lambda_phys_inf = 1.0

        # Adaptive ODE solver tolerances
        if self.model_name == 'ncde' and config['train']['adaptive']:
            self.atol = 1e-5
            self.rtol = 1e-3
        else:
            self.atol = None
            self.rtol = None

        # ---- Epoch-level loss history (for plotting) ----
        self.train_loss_history: list = []
        self.val_loss_history:   list = []
        self.sup_loss_history:   list = []
        self.phys_loss_history:  list = []
        self.uncertainties:      list = []

        # Step-level accumulators (reset each epoch)
        self._step_train = 0.0
        self._step_sup   = 0.0
        self._step_phys  = 0.0
        self._step_count = 0

        self._step_val       = 0.0
        self._step_val_count = 0

        # Global step counter for lambda updates
        self._global_train_step = 0

    # ---- Forward ----

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ---- Training ----

    def training_step(self, batch, batch_idx):
        y_pred_coeffs, y_true_coeffs, y_pred_params, y_true_params, b_values, _ = (
            eval_utils.get_net_outputs(
                self.config, batch, self.model, self.device,
                atol=self.atol, rtol=self.rtol,
            )
        )

        # Physics-informed loss
        if self.use_pil:
            phys_inf_loss = fast_physics_informed_loss(
                y_pred_params, y_true_params, b_values,
                self.var_length, self.param_model,
            )
        else:
            phys_inf_loss = torch.tensor(0.0, device=self.device)

        supervised_loss = self.criterion(y_pred_coeffs, y_true_coeffs)
        loss = self.lambda_sup * supervised_loss + self.lambda_phys_inf * phys_inf_loss

        # Track uncertainty weights
        if (self.config['train']['uncertainty_weighted']
                and isinstance(self.criterion, MultiParamUncertaintyLoss)):
            self.uncertainties.append(
                self.criterion.get_uncertainties().detach().cpu().numpy()
            )

        # Update loss-balancing lambdas
        freq = self.config['train']['lambda_update_frequency']
        if freq > 0 and self._global_train_step % freq == 0:
            self.lambda_sup, self.lambda_phys_inf = update_lambda(
                self.model, self.optimizers(), supervised_loss, phys_inf_loss,
                self.lambda_sup, self.lambda_phys_inf,
                self.config['train']['lambda_update_factor'],
            )
        self._global_train_step += 1

        # Lightning logging
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_sup_loss', supervised_loss, on_step=False, on_epoch=True)
        self.log('phys_inf_loss', phys_inf_loss, on_step=False, on_epoch=True)
        self.log('lambda_sup', float(self.lambda_sup), on_step=False, on_epoch=True)
        self.log('lambda_phys_inf', float(self.lambda_phys_inf), on_step=False, on_epoch=True)

        # Manual accumulation for loss history
        self._step_train += loss.item()
        self._step_sup   += supervised_loss.item()
        self._step_phys  += phys_inf_loss.item()
        self._step_count += 1

        return loss

    def on_train_epoch_end(self):
        n = max(self._step_count, 1)
        self.train_loss_history.append(self._step_train / n)
        self.sup_loss_history.append(self._step_sup / n)
        self.phys_loss_history.append(self._step_phys / n)
        self._step_train = 0.0
        self._step_sup   = 0.0
        self._step_phys  = 0.0
        self._step_count = 0

        # Adaptive ODE tolerance decay (NCDE only)
        if (self.model_name == 'ncde'
                and self.config['train']['adaptive']
                and self.current_epoch > 0
                and self.current_epoch % 50 == 0):
            self.atol /= 2
            self.rtol /= 2
            logger.info(
                f'Updated atol={self.atol}, rtol={self.rtol} '
                f'at epoch {self.current_epoch}'
            )

    # ---- Validation ----

    def validation_step(self, batch, batch_idx):
        y_pred_coeffs, y_true_coeffs, _, _, _, _ = eval_utils.get_net_outputs(
            self.config, batch, self.model, self.device,
        )
        val_loss = self.criterion(y_pred_coeffs, y_true_coeffs)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if not self.trainer.sanity_checking:
            self._step_val += val_loss.item()
            self._step_val_count += 1
        return val_loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        if self._step_val_count > 0:
            self.val_loss_history.append(self._step_val / self._step_val_count)
        self._step_val = 0.0
        self._step_val_count = 0

    # ---- Optimizer / scheduler ----

    def configure_optimizers(self):
        lr = self.config['train']['general_lr']

        if self.model_name == 'ncde':
            params = set_last_layer_lr(self.model, self.config)
        else:
            params = [{'params': self.model.parameters(), 'lr': lr}]

        if (self.config['train']['uncertainty_weighted']
                and isinstance(self.criterion, MultiParamUncertaintyLoss)):
            params.append({'params': self.criterion.eta, 'lr': 10 * lr})

        optimizer = torch.optim.Adam(
            params, lr=lr,
            weight_decay=self.config['train']['weight_decay'],
        )
        scheduler = ExponentialLR(
            optimizer, gamma=self.config['train']['lr_scheduler_gamma'],
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   # step per batch (matches original behaviour)
            },
        }


# ---------------------------------------------------------------------------
# Convenience entry-point (same return signature as the legacy function)
# ---------------------------------------------------------------------------

def train_network(
    config: dict,
    train_loader: torch.utils.data.DataLoader = None,
    val_loader: torch.utils.data.DataLoader = None,
    global_run_id: str = None,
    run_id: str = None,
    modality: str = 'ivim',
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    run_description: Optional[str] = None,
    use_mlflow: bool = True,
    *,
    datamodule: L.LightningDataModule = None,
) -> Tuple[torch.nn.Module, Optional[float], Optional[float], Path]:
    """Train a neural network with PyTorch Lightning.

    Accepts **either** explicit ``train_loader`` / ``val_loader`` (backward-
    compatible) **or** a Lightning ``datamodule``.

    Returns:
        Tuple of ``(model, atol, rtol, model_dir)`` -- identical to the
        previous non-Lightning interface so downstream callers need no
        changes.
    """
    if global_run_id is None:
        raise ValueError("global_run_id must be provided")
    if run_id is None:
        raise ValueError("run_id must be provided")

    # ---- Lightning module ----
    module = SupervisedRegressionModule(config, modality)

    # ---- Model directory & best-model path ----
    model_type = config['train']['model'].lower()
    fixed_length = config['data'].get('fixed_length', 0)
    best_model_path = get_model_path(
        model_type, modality, run_id, fixed_length,
        'best_model.pth', global_run_id,
    )
    model_dir = best_model_path.parent

    # ---- Determine monitor metric ----
    if datamodule is not None:
        datamodule.setup()
        has_val = datamodule.val_dataloader() is not None
    else:
        has_val = val_loader is not None
    monitor = 'val_loss' if has_val else 'train_loss'

    # ---- Callbacks ----
    ckpt_callback = BestModelCheckpoint(
        save_path=str(best_model_path),
        monitor=monitor,
    )
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=config['train']['patience'],
            mode='min',
        ),
        ckpt_callback,
        LearningRateMonitor(logging_interval='step'),
    ]
    if config['train']['pil']:
        callbacks.append(GradientHistogramCallback(model_dir))

    # ---- Logger ----
    pl_logger = True                     # default CSV logger
    if use_mlflow and MLFLOW_LOGGER_AVAILABLE:
        tags = {}
        if run_description:
            tags['mlflow.note.content'] = run_description
        pl_logger = MLFlowLogger(
            experiment_name=experiment_name or f'Training_{model_type.upper()}',
            run_name=run_name,
            tracking_uri='sqlite:///mlruns.db',
            artifact_location='results/mlflow_artifacts',
            tags=tags,
        )

    # ---- Trainer ----
    trainer = L.Trainer(
        max_steps=config['train']['max_iter'],
        callbacks=callbacks,
        logger=pl_logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    logger.info(
        f'Starting Lightning training (max_steps={config["train"]["max_iter"]})'
    )

    if datamodule is not None:
        trainer.fit(module, datamodule=datamodule)
    else:
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ---- Reload best weights ----
    if ckpt_callback.save_path.exists():
        best_state = torch.load(
            ckpt_callback.save_path, map_location='cpu', weights_only=True,
        )
        module.model.load_state_dict(best_state)
        logger.info(f'Loaded best model from epoch {ckpt_callback.best_epoch}')

    # ---- Save config JSON ----
    config_path = model_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(make_serializable(config), f, indent=2)
    logger.info(f'Saved config to {config_path}')

    # ---- Loss plot (MLflow only) ----
    if MLFLOW_LOGGER_AVAILABLE and isinstance(pl_logger, MLFlowLogger):
        analysis_plots.plot_losses(
            config,
            module.train_loss_history,
            module.val_loss_history,
            module.sup_loss_history,
            module.phys_loss_history,
        )
        loss_plot_path = model_dir / 'loss_plot.png'
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        pl_logger.experiment.log_artifact(pl_logger.run_id, str(loss_plot_path))
        logger.info(f'Logged loss plot to MLflow from {loss_plot_path}')

    logger.info('Training complete.')
    return module.model, module.atol, module.rtol, model_dir


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

class MultiParamUncertaintyLoss(torch.nn.Module):
    """
    Custom loss function that accounts for uncertainty in multiple parameters.

    Implements automatic parameter-wise uncertainty weighting where each parameter
    has its own learnable uncertainty (log-variance) parameter.
    """

    def __init__(self, n_params: int):
        """
        Initialize the uncertainty loss.

        Args:
            n_params (int): Number of output parameters
        """
        super(MultiParamUncertaintyLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.eta = torch.nn.Parameter(torch.ones(n_params))
        self._per_param_loss = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-parameter uncertainty-weighted loss.

        Loss = sum_i [ (y_pred_i - y_true_i)^2 * exp(-eta_i) + eta_i ]

        Args:
            y_pred (torch.Tensor): Predicted values, shape (batch_size, n_params)
            y_true (torch.Tensor): True values, shape (batch_size, n_params)

        Returns:
            torch.Tensor: Scalar loss value
        """
        residuals = self.loss_fn(y_pred, y_true).mean(dim=0)
        self._per_param_loss = residuals * torch.exp(-self.eta) + self.eta
        return self._per_param_loss.mean()

    def get_per_param_loss(self) -> torch.Tensor:
        if self._per_param_loss is None:
            raise RuntimeError("No losses computed yet. Call forward() first.")
        return self._per_param_loss

    def get_uncertainties(self) -> torch.Tensor:
        return torch.exp(0.5 * self.eta)


def set_last_layer_lr(model: torch.nn.Module, config: dict) -> list:
    """
    Set different learning rates for the last layer of NCDE encoder.

    Args:
        model (torch.nn.Module): The model used for training
        config (dict): Configuration dictionary with learning rate factors

    Returns:
        list: List of parameter groups with different learning rates
    """
    params = []
    base_params = []
    last_layer_params = []

    readout_indices = [
        int(name.split('.')[2])
        for name, _ in model.named_parameters()
        if 'readout' in name and len(name.split('.')) > 2
    ]
    if not readout_indices:
        raise ValueError(
            "No 'readout' layers found in model parameters. "
            "Cannot assign last-layer learning rate for NCDE."
        )
    last_layers_idx = readout_indices[-1]

    for name, param in model.named_parameters():
        if 'encoder' in name:
            parts = name.split('.')
            if len(parts) < 3:
                base_params.append(param)
                continue
            layer_index = int(parts[2])

            if layer_index == last_layers_idx:
                last_layer_params.append(param)
            else:
                base_params.append(param)
        else:
            base_params.append(param)

    params.append({'params': base_params, 'lr': config["train"]["general_lr"]})
    params.append({'params': last_layer_params, 'lr': config["train"]["general_lr"] * config["train"]["last_layer_lr_factor"]})

    return params


def fast_physics_informed_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    b_values: torch.Tensor,
    var_length: bool = True,
    param_model: str = 'ivim_bi_exp'
) -> torch.Tensor:
    """
    Compute physics-informed loss based on the physical forward model.

    Args:
        y_pred: Predicted parameters
        y_true: True parameter values
        b_values: Measurement offsets (b-values / TEs)
        var_length: Whether the model uses variable-length input
        param_model: Registry model name (e.g. ``'ivim_bi_exp'``)

    Returns:
        torch.Tensor: Physics-informed loss (scalar)
    """
    if var_length:
        b_values[torch.isnan(b_values)] = 0

    s_pred = biophysical_model.apply_phys_model(b_values, param_model, y_pred, torch_based=True, training=True)
    s_true = biophysical_model.apply_phys_model(b_values, param_model, y_true, torch_based=True, training=True)

    if var_length:
        s_pred[torch.isnan(s_pred)] = 0

    return torch.nn.functional.mse_loss(s_pred, s_true)


def update_lambda(
    model: torch.nn.Module,
    optimizer,
    supervised_loss: torch.Tensor,
    phys_inf_loss: torch.Tensor,
    current_lambda_sup: float,
    current_lambda_phys_inf: float,
    alpha: float
) -> Tuple[float, float]:
    """
    Update loss weights based on gradient magnitudes.

    Uses exponential moving average to balance supervised and physics-informed
    losses based on their relative gradient magnitudes.

    Args:
        model: The model
        optimizer: The optimizer
        supervised_loss: Supervised component of the loss
        phys_inf_loss: Physics-informed component
        current_lambda_sup: Current supervised weight
        current_lambda_phys_inf: Current physics-informed weight
        alpha: EMA smoothing factor

    Returns:
        (lambda_sup_new, lambda_phys_inf_new)
    """
    grad_sup = analysis_gradients.compute_gradient(model, optimizer, supervised_loss)
    grad_phys_inf = analysis_gradients.compute_gradient(model, optimizer, phys_inf_loss)

    l2_grad_sup = torch.sqrt(sum(torch.sum(g**2) for g in grad_sup))
    l2_grad_phys_inf = torch.sqrt(sum(torch.sum(g**2) for g in grad_phys_inf))

    sum_grads = l2_grad_sup + l2_grad_phys_inf
    lambda_sup_new = sum_grads / l2_grad_sup if l2_grad_sup > 0 else current_lambda_sup
    lambda_phys_inf_new = sum_grads / l2_grad_phys_inf if l2_grad_phys_inf > 0 else current_lambda_phys_inf

    lambda_sup_new = alpha * current_lambda_sup + (1 - alpha) * lambda_sup_new
    lambda_phys_inf_new = alpha * current_lambda_phys_inf + (1 - alpha) * lambda_phys_inf_new

    return lambda_sup_new, lambda_phys_inf_new


# ---------------------------------------------------------------------------
# Training orchestration  (used by scripts/train_networks.py)
# ---------------------------------------------------------------------------

def train_test_sup_network(
    config: dict,
    seed_run: int = 0,
    global_run_id: str = None,
    run_id: str = None) -> dict:
    """Train and test a supervised NCDE or MLP model on synthetic IVIM data for one seed run.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        seed_run (int): Current seed run number (for multiple runs).
        global_run_id (str): Global run ID for grouping multiple seed runs.
        run_id (str): Unique identifier for this specific seed/run.

    Returns:
        dict: Results dictionary with median/std errors for each parameter.
    """

    generator = torch.Generator()
    base_seed = config["train"]["seed"]
    current_seed = base_seed + seed_run
    generator.manual_seed(current_seed)

    datamodule = MRIDataModule(config=config, generator=generator)

    seed_nbr = config["train"]["seed_nbr"]
    run_suffix = f'_seed{seed_run}' if seed_nbr > 1 else ''

    model, atol, rtol, model_dir = train_network(
        config=config,
        global_run_id=global_run_id,
        run_id=run_id,
        modality=config['data']['modality'],
        experiment_name=f'Training_{config["train"]["model"].upper()}',
        run_name=f'{config["train"]["model"]}_run{run_suffix}',
        run_description=config['train']['exp_description'],
        use_mlflow=config['train']['ml_flow_tracking'],
        datamodule=datamodule,
    )

    print('Testing supervised model...')
    datamodule.setup('test')
    results = eval_utils.test_network(
        config, datamodule.test_dataloader(), model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        modality=config['data']['modality'],
        atol=atol, rtol=rtol
    )

    evaluate.finalize_and_save_metrics(
        results, model_dir, config['data']['modality'], seed=current_seed,
    )

    return results


def run_sup_network_seeds(
    config: dict,
    seed_nbr: int,
    base_seed: int,
    global_run_id: str = None) -> list:
    """Run supervised network training for multiple seeds and aggregate results.

    Args:
        config (dict): Configuration dictionary.        
        seed_nbr (int): Number of seed runs.
        base_seed (int): Base random seed.
        global_run_id (str, optional): Global run ID for grouping seed runs.

    Returns:
        list: List of results dicts, one per seed run.
    """
    all_results = []
    for seed_run in range(seed_nbr):
        print(f"\n{'='*80}")
        print(f"Training run {seed_run + 1}/{seed_nbr}")
        print(f"{'='*80}")

        current_seed = base_seed + seed_run
        set_seed(current_seed)

        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{seed_run}"
        results = train_test_sup_network(config, seed_run, global_run_id, run_id)
        all_results.append(results)

    return all_results


# ---------------------------------------------------------------------------
# Multi-fixed training orchestration  (used by scripts/train_multi_fixed_models.py)
# ---------------------------------------------------------------------------

def train_multi_fixed(
    config: dict,
    seed_run: int = 0,
    global_run_id: str = None,
) -> Tuple[dict, dict]:
    """Train multiple fixed-length models, one for each sequence length.

    Each model is trained on data filtered to a specific sequence length and
    evaluated on that same length. Metrics are saved alongside the model and
    config in results/{modality}/{model_type}/{global_run_id}/{run_id}/.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        seed_run (int): Current seed run number (for multiple runs).
        global_run_id (str): Global run ID for grouping multiple seed runs.

    Returns:
        tuple: (full_results dict, models_dict) aggregated across all sequence lengths.
    """
    model_type = config["train"]["model"]

    generator = torch.Generator()
    base_seed = config["train"]["seed"]
    current_seed = base_seed + seed_run
    generator.manual_seed(current_seed)

    data_type = config_utils.determine_data_type(config)

    full_results = eval_utils.init_results_dict(config['data']['modality'])
    models_dict = {}

    experiment_name = f'multi_{model_type}'
    seed_nbr = config["train"]["seed_nbr"]
    run_suffix = f'_seed{seed_run}' if seed_nbr > 1 else ''

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{seed_run}"

    min_x_length = config["data"]["simulation"]["min_x_length"]
    max_x_length = config["data"]["simulation"]["max_x_length"]
    for length in range(min_x_length, max_x_length + 1):
        print(f'\n{"="*60}')
        print(f'Training {model_type} model with {length} measurements (run {seed_run + 1}/{seed_nbr})...')
        print(f'{"="*60}')

        config_copy = copy.deepcopy(config)
        config_copy["data"]["fixed_length"] = length

        train_loader, val_loader, test_loader = get_dataset_loaders(config=config_copy, generator=generator)

        model, atol, rtol, model_dir = train_network(
            config_copy, train_loader, val_loader,
            global_run_id=global_run_id,
            run_id=run_id,
            modality=config['data']['modality'],
            experiment_name=experiment_name,
            run_name=f"{model_type}_length_{length}{run_suffix}",
            use_mlflow=config['train']['ml_flow_tracking'],
        )

        models_dict[length] = model

        print(f'Testing {model_type} model for length {length}...')
        results_one_length = eval_utils.test_network(
            config_copy, test_loader, model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            modality=config['data']['modality'],
            atol=atol, rtol=rtol
        )

        evaluate.finalize_and_save_metrics(
            results_one_length, model_dir, config['data']['modality'],
            metrics_filename=f'metrics_{length}.pt', print_results=False,
        )
        eval_utils.accumulate_results(full_results, results_one_length)

    print(f'\n{"="*60}')
    print("Final results across all b-value lengths:")
    print(f'{"="*60}')
    evaluate.finalize_and_save_metrics(
        full_results, model_dir, config['data']['modality'], seed=current_seed,
        scatter_filename='scatter_pred_vs_true_aggregated.png',
    )

    return full_results, models_dict


def run_multi_fixed_seeds(
    config: dict,
    seed_nbr: int,
    base_seed: int,
    global_run_id: str = None) -> list:
    """Run multi-fixed training for multiple seeds and aggregate results.

    Args:
        config (dict): Configuration dictionary.
        seed_nbr (int): Number of seed runs.
        base_seed (int): Base random seed.
        global_run_id (str, optional): Global run ID for grouping seed runs.

    Returns:
        list: List of results dicts, one per seed run.
    """
    all_results = []
    for seed_run in range(seed_nbr):
        print(f"\n{'='*80}")
        print(f"Training run {seed_run + 1}/{seed_nbr}")
        print(f"{'='*80}")

        current_seed = base_seed + seed_run
        set_seed(current_seed)

        results, _ = train_multi_fixed(config, seed_run=seed_run, global_run_id=global_run_id)
        all_results.append(results)

    return all_results
