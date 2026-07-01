"""Biophysical model abstraction and registry.

Provides a unified interface for MRI signal models (IVIM, T2*, etc.)
with multi-backend support (numpy, torch, scipy) and mutable parameter ranges.

Usage:
    from flexqmri.utils.biophysical_model import get_model, register_model

    model = get_model("ivim_bi_exp")
    model.update_param_range("f", 0.05, 0.4)
    signal = model.forward_numpy(b_values, params)
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BiophysicalModel(ABC):
    """Abstract base class for MRI biophysical signal models.

    Bundles the signal equation (multi-backend), parameter metadata, and
    mutable parameter ranges into one object. Concrete subclasses implement
    ``forward_numpy``, ``forward_torch``, and ``forward_scipy``.

    Args:
        name (str): Unique model identifier (e.g. 'ivim_bi_exp').
        param_names (list[str]): Ordered list of parameter names.
        default_ranges (list[list[float]]): Default [min, max] per parameter.
    """

    def __init__(self, name: str, param_names: List[str],
                 default_ranges: List[List[float]]):
        self._name = name
        self._param_names = list(param_names)
        self._param_ranges = [list(r) for r in default_ranges]

        if len(self._param_names) != len(self._param_ranges):
            raise ValueError(
                f"param_names ({len(self._param_names)}) and "
                f"default_ranges ({len(self._param_ranges)}) must have the same length."
            )

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def param_names(self) -> List[str]:
        return list(self._param_names)

    @property
    def n_params(self) -> int:
        return len(self._param_names)

    @property
    def param_ranges(self) -> List[List[float]]:
        """Return a deep copy so callers cannot silently mutate internal state."""
        return deepcopy(self._param_ranges)

    # -- range mutation -----------------------------------------------------

    def set_param_ranges(self, ranges: List[List[float]]) -> None:
        """Replace all parameter ranges.

        Args:
            ranges (list[list[float]]): New [min, max] per parameter.

        Raises:
            ValueError: If length does not match ``n_params``.
        """
        if len(ranges) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} ranges, got {len(ranges)}."
            )
        self._param_ranges = [list(r) for r in ranges]

    def update_param_range(self, param_name: str,
                           new_min: float, new_max: float) -> None:
        """Update the range of a single parameter by name.

        Args:
            param_name (str): Name of the parameter (must be in ``param_names``).
            new_min (float): New lower bound.
            new_max (float): New upper bound.

        Raises:
            ValueError: If ``param_name`` not found.
        """
        idx = self._param_index(param_name)
        self._param_ranges[idx] = [new_min, new_max]

    # -- rescaling / bounds -------------------------------------------------

    def rescale_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        """Map unit-range coefficients [0, 1] to physical parameter values.

        Args:
            coeffs (np.ndarray): Array of shape (..., n_params) in [0, 1].

        Returns:
            np.ndarray: Rescaled parameter values.
        """
        coeffs_t = torch.from_numpy(np.asarray(coeffs, dtype=np.float32))
        return rescale_coeffs_torch(self._param_ranges, coeffs_t).numpy()

    def rescale_coeffs_torch(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Map unit-range coefficients [0, 1] to physical parameter values (torch).

        Args:
            coeffs (torch.Tensor): Tensor of shape (..., n_params) in [0, 1].

        Returns:
            torch.Tensor: Rescaled parameter values.
        """
        return rescale_coeffs_torch(self._param_ranges, coeffs)

    def get_bounds(self) -> Tuple[list, list]:
        """Return (lower_bounds, upper_bounds) lists for scipy ``curve_fit``.

        Returns:
            tuple[list, list]: Lower and upper bounds.
        """
        lower = [r[0] for r in self._param_ranges]
        upper = [r[1] for r in self._param_ranges]
        return lower, upper

    # -- forward methods (to be overridden) ---------------------------------

    @abstractmethod
    def forward_numpy(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Vectorised numpy forward pass for data generation.

        Args:
            x (np.ndarray): Measurement points (e.g. b-values), shape varies.
            params (np.ndarray): Parameter array, shape (n_samples, n_params).

        Returns:
            np.ndarray: Predicted signal.
        """

    @abstractmethod
    def forward_torch(self, x: torch.Tensor, params: torch.Tensor,
                      training: bool = False) -> torch.Tensor:
        """Differentiable torch forward pass for training loss computation.

        Args:
            x (torch.Tensor): Measurement points.
            params (torch.Tensor): Parameters.
            training (bool): If True, reshape params for batch broadcasting.

        Returns:
            torch.Tensor: Predicted signal.
        """

    @abstractmethod
    def forward_scipy(self, x: np.ndarray, *args: float) -> np.ndarray:
        """Unpacked-parameter forward pass for ``scipy.optimize.curve_fit``.

        Args:
            x (np.ndarray): Measurement points (1-D).
            *args: Individual parameter scalars.

        Returns:
            np.ndarray: Predicted signal (1-D).
        """

    # -- helpers ------------------------------------------------------------

    def _param_index(self, name: str) -> int:
        """Return the index of a parameter by name.

        Raises:
            ValueError: If not found.
        """
        if name not in self._param_names:
            raise ValueError(
                f"Unknown parameter '{name}'. "
                f"Available: {self._param_names}"
            )
        return self._param_names.index(name)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self._name}', "
            f"params={self._param_names})"
        )


# ---------------------------------------------------------------------------
# Shared rescaling utility (used by the ABC and by legacy code paths)
# ---------------------------------------------------------------------------

def rescale_coeffs_torch(param_ranges: list,
                         y_coeffs: torch.Tensor) -> torch.Tensor:
    """Rescale output coefficients to their original physical parameter range.

    Args:
        param_ranges (list): List of [min, max] per parameter.
        y_coeffs (torch.Tensor): Coefficients in [0, 1].

    Returns:
        torch.Tensor: Rescaled parameters.
    """
    y_params = torch.zeros(y_coeffs.shape, device=y_coeffs.device)
    for i in range(y_coeffs.shape[-1]):
        y_params[..., i] = param_ranges[i][0] + y_coeffs[..., i] * (
            param_ranges[i][1] - param_ranges[i][0]
        )
    return y_params


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class IVIMBiExp(BiophysicalModel):
    """IVIM bi-exponential model: S(b) = S0 * [f * exp(-b*D*) + (1-f) * exp(-b*D)].

    Parameter order: [S0, f, D, D*].
    """

    DEFAULT_RANGES = [
        [0.9, 1.1],       # S0
        [0.01, 0.6],      # f  (perfusion fraction)
        [0.0001, 0.0035],  # D  (diffusion coefficient, mm²/s)
        [0.003, 0.1],      # D* (pseudo-diffusion coefficient, mm²/s), [0.05, 0.2],  
    ]

    def __init__(self, param_ranges: List[List[float]] = None):
        ranges = param_ranges if param_ranges is not None else self.DEFAULT_RANGES
        super().__init__(
            name="ivim_bi_exp",
            param_names=["S0", "f", "D", "D*"],
            default_ranges=ranges,
        )

    # -- numpy (data generation) -------------------------------------------

    def forward_numpy(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute IVIM signal using numpy.

        Args:
            x (np.ndarray): b-values, shape (n_samples, n_b) or (n_b,).
            params (np.ndarray): Shape (n_samples, 4) — [S0, f, D, D*].

        Returns:
            np.ndarray: Signal, same leading dims as ``x``.
        """
        S0 = params[..., 0:1]
        f = params[..., 1:2]
        D = params[..., 2:3]
        D_star = params[..., 3:4]
        perfusion = f * np.exp(-x * D_star)
        diffusion = (1 - f) * np.exp(-x * D)
        return S0 * (perfusion + diffusion)

    # -- torch (training loss) ---------------------------------------------

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor,
                      training: bool = False) -> torch.Tensor:
        """Compute IVIM signal using torch.

        Args:
            x (torch.Tensor): b-values.
            params (torch.Tensor): Parameters tensor with last dim = 4.
            training (bool): If True, unsqueeze params for batch broadcasting.

        Returns:
            torch.Tensor: Signal.
        """
        S0 = params[..., 0]
        f = params[..., 1]
        d_slow = params[..., 2]
        d_fast = params[..., 3]
        if training:
            S0 = S0.view(-1, 1)
            f = f.view(-1, 1)
            d_slow = d_slow.view(-1, 1)
            d_fast = d_fast.view(-1, 1)
        relative = f * torch.exp(-x * d_fast) + (1 - f) * torch.exp(-x * d_slow)
        return S0 * relative

    # -- scipy (curve_fit) --------------------------------------------------

    def forward_scipy(self, x: np.ndarray, S0: float, f: float,
                      d_slow: float, d_fast: float) -> np.ndarray:
        """Unpacked IVIM forward for ``scipy.optimize.curve_fit``.

        Args:
            x (np.ndarray): b-values (1-D).
            S0 (float): Signal at b=0.
            f (float): Perfusion fraction.
            d_slow (float): Diffusion coefficient D.
            d_fast (float): Pseudo-diffusion coefficient D*.

        Returns:
            np.ndarray: Signal (1-D).
        """
        exp_fast = np.exp(np.clip(-x * d_fast, -100, 100))
        exp_slow = np.exp(np.clip(-x * d_slow, -100, 100))
        return S0 * (f * exp_fast + (1 - f) * exp_slow)


class T2StarMonoExp(BiophysicalModel):
    """T2* mono-exponential decay: S(TE) = S0 * exp(-TE / T2*).

    Parameter order: [S0, T2*].
    """

    DEFAULT_RANGES = [
        [0.5, 1.5],  # S0
        [1, 100],    # T2* (ms)
    ]

    def __init__(self, param_ranges: List[List[float]] = None):
        ranges = param_ranges if param_ranges is not None else self.DEFAULT_RANGES
        super().__init__(
            name="t2star_mono_exp",
            param_names=["S0", "t2star"],
            default_ranges=ranges,
        )

    # -- numpy --------------------------------------------------------------

    def forward_numpy(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute T2* signal using numpy.

        Args:
            x (np.ndarray): Echo times (TE), may contain NaN.
            params (np.ndarray): Shape (n_samples, 2) — [S0, T2*].

        Returns:
            np.ndarray: Signal.
        """
        S0 = params[..., 0:1]
        t2star = params[..., 1:2]
        return S0 * np.exp(-x / t2star)

    # -- torch --------------------------------------------------------------

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor,
                      training: bool = False) -> torch.Tensor:
        """Compute T2* signal using torch.

        Args:
            x (torch.Tensor): Echo times.
            params (torch.Tensor): Parameters tensor with last dim = 2.
            training (bool): If True, unsqueeze params for batch broadcasting.

        Returns:
            torch.Tensor: Signal.
        """
        S0 = params[..., 0]
        t2star = params[..., 1]
        if training:
            S0 = S0.view(-1, 1)
            t2star = t2star.view(-1, 1)
        return S0 * torch.exp(-x / t2star)

    # -- scipy --------------------------------------------------------------

    def forward_scipy(self, x: np.ndarray, S0: float,
                      t2star: float) -> np.ndarray:
        """Unpacked T2* forward for ``scipy.optimize.curve_fit``.

        Args:
            x (np.ndarray): Echo times (1-D).
            S0 (float): Signal at TE=0.
            t2star (float): T2* relaxation time.

        Returns:
            np.ndarray: Signal (1-D).
        """
        exponent = np.clip(-x / t2star, -100, 100)
        return S0 * np.exp(exponent)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict = {}


def register_model(name: str, model: BiophysicalModel) -> None:
    """Register a biophysical model in the global registry.

    Args:
        name (str): Lookup key (e.g. 'ivim_bi_exp').
        model (BiophysicalModel): Model instance.
    """
    _REGISTRY[name] = model


def get_model(name: str) -> BiophysicalModel:
    """Retrieve a **fresh copy** of a registered model by name.

    A deep copy is returned so that callers can mutate param_ranges without
    affecting the registry's default instance.

    Args:
        name (str): Registered model name.

    Returns:
        BiophysicalModel: Deep copy of the registered model.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown biophysical model '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return deepcopy(_REGISTRY[name])


def get_model_from_config(config: dict) -> BiophysicalModel:
    """Retrieve a model by name, applying a config-level param_ranges override.

    Looks up the model named by ``config["data"]["param_model"]`` and, when the
    optional ``config["data"]["param_ranges"]`` key is present, replaces the
    model's default ranges with it. This lets a modality YAML override the
    defaults baked into the biophysical model.

    Args:
        config (dict): Full config dict; reads ``config["data"]["param_model"]``
            and the optional override ``config["data"]["param_ranges"]``.

    Returns:
        BiophysicalModel: Deep copy of the registered model, with param_ranges
            replaced if ``config["data"]["param_ranges"]`` is set.

    Raises:
        ValueError: If the model name is not registered, or the override length
            does not match the model's number of parameters.
    """
    model = get_model(config["data"]["param_model"])
    param_ranges = config["data"].get("param_ranges")
    if param_ranges is not None:
        model.set_param_ranges(param_ranges)
    return model


def list_models() -> List[str]:
    """Return the names of all registered models.

    Returns:
        list[str]: Sorted model names.
    """
    return sorted(_REGISTRY.keys())


# Auto-register built-in models on import
register_model("ivim_bi_exp", IVIMBiExp())
register_model("t2star_mono_exp", T2StarMonoExp())


# ---------------------------------------------------------------------------
# High-level dispatch helpers
# ---------------------------------------------------------------------------

def apply_phys_model(
    x,
    param_model: str,
    params,
    torch_based: bool = False,
    training: bool = False,
):
    """Apply a physical model to input data via the registry.

    Args:
        x: Measurement points (e.g. b-values or echo times).
        param_model (str): Registry model name (e.g. ``"ivim_bi_exp"``).
        params: Parameter array or tensor.
        torch_based (bool): Use PyTorch forward pass when ``True``, numpy otherwise.
        training (bool): Reshape params for batch broadcasting (torch path only).

    Returns:
        Predicted signal — ``torch.Tensor`` when ``torch_based`` is ``True``,
        ``np.ndarray`` otherwise.

    Raises:
        ValueError: If ``param_model`` is not in the registry.
    """
    model = get_model(param_model)
    if torch_based:
        return model.forward_torch(x, params, training=training)
    return model.forward_numpy(x, params)


def get_phys_param(param_model: str) -> dict:
    """Return physical model metadata from the registry.

    Args:
        param_model (str): Registry model name (e.g. ``"ivim_bi_exp"``).

    Returns:
        dict: Contains:
            - ``"param_ranges"``: list of ``[min, max]`` per parameter.
            - ``"phys_model_function"``: scipy-compatible forward callable.
            - ``"biophysical_model"``: the ``BiophysicalModel`` instance.

    Raises:
        ValueError: If ``param_model`` is not registered.
    """
    model = get_model(param_model)
    return {
        "param_ranges": model.param_ranges,
        "phys_model_function": model.forward_scipy,
        "biophysical_model": model,
    }


# ---------------------------------------------------------------------------
# Standalone model functions (used by curve fitting scripts and figures)
# ---------------------------------------------------------------------------

def ivim_model(
    b,
    S0,
    f,
    d_slow,
    d_fast,
    torch_based: bool = False,
    training: bool = False,
):
    """Compute the IVIM bi-exponential signal.

    Args:
        b: b-values.
        S0: Signal at b=0.
        f: Perfusion fraction.
        d_slow: Diffusion coefficient D (mm²/s).
        d_fast: Pseudo-diffusion coefficient D* (mm²/s).
        torch_based (bool): Use torch operations when ``True``.
        training (bool): Reshape scalar params for batch broadcasting.

    Returns:
        Signal — same type as ``b`` (``torch.Tensor`` or ``np.ndarray``).
    """
    if torch_based:
        if training:
            S0 = S0.view(-1, 1)
            f = f.view(-1, 1)
            d_slow = d_slow.view(-1, 1)
            d_fast = d_fast.view(-1, 1)
        relative = f * torch.exp(-b * d_fast) + (1 - f) * torch.exp(-b * d_slow)
    else:
        relative = f * np.exp(np.clip(-b * d_fast, -100, 100)) + (
            (1 - f) * np.exp(np.clip(-b * d_slow, -100, 100))
        )
    return S0 * relative


def ivim_model_function(b, S0: float, f: float, d_slow: float, d_fast: float) -> np.ndarray:
    """Scipy-compatible IVIM wrapper for ``curve_fit``.

    Args:
        b (np.ndarray): b-values (1-D).
        S0 (float): Signal at b=0.
        f (float): Perfusion fraction.
        d_slow (float): Diffusion coefficient D.
        d_fast (float): Pseudo-diffusion coefficient D*.

    Returns:
        np.ndarray: Predicted signal.
    """
    return ivim_model(b, S0, f, d_slow, d_fast, torch_based=False)


def t2star_model(
    te,
    S0,
    t2star,
    torch_based: bool = False,
    training: bool = False,
):
    """Compute the T2* mono-exponential decay signal.

    Args:
        te: Echo times (TE).
        S0: Signal at TE=0.
        t2star: T2* relaxation time (ms).
        torch_based (bool): Use torch operations when ``True``.
        training (bool): Reshape scalar params for batch broadcasting.

    Returns:
        Signal — same type as ``te``.
    """
    if torch_based:
        if training:
            S0 = S0.view(-1, 1)
            t2star = t2star.view(-1, 1)
        return S0 * torch.exp(-te / t2star)
    return S0 * np.exp(np.clip(-te / t2star, -100, 100))


def t2star_model_function(te, S0: float, t2star: float) -> np.ndarray:
    """Scipy-compatible T2* wrapper for ``curve_fit``.

    Args:
        te (np.ndarray): Echo times (1-D).
        S0 (float): Signal at TE=0.
        t2star (float): T2* relaxation time.

    Returns:
        np.ndarray: Predicted signal.
    """
    return t2star_model(te, S0, t2star, torch_based=False)
