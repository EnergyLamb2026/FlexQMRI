"""
Lightning DataModule for MRI regression datasets.

Wraps the existing ``get_dataset_loaders`` pipeline into a single
``LightningDataModule`` so that the ``Trainer`` can handle data setup
and data-loader creation automatically.
"""

from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from flexqmri.dataset.factory import get_dataset_loaders


class MRIDataModule(L.LightningDataModule):
    """Lightning DataModule for MRI regression (synthetic).

    Delegates data generation, splitting, and NCDE interpolation to the
    existing :func:`dataset.get_dataset_loaders` factory and caches the
    resulting loaders.

    Example::

        dm = MRIDataModule(config)
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    """

    def __init__(
        self,
        config: dict,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            config: Full experiment configuration dictionary.
            generator: PyTorch Generator for reproducible splits.
        """
        super().__init__()
        self.config = config
        self.generator = generator

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Create loaders once and cache them."""
        if self._train_loader is not None:
            return  # already set up

        self._train_loader, self._val_loader, self._test_loader = (
            get_dataset_loaders(
                config=self.config,
                generator=self.generator,
            )
        )

    def train_dataloader(self) -> Optional[DataLoader]:
        return self._train_loader

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._val_loader

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._test_loader
