'''Multi-layer Perceptron (MLP) network for regression tasks.'''

import torch
from .ncde import Readout


class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, depth, activation, dropout=0.0, batchnorm=False, parallel=False):
        super(MLP, self).__init__()
        self.readout = Readout({'model': {'readout': {'depth': depth, 'activation': activation, 'input_channels': input_channels, 'hidden_channels': hidden_channels, 'output_channels': output_channels, 'dropout': dropout, 'use_batchnorm': batchnorm, 'parallel': parallel}}})

    def forward(self, x):
        # Use the signal channel only (channel 1); input is (n_samples, seq_len, 2)
        x = x[:, :, 1]  # (n_samples, seq_len)
        pred_y = self.readout(x)
        return pred_y