'''
This file contains the definition of the NeuralCDE model, which is a neural network that solves
continuous-time differential equations (CDEs) using the torchcde library.
It includes the definition of the CDE function, the readout function, and the main NeuralCDE model.
'''

import torch
import torchcde

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. It's the batch size.
        # hidden_channels is the number of channels for z_t. It's the number of parameters to estimate.
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, 512)
        self.linear4 = torch.nn.Linear(512, 256)
        self.linear5 = torch.nn.Linear(256, 128)
        self.linear6 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.relu()
        z = self.linear3(z)
        z = z.relu()
        z = self.linear4(z)
        z = z.relu()
        z = self.linear5(z)
        z = z.relu()
        z = self.linear6(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        batch_dims = z.shape[:-1]
        z = z.view(*batch_dims, self.hidden_channels, self.input_channels)
        return z
    
def linear_act_norm_dropout(input_channels, output_channels, use_batchnorm, dropout_ratio, activation='relu', w0=1.0):
    """
    function to build a linear layer with activation, batchnorm and dropout
    Args:
        input_channels: amount of input channels
        output_channels: amount of output channels
        use_batchnorm: boolean to use batchnorm
        dropout_ratio: dropout ratio
        activation: type of activation function
        w0: frequency for sine activation

    Returns:
        module_list: list of modules

    """
    module_list = torch.nn.ModuleList()
    module_list.extend([torch.nn.Linear(input_channels, output_channels)])  # fully connected 1
    if use_batchnorm:
        module_list.extend([torch.nn.BatchNorm1d(output_channels)])
    if activation == 'relu':
        module_list.extend([torch.nn.ReLU()])
    elif activation == 'tanh':
        module_list.extend([torch.nn.Tanh()])
    elif activation == 'gelu':
        module_list.extend([torch.nn.GELU()])
    elif activation == 'sine':
        raise NotImplementedError('Sine activation not implemented in this version.')
        # module_list.extend([Sine(w0)])
    if dropout_ratio > 0:
        module_list.extend([torch.nn.Dropout(dropout_ratio)])
    return module_list


def build_module_list(depth, input_channels, hidden_channels, output_channels, dropout,
                      use_batchnorm, activation='relu'):
    """
    function to build a list of modules
    Args:
        depth: amount of layers
        input_channels: amount of input channels
        hidden_channels: amount of hidden channels
        output_channels: amount of output channels
        dropout: dropout ratio
        use_batchnorm: boolean to use batchnorm
        activation: type of activation function

    Returns:
        module_list: list of modules

    """
    # build model
    module_list = torch.nn.ModuleList()

    assert depth > 1, 'assuming minimum depth of 2'

    # input to hidden
    module_list.extend(linear_act_norm_dropout(input_channels, hidden_channels,
                                               use_batchnorm, dropout, activation, w0=30.0))

    # hidden state
    if depth > 2:
        for _ in range(depth - 2):
            module_list.extend(linear_act_norm_dropout(hidden_channels, hidden_channels,
                                                       use_batchnorm, dropout, activation, w0=1.0))

    module_list.extend([torch.nn.Linear(hidden_channels, int(output_channels))])

    return module_list


class Readout(torch.nn.Module):
    """
    class for reading out hidden state into tensor of predefined shape
    """
    def __init__(self, parameters):
        """
        Constructor for Readout
        Args:
            parameters: readout parameters
        """
        super(Readout, self).__init__()

        # retrieve parameters
        self.depth = parameters['model']['readout']['depth']  # number of blocks
        self.input_channels = parameters['model']['readout']['input_channels']  # hidden state
        self.hidden_channels = parameters['model']['readout']['hidden_channels']  # hidden state
        self.output_channels = parameters['model']['readout']['output_channels']
        self.activation = parameters['model']['readout']['activation']
        self.dropout = parameters['model']['readout']['dropout']
        self.use_batchnorm = parameters['model']['readout']['use_batchnorm']
        self.parallel = parameters['model']['readout']['parallel']

        # Apply Kaiming Normal initialization to all ReLU layers
        self._init_weights()

        # readout
        if not self.parallel:
            module_list = build_module_list(self.depth, self.input_channels, self.hidden_channels, self.output_channels,
                                            self.dropout, self.use_batchnorm, self.activation)
            self.encoder = torch.nn.Sequential(*module_list)
        else:
            if self.output_channels >= 1:
                module_list_1 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm,
                                                  self.activation)
                self.encoder_1 = torch.nn.Sequential(*module_list_1)
            if self.output_channels >= 2:
                module_list_2 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm, self.activation)
                self.encoder_2 = torch.nn.Sequential(*module_list_2)
            if self.output_channels >= 3:
                module_list_3 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm, self.activation)
                self.encoder_3 = torch.nn.Sequential(*module_list_3)
            if self.output_channels >= 4:
                module_list_4 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm, self.activation)
                self.encoder_4 = torch.nn.Sequential(*module_list_4)
            if self.output_channels >= 5:
                raise NotImplementedError('only 4 parallel outputs implemented')

    def forward(self, x):
        """
        forward pass of the readout
        Args:
            x: input tensor

        Returns:
            encoding: output tensor
        """
        if not self.parallel:
            encoding = self.encoder(x)
        else:
            encoding_list = []
            if self.output_channels >= 1:
                encoding_list.append(self.encoder_1(x))
            if self.output_channels >= 2:
                encoding_list.append(self.encoder_2(x))
            if self.output_channels >= 3:
                encoding_list.append(self.encoder_3(x))
            if self.output_channels >= 4:
                encoding_list.append(self.encoder_4(x))
            if self.output_channels >= 5:
                raise NotImplementedError
            encoding = torch.cat(encoding_list, dim=1)
        return encoding    
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                # torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
def fill_forward( x, max_length):
    '''Fill forward the last value to pad the sequence to max_length.
    Args:
        x (torch.Tensor): Input tensor of shape (length, features)
        max_length (int): Desired length after padding
    Returns:
        torch.Tensor: Padded tensor of shape (max_length, features)
    '''
    padding_size = max_length - x.size(1)
    if padding_size <= 0:
        return x
    padding = x[:, -1].unsqueeze(1).expand(x.size(0), padding_size, x.size(2))

    return torch.cat([x, padding], dim=1)

def prep_inputs(X, x_max=None):
    '''Prepare the CDE path from raw (b-values, signal) pairs.

    Normalizes the x-axis channel (b-values or TEs) to [0, 1] so that all
    three path channels are on comparable scales for the CDE matrix product.
    The original X tensor is not modified; normalization applies only to the
    path fed to the interpolation routine.

    Args:
        X (torch.Tensor): shape (n_samples, max_length, 2) where channel 0 is
            the x-axis (b-values or echo times) and channel 1 is the signal.
        x_max (float, optional): value used to normalize the x-axis to [0, 1].
            If None, computed as the max non-NaN value across all samples in X.

    Returns:
        torch.Tensor: shape (n_samples, max_length, 3) with channels
            (x_axis / x_max, signal, cumulative mask).
    '''
    x_coords = X[:, :, 0]
    if x_max is None:
        x_max = x_coords.nan_to_num(nan=0.0).max().clamp(min=1.0).item()
    x_coords_norm = x_coords / x_max
    signals = X[:, :, 1]
    masks = (~torch.isnan(signals)).cumsum(dim=1)
    X_prep = torch.stack([x_coords_norm, signals, masks], dim=2)
    X_prep = fill_forward(X_prep, max_length=signals.shape[1])
    return X_prep

class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, depth, activation='relu', interpolation="cubic", dropout=0.0, batchnorm=False, adjoint=False, parallel=False, adaptive=True, interpolation_during_training=False, step_size=0):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        # output_channels is the number of output channels in the prediction. 2 for binary classification, m for the number of parameters to estimate in regression.
        super(NeuralCDE, self).__init__()
        self.initial = torch.nn.Linear(input_channels, hidden_channels) # ltheta1, the initial value of z_t
        self.func = CDEFunc(input_channels, hidden_channels) # func is ftheta, the function that defines the CDE
        self.readout = Readout({'model': {'readout': {'depth': depth, 'activation': activation, 'input_channels': hidden_channels, 'hidden_channels': hidden_channels, 'output_channels': output_channels, 'dropout': dropout, 'use_batchnorm': batchnorm, 'parallel': parallel}}})
        self.interpolation = interpolation
        self.adjoint = adjoint
        self.adaptive = adaptive
        self.interpolation_during_training = interpolation_during_training
        self.step_size = step_size
        ######################
    def forward(self, X, atol=1e-5, rtol=1e-3):
        if self.interpolation_during_training:
            X = prep_inputs(X)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
        else:
            coeffs = X # assume coeffs are already computed outside the model

        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0]) # performs interpolation 
        z0 = self.initial(X0) # ltheta1 is initial, hidden state at time 0

        if self.adaptive: 
            solver = 'dopri5'
        else:
            solver = 'rk4'

        if self.step_size > 0:
            options = dict(step_size=self.step_size)
        else:
            options = dict()

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X, # This will be a tensor of shape (..., len(t)-1, hidden_channels).
                              z0=z0,
                              func=self.func, # func is ftheta 
                              t=X.interval, #self._t[0], self._t[-1]
                              atol=atol, 
                              rtol=rtol, 
                              method=solver, 
                              backend='torchdiffeq', 
                              adjoint=self.adjoint,
                              options=options) # This is the CDE solver, which integrates the CDE defined by ftheta.

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value (not z0),
        # and then apply a linear map.
        ######################
        z_T = z_T[..., -1, :]  # get the terminal value of the CDE
        pred_y = self.readout(z_T) # readout is ltheta2
        return pred_y