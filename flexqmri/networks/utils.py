import torch

from . import ncde
from . import mlp
from . import transformer

def net_factory(config):
    '''Factory function to create neural network models.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        model: Instantiated neural network model
    '''
    if config['data']['param_model'] == 'ivim_bi_exp': 
        output_channels = 4
    elif config['data']['param_model'] == 't2star_mono_exp': 
        output_channels = 2
    else: 
        raise ValueError

    if config["train"]["model"] == 'ncde':
        model = ncde.NeuralCDE(input_channels=3, # b-values, signal, mask
                          hidden_channels=config["train"]["hidden_channels"],
                          output_channels=output_channels,
                          depth=config["train"]["depth"],
                          activation=config["train"]["activation"],
                          dropout=config["train"]["dropout"],
                          batchnorm=config["train"]["batchnorm"],
                          adjoint=config["train"]["adjoint"],
                          step_size=config["train"]["time_step"],
                          parallel=config["train"]["parallel"],
                          adaptive=config["train"]["adaptive"],
                          interpolation_during_training=config["train"].get("interpolation_during_training", False))
    elif config["train"]["model"] == 'mlp':
        print("Using MLP model with fixed length:", config["data"]["fixed_length"])
        assert config["data"]["fixed_length"] > 0, "Fixed length must be specified for MLP"
        model = mlp.MLP(input_channels=config["data"]["fixed_length"],
                    hidden_channels=config["train"]["hidden_channels"],
                    output_channels=output_channels,
                    depth=config["train"]["depth"],
                    activation=config["train"]["activation"],
                    dropout=config["train"]["dropout"],
                    batchnorm=config["train"]["batchnorm"],
                    parallel=config["train"]["parallel"])
    elif config["train"]["model"] == 'transformer':
        assert config["data"]["fixed_length"] > 0, "Fixed length must be specified for Transformer"
        model = transformer.Transformer(
                    input_channels=1,
                    output_channels=output_channels,
                    max_length=config["data"]["fixed_length"],
                    d_model=config["train"]["hidden_channels"],
                    n_head=config["train"].get("n_head", 4),
                    n_layers=config["train"]["depth"],
                    d_inner=config["train"]["hidden_channels"],
                    activation=config["train"]["activation"],
                    dropout=config["train"]["dropout"])
    else:
        raise ValueError(f"Unknown model name: {config['train']['model']}")

    return model


def load_model(model_path: str, config: dict) -> torch.nn.Module:
    """Load a trained model from a .pth file.

    Args:
        model_path (str): Path to the .pth state dict file.
        config (dict): Configuration dictionary used to build the model architecture.

    Returns:
        torch.nn.Module: Model with loaded weights set to evaluation mode.
    """
    model = net_factory(config)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model