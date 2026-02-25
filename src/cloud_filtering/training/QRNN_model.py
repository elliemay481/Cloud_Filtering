"""

Machine learning model for AWS cloud signal retrievals
"""
from torch import nn

from quantnn.models.pytorch.encoders import SpatialEncoder
from quantnn.models.pytorch.decoders import SpatialDecoder
from quantnn.models.pytorch.fully_connected import MLP, FullyConnectedBlock
#import quantnn.models.pytorch.torchvision as blocks
from torch import nn
import torch
from quantnn.models.pytorch.common import PytorchModel, activations
import numpy as np


class CloudSignalModel(PytorchModel, nn.Module):
    """
    A QRNN model for AWS cloud signal retrievals.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_layers,
        width,
        activation=nn.ReLU,
        batch_norm=False,
        skip_connections=False,
    ):
        """
        Create a fully-connect neural network model.

        Args:
            n_inputs: The number of input features to the network.
            n_outputs: The number of outputs of the model.
            layers: The number of hidden layers in the model.
            width: The number of neurons in the hidden layers.
            activation: The activation function to use in the hidden
                  layers.
            batch_norm: Whether to include a batch-norm layer after
                 each hidden layer.
        """
        self.skips = skip_connections

        super().__init__()
        nn.Module.__init__(self)

        if isinstance(activation, str):
            activation = activations[activation]

        nominal_width = width
        if self.skips:
            nominal_width = (nominal_width - n_inputs) // 2

        n_in = n_inputs
        n_out = nominal_width

        modules = []
        for i in range(n_layers):
            modules.append(
                FullyConnectedBlock(n_in, n_out, activation, batch_norm=batch_norm)
            )
            if self.skips:
                if i == 0:
                    n_in = n_out + n_inputs
                else:
                    n_in = 2 * n_out + n_inputs
            else:
                n_in = n_out
        
        # Modify to output two sets of quantiles
        modules.append(nn.Linear(n_in, sum(n_outputs)))

        self.mods = nn.ModuleList(modules)
        
        self.n_variables = len(n_outputs)

    def forward(self, x):
        """Propagate input through network."""

        y_p = []
        y_l = self.mods[0](x) # run input through first layer

        for layer in self.mods[1:]:
            if self.skips:
                y = torch.cat(y_p + [y_l, x], 1)
                y_p = [y_l]
            else:
                y = y_l
            y_l = layer(y)

        if self.n_variables > 1:
            out = y_l.view(y_l.shape[0], -1, self.n_variables)
        else:
            out = y_l

        return out
    

class CloudSignalModel_MultiOutput(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_layers,
        width,
        activation=nn.ReLU,
        batch_norm=False,
        skip_connections=False,
        threshold=1e-6  # Small threshold to decide if Dm profile should be predicted
    ):
        """
        Create a fully-connect neural network model with multiple
        sets of outputs.

        Args:
            n_inputs: The number of input features to the network.
            n_outputs: Dict of outputs of the model and no. of outputs (quantiles).
            layers: The number of hidden layers in the model.
            width: The number of neurons in the hidden layers.
            activation: The activation function to use in the hidden
                  layers.
            batch_norm: Whether to include a batch-norm layer after
                 each hidden layer.
        """

        self.threshold = threshold

        self.skips = skip_connections
        
        super().__init__()
        nn.Module.__init__(self)

        if isinstance(activation, str):
            activation = activations[activation]

        nominal_width = width
        if self.skips:
            nominal_width = (nominal_width - n_inputs) // 2

        n_in = n_inputs
        n_out = nominal_width

        # Set up the shared layers
        self.shared_layers = nn.ModuleList()
        n_in = n_inputs
        for i in range(n_layers):
            self.shared_layers.append(
                FullyConnectedBlock(n_in, n_out, activation, batch_norm=batch_norm)
            )
            if self.skips:
                if i == 0:
                    n_in = n_out + n_inputs
                else:
                    n_in = 2 * n_out + n_inputs
            else:
                n_in = n_out


        # Set up the output heads (layers diverge)
        self.heads = nn.ModuleDict()
        for name, output_size in n_outputs.items():  # n outputs needs to be a dict here!
            head_layers = nn.ModuleList([
                nn.Linear(n_in, width),
                activation(),
                nn.BatchNorm1d(width) if batch_norm else nn.Identity(),
                nn.Linear(width, output_size)
            ])
            self.heads[name] = head_layers

    def forward(self, x):
        """Propagate input through network."""
        #print(x.shape)
        y_p = []
        y_l = self.shared_layers[0](x) # run input through first layer

        # run through shared layers
        for layer in self.shared_layers[1:]:
            if self.skips:
                y = torch.cat(y_p + [y_l, x], 1)
                y_p = [y_l]
            else:
                y = y_l
                #print(y.shape)
            y_l = layer(y)   # torch.Size([256, 512])
        
        #print(y_l.shape)

        # Pass the shared output through each output head
        outputs = {}
        for name, layers in self.heads.items():
            y = y_l # take output from shared layers
            for layer in layers:
                #print(layer)
                y = layer(y)
            outputs[name] = y


        return outputs
