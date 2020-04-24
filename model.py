import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layers, seed, drop_p=0.):
        """Initialize parameters and build model with arbitrary hidden layers.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (list): list of integers, the sizes of the hidden layers
            drop_p (float): float between 0 and 1, dropout probability
            seed (int): Random seed
        """
        super().__init__()            
        self.seed = torch.manual_seed(seed)
        
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)        
        #self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
         # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            #x = self.dropout(x)
        
        return self.output(x)
