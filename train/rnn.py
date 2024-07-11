import torch
from torch.nn import Linear, ELU, Sequential, ReLU, Sigmoid, RNN
import torch.nn.functional as F

class _RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(_RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_size = input_size

        #Defining the layers
        # RNN Layer
        self.rnn = RNN(input_size, hidden_dim, n_layers)

        # Fully connected layer
        self.fc = Linear(hidden_dim, output_size)

        # Output
        self.output = Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        input  = x.view(batch_size, 1, self.input_size)

        # Initializing hidden state for first input using method defined below
        hidden = torch.zeros(self.n_layers, 1, self.hidden_dim).to(device='cpu')
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.rnn(input, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        out = out.view((batch_size, 1))

        out = self.output(out)
        
        return out