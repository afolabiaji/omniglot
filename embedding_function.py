from torch import nn

def conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class EmbeddingFunction(nn.Module):
    def __init__(self, **kwargs):
        super(EmbeddingFunction, self).__init__()
        self.input_dims = kwargs['input_dims']
        self.hidden_dim = kwargs['hidden_dim']
        self.output_dim = kwargs['output_dim']

        self.network = nn.Sequential(
            conv_layer(self.input_dims[0], self.hidden_dim),
            conv_layer(self.hidden_dim, self.hidden_dim),
            conv_layer(self.hidden_dim, self.hidden_dim),
            conv_layer(self.hidden_dim, self.output_dim),
            # nn.Flatten()
        )
    
    def forward(self, x):
        outputs = self.network(x)
        return outputs