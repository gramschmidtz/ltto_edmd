# src/nndmd/network.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), activation()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, output_dim-input_dim))

        self.net = nn.Sequential(*layers)

        # self.A = nn.Parameter(torch.eye(output_dim))

    def forward(self, x):
        latent = self.net(x)
        combined = torch.cat((x,latent),dim=1)
        # out = self.A @ combined
        # out = self.net(x)
        return combined
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), activation()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        self.final_activation = nn.Tanh()

    def forward(self, x):
        z = self.net(x)
        out = self.final_activation(z)
        return out