import torch
import torch.nn.functional as Func
from torch import nn
import numpy as np


class FCNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FCNet, self).__init__()
        composite_dims = [input_dim, *hidden_dims, 1]
        self.hidden_layers = nn.ModuleList([
            torch.nn.Linear(composite_dims[i], composite_dims[i + 1]).double()
            for i in range(len(hidden_dims) + 1)])

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = Func.relu(layer(x))
        return self.hidden_layers[-1](x)


class ResNet(nn.Module):
    def __init__(self, d, m, activation="elu", groups=2, layer_per_group=2):
        super(ResNet, self).__init__()
        self.d = d
        self.m = m

        self.preprocess = nn.Linear(d, m)

        self.groups = groups
        self.layer_per_group = layer_per_group

        self.res_fcs = nn.ModuleList([
            nn.ModuleList([nn.Linear(m, m) for _ in range(self.layer_per_group)]) for _ in range(self.groups)])

        self.fc = nn.Linear(m, 1)
        if activation == "relu":
            self.activation = Func.relu
        elif activation == "relu3":
            self.activation = lambda x: Func.relu(x ** 3)
        elif activation == "elu":
            self.activation = Func.elu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise Exception(f"Can't recognize activation {activation}!")

    def forward(self, x):
        z = self.activation(self.preprocess(x))
        for group in self.res_fcs:
            z_id = z
            for layer in group:
                z = self.activation(layer(z))
            z = z_id + z
        return self.fc(z)

    def dof(self) -> int:
        count = 0
        for param in self.parameters():
            count += np.prod(param.size())
        return count


class StrictResNet(nn.Module):
    def __init__(self, d, m, activation="elu", groups=2, layer_per_group=2):
        super(StrictResNet, self).__init__()
        self.d = d
        self.m = m

        self.preprocess = nn.Linear(d, m)

        self.groups = groups
        self.layer_per_group = layer_per_group

        self.res_fcs = nn.ModuleList([
            nn.ModuleList([nn.Linear(m, m) for _ in range(self.layer_per_group)]) for _ in range(self.groups)])

        self.fc = nn.Linear(m, 1)
        if activation == "relu":
            self.activation = Func.relu
        elif activation == "relu3":
            self.activation = lambda x: Func.relu(x ** 3)
        elif activation == "elu":
            self.activation = Func.elu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise Exception(f"Can't recognize activation {activation}!")

    def forward(self, x):
        z = self.activation(self.preprocess(x))
        for group in self.res_fcs:
            z_id = z
            for i, layer in enumerate(group):
                z = layer(z)
                if i < len(group) - 1:
                    z = self.activation(layer(z))
            z = self.activation(z_id + z)
        return self.fc(z)

    def dof(self) -> int:
        count = 0
        for param in self.parameters():
            count += np.prod(param.size())
        return count
