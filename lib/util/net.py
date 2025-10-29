import torch
import torch.nn.functional as Func
from torch import nn
import numpy as np

from typing import Callable


class FCNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FCNet, self).__init__()
        composite_dims = [input_dim, *hidden_dims, 1]
        self.hidden_layers = nn.ModuleList(
            [torch.nn.Linear(composite_dims[i], composite_dims[i + 1]).double() for i in range(len(hidden_dims) + 1)]
        )

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

        self.res_fcs = nn.ModuleList(
            [nn.ModuleList([nn.Linear(m, m) for _ in range(self.layer_per_group)]) for _ in range(self.groups)]
        )

        self.fc = nn.Linear(m, 1)
        if activation == "relu":
            self.activation = Func.relu
        elif activation == "relu3":
            self.activation = lambda x: Func.relu(x**3)
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
            for layer in group:  # type: ignore
                z = self.activation(layer(z))
            z = z_id + z
        return self.fc(z)

    def dof(self) -> int:
        count = 0
        for param in self.parameters():
            count += np.prod(param.size())
        return count  # type:ignore


class StrictResNet(nn.Module):
    def __init__(self, d, m, activation="elu", groups=2, layer_per_group=2):
        super(StrictResNet, self).__init__()
        self.d = d
        self.m = m

        self.preprocess = nn.Linear(d, m)

        self.groups = groups
        self.layer_per_group = layer_per_group

        self.res_fcs = nn.ModuleList(
            [nn.ModuleList([nn.Linear(m, m) for _ in range(self.layer_per_group)]) for _ in range(self.groups)]
        )

        self.fc = nn.Linear(m, 1)
        if activation == "relu":
            self.activation = Func.relu
        elif activation == "relu3":
            self.activation = lambda x: Func.relu(x**3)
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
            for i, layer in enumerate(group):  # type: ignore
                z = layer(z)
                if i < len(group) - 1:  # type: ignore
                    z = self.activation(z)
            z = self.activation(z_id + z)
        return self.fc(z)

    def dof(self) -> int:
        count = 0
        for param in self.parameters():
            count += np.prod(param.size())
        return count  # type:ignore


Transform = Callable[[torch.Tensor], torch.Tensor]
TransformPair = tuple[Transform, Transform]


class CombinedResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        transform_pair: TransformPair,
        activation="elu",
        output_dim=2,
        groups=2,
        layer_per_group=2,
    ):
        super(CombinedResNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.in_transform, self.out_transform = transform_pair

        self.preprocess = nn.Linear(input_dim, hidden_dim)

        self.groups = groups
        self.layer_per_group = layer_per_group

        self.res_fcs = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layer_per_group)])
                for _ in range(self.groups)
            ]
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        if activation == "relu":
            self.activation = Func.relu
        elif activation == "relu3":
            self.activation = lambda x: Func.relu(x**3)
        elif activation == "elu":
            self.activation = Func.elu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise Exception(f"Can't recognize activation {activation}!")

    def forward(self, x):
        x = self.in_transform(x.clone())
        z = self.activation(self.preprocess(x))
        for group in self.res_fcs:
            z_id = z
            for i, layer in enumerate(group):  # type: ignore
                z = layer(z)
                if i < len(group) - 1:  # type: ignore
                    z = self.activation(z)
            z = self.activation(z_id + z)
        y = self.fc(z)
        if torch.any(torch.isnan(y)):
            breakpoint()
            raise ValueError("NaN encountered in CombinedResNet forward!")
        return self.out_transform(y)

    def dof(self) -> int:
        count = 0
        for param in self.parameters():
            count += np.prod(param.size())
        return count  # type:ignore
