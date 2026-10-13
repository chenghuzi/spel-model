"""Shared initialization, activations, readout, and visual encoder."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim - n + 1,))
        else:
            x = rng.normal(size=(dim - n + 1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = -D * (np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1 :, n - 1 :] = Hx
        H = np.dot(H, mat)
    return H


class NonNegativeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NonNegativeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        eps = 1e-3
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, requires_grad=True) + eps)
        self.bias = nn.Parameter(torch.zeros(out_features, requires_grad=True) + eps)

    def forward(self, input):
        # Ensure non-negativity
        weight = self.weight
        bias = self.bias
        # weight = F.relu(self.weight)
        # bias = F.relu(self.bias)

        return F.linear(input, weight, bias)


class SaturatedSoftplus(nn.Module):
    def __init__(self, peak=5.0):
        super().__init__()
        self.peak = peak

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=0.0, max=self.peak)


def build_recurrent_activation(config):
    activation = str(config["activation"]).lower()
    if activation in {"softplus", "saturated_softplus"}:
        return SaturatedSoftplus(peak=float(config["activation_peak"]))
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported recurrent activation: {activation}")


def build_readout(n_h, n_o, config):
    readout = str(config["readout"]).lower()
    if readout in {"axby", "nonnegative", "non_negative"}:
        return NonNegativeLinear(n_h, n_o)
    if readout == "linear":
        return nn.Linear(n_h, n_o)
    raise ValueError(f"Unsupported readout: {readout}")


def group_norm_groups(num_channels):
    for groups in (8, 4, 2):
        if num_channels % groups == 0:
            return groups
    return 1


class CNNVisualEncoder(nn.Module):
    def __init__(self, n_in, latent_dim, config):
        super().__init__()
        self.image_channels = int(config["image_channels"])
        if n_in % self.image_channels != 0:
            raise ValueError(
                f"CNNVisualEncoder cannot split n_in={n_in} across {self.image_channels} channels"
            )
        pixels_per_channel = n_in // self.image_channels
        side = int(np.sqrt(pixels_per_channel))
        if side * side != pixels_per_channel:
            raise ValueError(
                "CNNVisualEncoder expects square channel-first images, "
                f"got n_in={n_in}, channels={self.image_channels}"
            )
        self.side = side
        channels = list(config["cnn_channels"])
        layers = []
        in_channels = self.image_channels
        for out_channels in channels:
            out_channels = int(out_channels)
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(group_norm_groups(out_channels), out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, flat_images):
        images = flat_images.reshape(
            flat_images.shape[0],
            self.image_channels,
            self.side,
            self.side,
        )
        features = self.pool(self.cnn(images))
        return self.projection(features)
