"""Optimizer and learning-rate schedule construction."""

import math

import torch


def build_optimizer(model, readout_lr, config):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adam(trainable, lr=readout_lr)


def cosine_lr_multiplier(epoch, initial_lr, min_lr, decay_epochs):
    # Validate the public schedule parameters rather than silently turning cosine into constant LR.
    if decay_epochs <= 0 or initial_lr <= 0 or not 0 <= min_lr <= initial_lr or epoch < 0:
        raise ValueError(
            "Cosine scheduling requires positive LR/decay, 0 <= min_lr <= initial_lr, and epoch >= 0"
        )
    min_ratio = float(min_lr) / float(initial_lr)
    progress = min(float(epoch) / float(decay_epochs), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_lr_scheduler(optimizer, config):
    schedule = str(config["lr_schedule"]).lower()
    if schedule in {"none", "constant"}:
        return None
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    initial_lr = float(optimizer.param_groups[0]["lr"])
    min_lr = float(config["min_lr"])
    decay_epochs = int(config["lr_decay_epochs"])
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_lr_multiplier(
            epoch,
            initial_lr,
            min_lr,
            decay_epochs,
        ),
    )
