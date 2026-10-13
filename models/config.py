"""Load experiment settings and derive model/task dimensions."""

import json5 as json

from environments.metadata import get_task_metadata
from environments.numerosity import NUMEROSITY_TASK, get_numerosity_config
from environments.stimuli import NUMEROSITY_STIMULUS_MODES
from models.losses import get_action_loss_config


def read_config(fin):
    # Resolve supported defaults once at the file boundary; internal code uses required keys.
    with open(fin) as handle:
        conf = json.load(handle)
    conf["control_eval"] = get_control_eval_config(conf)
    conf = {
        "task": "axby",
        "numerosity_config": None,
        "eval_total_t": 10000000,
        "eval_closed_loop_trials": 160,
        "training_cache_format": "torch",
        "cache_training_data": True,
        "load_training_cache_to_memory": False,
        "resume_training": False,
        "sync2wandb": False,
        **conf,
    }
    sim = {
        "activation": "softplus",
        "activation_peak": 20.0,
        "readout": "linear",
        "image_channels": 1,
        "cnn_channels": [16, 32, 64],
        "rnn_cell": "leaky",
        "prediction_error_input": False,
        "use_ema_target_encoder": True,
        "latent_ema_decay": 0.99,
        "latent_activation": "none",
        "zero_delay_latent": False,
        "dropout": 0.0,
        "teacher_forcing": True,
        "balanced_relation_batches": False,
        "action_pred_weight": 1.0,
        "latent_pred_weight": 0.0,
        "context_pred_weight": 0.0,
        "output_weight_regularizer": "norm",
        "grad_clip_norm": 0.0,
        "checkpoint_interval_epochs": 10,
        "plot_interval_epochs": 10,
        "save_initial_checkpoint": False,
        "lr_schedule": "none",
        "min_lr": 0.0,
        "lr_decay_epochs": 0,
        **conf["sim_config"],
    }
    conf["sim_config"] = sim
    sim["action_loss"] = get_action_loss_config(sim)

    # Reject unsupported experiment choices instead of changing the model or training policy.
    if conf["task"] not in {"axby", NUMEROSITY_TASK}:
        raise ValueError(f"Unsupported task: {conf['task']}")
    if sim["rnn_type"] not in {"LeakyRNN", "CNNLeakyRNN"}:
        raise ValueError(f"Unsupported rnn_type: {sim['rnn_type']}")
    if conf["task"] == NUMEROSITY_TASK:
        conf["numerosity_config"] = get_numerosity_config(conf["numerosity_config"])
    for key in ("checkpoint_interval_epochs", "plot_interval_epochs"):
        if int(sim[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if str(conf["training_cache_format"]).lower() not in {"torch", "hdf5"}:
        raise ValueError(f"Unsupported training_cache_format: {conf['training_cache_format']}")
    sim["alpha"] = float(sim["dt"] / sim["tau"])
    sim["alpha_noise"] = float(sim["dt"] / sim["tau_noise"])
    get_model_metadata(conf)
    return conf


def configure_task_model_config(
    task_name,
    config,
    task_metadata,
    numerosity_config=None,
):
    # Match the behavior readout and phase timing to the selected task.
    config["action_output_dim"] = int(task_metadata["n_o"])

    if task_name == NUMEROSITY_TASK:
        for timing_key in (
            "fixation_ms",
            "sample_ms",
            "delay_ms",
            "test_ms",
        ):
            config[timing_key] = numerosity_config[timing_key]
        config["phase_context_dim"] = 4
        config["trial_end_ms"] = int(task_metadata["trial_end_ms"])

        if config["rnn_type"] == "CNNLeakyRNN":
            # Numerosity CNN/RNN paths receive only visual input, non-leaky phase
            # context, and action history. Quantity/relation cognitive heads
            # would leak labels, so keep the readout behavior-only.
            config["predict_all"] = False
            return


def get_model_metadata(exp_config):
    sim_config = exp_config["sim_config"]
    # Collect task dimensions and response timing from the task module.
    task_name = exp_config["task"]
    numerosity_config = exp_config["numerosity_config"]
    task_metadata = get_task_metadata(
        task_name=task_name,
        dt=sim_config["dt"],
        numerosity_config=numerosity_config,
    )
    configure_task_model_config(
        task_name,
        sim_config,
        task_metadata,
        numerosity_config=numerosity_config,
    )
    return task_metadata


def get_control_eval_config(config):
    # Normalize the optional control-evaluation section before training consumes it.
    default_modes = ["standard", "equal_area", "high_density", "low_density"]
    control_config = {
        "enabled": False,
        "interval_epochs": 10,
        "total_t": int(config.get("eval_total_t", config["total_t"])),
        "max_trials_per_type": 80,
        "modes": default_modes,
        "seed_offset": 1000,
    }
    control_config.update(config.get("control_eval", {}))
    control_config.setdefault(
        "closed_loop_trials", max(20, control_config["max_trials_per_type"] * 4)
    )
    if control_config["enabled"]:
        if control_config["interval_epochs"] <= 0 or control_config["closed_loop_trials"] <= 0:
            raise ValueError("Control evaluation requires positive intervals and trial counts")
        if "standard" not in control_config["modes"]:
            raise ValueError("Control evaluation requires standard mode for the stopping criterion")
        if not set(control_config["modes"]) <= set(NUMEROSITY_STIMULUS_MODES):
            raise ValueError(f"Unsupported control modes: {control_config['modes']}")
    return control_config
