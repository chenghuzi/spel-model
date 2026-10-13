"""Coordinate single-task and sequential training phases."""

import torch
import wandb

from environments.dataset import read_data
from environments.numerosity import NUMEROSITY_TASK
from models.config import get_model_metadata
from models.evaluation import eval_model
from models.factory import build_model
from models.leaky import LeakyRNN
from models.optim import build_optimizer
from models.storage import phase_name, wandb_directory
from models.training import train_model


def train_one(exp_config, device, output_dir, resume=False):
    sim_config = exp_config["sim_config"]

    # generate training samples.
    dt = sim_config["dt"]
    total_t = int(exp_config["total_t"])
    trial_type = exp_config["trial_type"]
    task_name = exp_config["task"]
    task_metadata = get_model_metadata(exp_config)

    n_in = task_metadata["n_in"]
    n_o = task_metadata["n_o"]
    n_h = exp_config["n_hidden"]
    lr = exp_config["lr"]

    model = build_model(n_in, n_h, n_o, sim_config)
    model.to(device)
    if resume:
        model.load_state_dict(torch.load(exp_config["chkpt"]))
        print("loaded checkpoint from {}".format(exp_config["chkpt"]))
    optimizer = build_optimizer(model, lr, sim_config)

    batch_size = exp_config["batch_size"]
    epochs = exp_config["epochs"]
    exp_config["optimizer"] = str(type(optimizer).__name__)

    if exp_config["sync2wandb"] is True:
        wandb_run = wandb.init(
            dir=str(wandb_directory()),
            project="New_SPEL3",
            entity="chenghuzi",
            config=exp_config,
            resume="never",
        )

    ep_r, run_name = read_data(exp_config, trial_type, total_t, dt)
    model, _ = train_model(
        exp_config,
        ep_r,
        epochs,
        batch_size,
        model,
        optimizer,
        device,
        run_name,
        n_in,
        n_o,
        sim_config,
        task_name=task_name,
        response_start_ms=task_metadata["response_start_ms"],
        output_dir=output_dir,
    )

    if exp_config["sync2wandb"] is True:
        wandb_run.finish()


def train_sequence(exp_config, device, output_dir, predictall=True):
    sim_config = exp_config["sim_config"]
    seed = exp_config["seed"]

    sim_config["predict_all"] = predictall
    # The numerosity model predicts latent inputs through its dedicated heads.
    if sim_config["rnn_type"] == "CNNLeakyRNN":
        sim_config["predict_all"] = False
    # generate training samples.
    dt = sim_config["dt"]
    total_t = int(exp_config["total_t"])
    task_name = exp_config["task"]
    task_metadata = get_model_metadata(exp_config)
    type_order = "pb"  # first prepotent, then balanced
    # type_order = "bp"  # first balanced, then prepotent
    if task_name in {NUMEROSITY_TASK}:
        trial_types = [exp_config["trial_type"]]
    elif type_order == "pb":
        trial_types = ["prepotent", "balanced"]
    elif type_order == "bp":
        trial_types = ["balanced", "prepotent"]
    else:
        raise ValueError("type_order must be 'pb' or 'bp'")

    n_in = task_metadata["n_in"]
    n_o = task_metadata["n_o"]
    n_h = exp_config["n_hidden"]
    lr = exp_config["lr"]

    model = build_model(n_in, n_h, n_o, sim_config)
    model.to(device)
    if isinstance(model, LeakyRNN):
        model.h2hmask = model.h2hmask.to(device)

    resume = bool(exp_config["resume_training"])
    if resume:
        model.load_state_dict(
            torch.load(exp_config["chkpt"], map_location=device, weights_only=True)
        )
        print("loaded checkpoint from {}".format(exp_config["chkpt"]))
    optimizer = build_optimizer(model, lr, sim_config)

    batch_size = exp_config["batch_size"]
    epochs = exp_config["epochs"]
    exp_config["optimizer"] = str(type(optimizer).__name__)
    stop_threshold = exp_config["stop_threshold"]

    if exp_config["sync2wandb"] is True:
        wandb_projects = {
            "axby": "New_SPEL-Seq2",
            NUMEROSITY_TASK: "New_SPEL-Seq2-numerosity",
        }
        wandb_project = wandb_projects[task_name]
        wandb_run = wandb.init(
            dir=str(wandb_directory()),
            project=wandb_project,
            entity="chenghuzi",
            config=exp_config,
            resume="never",
        )
    # phase 1 training
    trial_type = trial_types[0]
    exp_config["trial_type"] = trial_type
    # exp_config['total_t'] = int(1e6)
    ep_r, run_name = read_data(exp_config, trial_type, total_t, dt)
    model, ep_used_1 = train_model(
        exp_config,
        ep_r,
        epochs,
        batch_size,
        model,
        optimizer,
        device,
        run_name,
        n_in,
        n_o,
        sim_config,
        save_step=task_name in {NUMEROSITY_TASK},
        stop_threshold=stop_threshold,
        task_name=task_name,
        response_start_ms=task_metadata["response_start_ms"],
        control_eval_config=exp_config["control_eval"],
        output_dir=output_dir,
    )
    torch.save(
        model.state_dict(),
        output_dir / f"{phase_name(run_name)}-last.pt",
    )
    eval_model(
        exp_config,
        sim_config,
        seed,
        model=model,
        rec_suffix=type_order,
        device=device,
        output_dir=output_dir,
    )

    if task_name in {NUMEROSITY_TASK}:
        if exp_config["sync2wandb"] is True:
            wandb_run.finish()
        return

    # phase 2 training
    trial_type = trial_types[1]
    exp_config["trial_type"] = trial_type
    # exp_config['total_t'] = int(2e5)
    ep_r, run_name = read_data(exp_config, trial_type, total_t, dt)

    model, ep_used_2 = train_model(
        exp_config,
        ep_r,
        ep_used_1,
        batch_size,
        model,
        optimizer,
        device,
        run_name,
        n_in,
        n_o,
        sim_config,
        save_step=task_name in {NUMEROSITY_TASK},
        stop_threshold=1.2,  # set to 1.2 to make sure it runs to the end.
        task_name=task_name,
        response_start_ms=task_metadata["response_start_ms"],
        control_eval_config=exp_config["control_eval"],
        output_dir=output_dir,
    )
    torch.save(
        model.state_dict(),
        output_dir / f"{phase_name(run_name)}-last.pt",
    )
    eval_model(
        exp_config,
        sim_config,
        seed,
        model=model,
        rec_suffix=type_order,
        device=device,
        output_dir=output_dir,
    )

    if exp_config["sync2wandb"] is True:
        wandb_run.finish()
