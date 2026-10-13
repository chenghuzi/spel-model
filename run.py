"""Command-line entry point for manuscript training and evaluation."""

import click
import numpy as np
import torch

from models.config import get_model_metadata, read_config
from models.evaluation import eval_model
from models.runner import train_one, train_sequence
from models.storage import run_directory


@click.group()
@click.argument("conf", type=click.Path(exists=True))
@click.option("--seedx", default=-1, type=int)
@click.option("--checkpoint", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto")
@click.option("--run-id", default=None, help="Optional short output directory ID.")
@click.pass_context
def cli(ctx, conf, seedx, checkpoint, device, run_id):
    # Keep configuration and output ownership in this invocation, never in module globals.
    config = read_config(conf)
    if seedx >= 0:
        config["seed"] = seedx
    if checkpoint is not None:
        config["chkpt"] = checkpoint
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    ctx.obj = {"config": config, "device": device, "run_id": run_id}


@cli.command("train1")
@click.option("--resume/--no-resume", default=False)
@click.pass_obj
def train1(context, resume):
    config = context["config"]
    directory = run_directory(config, "train1", context["run_id"])
    train_one(config, context["device"], directory, resume=resume)


@cli.command("train2")
@click.option("--predictall", default=True, type=bool)
@click.pass_obj
def train2(context, predictall):
    config = context["config"]
    config["sim_config"]["predict_all"] = predictall
    get_model_metadata(config)
    directory = run_directory(config, "train2", context["run_id"])
    train_sequence(config, context["device"], directory, predictall=predictall)


@cli.command("eval")
@click.pass_obj
def evaluate(context):
    config = context["config"]
    directory = run_directory(config, "eval", context["run_id"])
    eval_model(
        config, config["sim_config"], config["seed"], device=context["device"], output_dir=directory
    )


if __name__ == "__main__":
    cli()
