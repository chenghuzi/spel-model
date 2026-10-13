"""Training loop with explicit configuration and output ownership."""

from collections import Counter, defaultdict

import numpy as np
import torch
import wandb
from tqdm import tqdm

from environments.dataset import iter_epoch_batches, make_sequence_mask
from environments.numerosity import NUMEROSITY_TASK
from environments.trials import get_trial_relation
from models.evaluation import run_control_eval
from models.losses import (
    get_auxiliary_loss_metrics,
    get_loss_metrics,
    get_output_weight_penalty,
    get_recurrent_weight_norms,
)
from models.metrics import (
    get_prediction_error_metrics,
    has_required_rt_coverage,
    print_rt_summary,
    update_relation_corrects,
)
from models.optim import build_lr_scheduler
from models.storage import checkpoint_path, phase_name
from models.visual import CNNLeakyRNN
from visualizations.trials import plot_trial


def train_model(
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
    output_dir,
    save_step=False,
    stop_threshold=0.99,
    min_epochs=50,
    task_name="axby",
    response_start_ms=2500,
    control_eval_config=None,
):

    loss_list = []
    trial_correct_rates = []
    relation_correct_rates = []

    RTs = defaultdict(list)
    relation_corrects = defaultdict(list)
    phbar = tqdm(range(1, epochs + 1), dynamic_ncols=True)
    ep_used = 0
    global_steps = 0
    last_control_eval_ep = 0
    lr_scheduler = build_lr_scheduler(optimizer, sim_config)
    grad_clip_norm = float(sim_config["grad_clip_norm"])
    checkpoint_interval = int(sim_config["checkpoint_interval_epochs"])
    plot_interval = int(sim_config["plot_interval_epochs"])
    if save_step and bool(sim_config["save_initial_checkpoint"]):
        torch.save(model.state_dict(), checkpoint_path(output_dir, run_name, 0))
    for ep in phbar:
        ep_used = ep
        epoch_episodes = ep_r
        for ep_r_batched in iter_epoch_batches(
            epoch_episodes,
            batch_size,
            task_name,
            sim_config,
        ):
            batch = [ep_r_b[0] for ep_r_b in ep_r_batched]
            trial_types = [str(ep_r_b[1]) for ep_r_b in ep_r_batched]

            real_bs = len(batch)
            global_steps += real_bs
            lengths = [ep.shape[0] for ep in batch]
            batched_tensor = torch.zeros(real_bs, max(lengths), n_in + n_o)

            for i in range(real_bs):
                batched_tensor[i, : lengths[i]] = torch.from_numpy(batch[i]).float()
                pass

            batched_tensor = batched_tensor.to(device)
            stimuli = batched_tensor[:, :, :n_in]
            actions = batched_tensor[:, :, n_in:]
            mask = make_sequence_mask(lengths, stimuli.shape[1], stimuli.device)

            # stimuli_mask = stimuli.sum(dim=-1).data.unsqueeze(2).repeat(1, 1, stimuli.shape[-1])
            stimuli = stimuli + (
                sim_config["stimuli_noise"] * torch.randn_like(stimuli) * mask.unsqueeze(-1)
            )

            optimizer.zero_grad()
            model_kwargs = {
                "actions": actions,
                "teacher_forcing": bool(sim_config["teacher_forcing"]),
            }
            if isinstance(model, CNNLeakyRNN):
                model_kwargs["mask"] = mask
            noises, hiddens, outputs, xs = model(stimuli, **model_kwargs)

            (
                trial_corrects,
                pred_loss,
                homeostatic_loss,
                RTs,
                _,
                cognitive_metrics,
            ) = get_loss_metrics(
                sim_config,
                real_bs,
                lengths,
                trial_types,
                outputs,
                actions,
                hiddens,
                RTs,
                global_steps,
                stimuli,
                sim_config["predict_all"],
                sim_config["dt"],
                task_name=task_name,
                response_start_ms=response_start_ms,
                numerosity_config=exp_config["numerosity_config"],
                log_to_wandb=exp_config["sync2wandb"],
            )

            trial_correct_rate = sum(trial_corrects) / real_bs
            trial_correct_rates.append(trial_correct_rate)
            relation_metrics = update_relation_corrects(
                relation_corrects, trial_types, trial_corrects, task_name
            )
            if "balanced_correct_rate" in relation_metrics:
                relation_correct_rates.append(relation_metrics["balanced_correct_rate"])

            pred_loss = pred_loss / real_bs

            homeostatic_loss = homeostatic_loss / real_bs

            l2_norm_wI, l2_norm_wH, l2_norm_wO = get_recurrent_weight_norms(
                model,
                device,
            )
            output_weight_penalty = get_output_weight_penalty(model, device, sim_config)
            latent_pred_loss = get_auxiliary_loss_metrics(model, device)
            context_pred_loss = (
                model.latest_context_pred_loss
                if isinstance(model, CNNLeakyRNN)
                else torch.zeros((), device=device)
            )

            # Combine the behavior loss with optional latent predictive losses;
            # legacy models get zero auxiliary terms and keep their old objective.
            loss = (
                sim_config["action_pred_weight"] * pred_loss
                + sim_config["latent_pred_weight"] * latent_pred_loss
                + sim_config["context_pred_weight"] * context_pred_loss
                + sim_config["l2_h"] * homeostatic_loss
                + sim_config["l2_wI"] * l2_norm_wI
                + sim_config["l2_wR"] * l2_norm_wH
                + sim_config["l2_wO"] * output_weight_penalty
            )

            loss.backward()
            if grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )
            else:
                grad_norm = torch.zeros((), device=device)
            optimizer.step()
            if isinstance(model, CNNLeakyRNN):
                model.update_target_encoder()
            loss_list.append(loss.item())

            loss_info = {
                "global_steps": global_steps,
                "correct_rate": trial_correct_rate,
                "loss": loss.item(),
                "pred_loss": pred_loss.item(),
                "homeostatic_loss": homeostatic_loss.item(),
                "l2_norm_wI": l2_norm_wI.item(),
                "l2_norm_wH": l2_norm_wH.item(),
                "l2_norm_wO": l2_norm_wO.item(),
                "l2_penalty_wO": output_weight_penalty.item(),
                "grad_norm": float(grad_norm.detach().item()),
                "grad_clip_norm": grad_clip_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "latent_pred_loss": latent_pred_loss.item(),
                "context_pred_loss": context_pred_loss.item(),
            }
            if len(optimizer.param_groups) > 1:
                loss_info["readout_learning_rate"] = float(optimizer.param_groups[1]["lr"])
            loss_info.update(get_prediction_error_metrics(model, xs, mask))
            loss_info.update(cognitive_metrics)
            loss_info.update(relation_metrics)
            if task_name == NUMEROSITY_TASK:
                batch_relation_counts = Counter(
                    get_trial_relation(trial_type) for trial_type in trial_types
                )
                loss_info["batch_match_fraction"] = batch_relation_counts["match"] / real_bs
            phbar.set_postfix(loss_info)
            if exp_config["sync2wandb"] is True:
                wandb.log(loss_info)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if save_step and ep % checkpoint_interval == 0:
            torch.save(model.state_dict(), checkpoint_path(output_dir, run_name, ep))

        # visualize one trial
        if ep % plot_interval == 0:
            if has_required_rt_coverage(RTs, task_name):
                print_rt_summary(RTs, task_name, "RT")
            plot_trial(
                hiddens,
                outputs,
                xs,
                lengths,
                output_dir / f"{phase_name(run_name)}-ep{ep:06d}.png",
                title=f"epoch {ep}, sample ({trial_types[0]})",
            )

        if task_name == NUMEROSITY_TASK and control_eval_config is not None:
            interval = int(control_eval_config["interval_epochs"])
            if control_eval_config["enabled"] and ep % interval == 0:
                control_eval_results = run_control_eval(
                    exp_config,
                    model,
                    batch_size,
                    n_in,
                    n_o,
                    sim_config,
                    {
                        "response_start_ms": response_start_ms,
                    },
                    ep,
                    global_steps,
                    control_eval_config,
                )
                last_control_eval_ep = ep
                standard_metrics = control_eval_results["standard"]
                real_stop_rate = standard_metrics["balanced_correct_rate"]
                if real_stop_rate >= stop_threshold:
                    print(
                        "early stop: standard closed-loop balanced_correct_rate "
                        f"{real_stop_rate:.4f} >= {stop_threshold:.4f}"
                    )
                    break
        else:
            stop_rates = trial_correct_rates
            if len(stop_rates) > 100 and ep > min_epochs:  # minimal
                if np.mean(stop_rates[-100:]) >= stop_threshold:
                    if has_required_rt_coverage(RTs, task_name):
                        print("early stop")
                        break

    torch.save(model.state_dict(), checkpoint_path(output_dir, run_name, ep))
    if (
        task_name == NUMEROSITY_TASK
        and control_eval_config is not None
        and control_eval_config["enabled"]
        and last_control_eval_ep != ep_used
    ):
        run_control_eval(
            exp_config,
            model,
            batch_size,
            n_in,
            n_o,
            sim_config,
            {
                "response_start_ms": response_start_ms,
            },
            ep_used,
            global_steps,
            control_eval_config,
        )

    return model, ep_used
