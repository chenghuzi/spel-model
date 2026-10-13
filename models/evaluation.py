"""Teacher-forced and closed-loop evaluation with recording export."""

import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from environments.dataset import episode_cache, make_sequence_mask
from environments.episodes import action_id_to_vector, gen_training_samples
from environments.numerosity import NUMEROSITY_TASK, build_numerosity_env
from environments.random import restore_rng_snapshot, rng_snapshot
from environments.trials import get_trial_relation
from models.config import get_model_metadata
from models.factory import build_model
from models.losses import (
    get_auxiliary_loss_metrics,
    get_loss_metrics,
    get_output_weight_penalty,
    get_recurrent_weight_norms,
)
from models.metrics import (
    get_action_predictions,
    has_required_rt_coverage,
    print_rt_summary,
    summarize_relation_corrects,
    update_relation_corrects,
)
from models.storage import phase_name
from models.visual import CNNLeakyRNN
from visualizations.trials import plot_trial


def evaluate_episode_set(
    exp_config,
    ep_r,
    model,
    batch_size,
    n_in,
    n_o,
    sim_config,
    task_metadata,
    numerosity_config,
    eval_seed=None,
):
    # Reject an empty dataset at the evaluation boundary.
    if len(ep_r) == 0:
        raise ValueError("Episode evaluation requires at least one episode")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    state = rng_snapshot()

    all_corrects = []
    relation_corrects = defaultdict(list)
    cognitive_sums = defaultdict(float)
    cognitive_counts = defaultdict(float)
    loss_values = []

    try:
        if eval_seed is not None:
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
        with torch.no_grad():
            for b_id in range(0, len(ep_r), batch_size):
                ep_r_batched = ep_r[b_id : b_id + batch_size]
                batch = [ep_r_b[0] for ep_r_b in ep_r_batched]
                trial_types = [str(ep_r_b[1]) for ep_r_b in ep_r_batched]
                real_bs = len(batch)
                lengths = [episode.shape[0] for episode in batch]
                batched_tensor = torch.zeros(real_bs, max(lengths), n_in + n_o)
                for i, episode in enumerate(batch):
                    batched_tensor[i, : lengths[i]] = torch.from_numpy(episode).float()

                batched_tensor = batched_tensor.to(device)
                stimuli = batched_tensor[:, :, :n_in]
                actions = batched_tensor[:, :, n_in:]
                mask = make_sequence_mask(lengths, stimuli.shape[1], stimuli.device)
                stimuli = stimuli + (
                    sim_config["stimuli_noise"] * torch.randn_like(stimuli) * mask.unsqueeze(-1)
                )

                model_kwargs = {
                    "actions": actions,
                    "teacher_forcing": bool(sim_config["teacher_forcing"]),
                }
                if isinstance(model, CNNLeakyRNN):
                    model_kwargs["mask"] = mask
                _, hiddens, outputs, _ = model(stimuli, **model_kwargs)
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
                    defaultdict(list),
                    0,
                    stimuli,
                    sim_config["predict_all"],
                    sim_config["dt"],
                    task_name=NUMEROSITY_TASK,
                    response_start_ms=task_metadata["response_start_ms"],
                    numerosity_config=numerosity_config,
                    log_to_wandb=False,
                )

                all_corrects.extend([float(value) for value in trial_corrects])
                update_relation_corrects(
                    relation_corrects, trial_types, trial_corrects, NUMEROSITY_TASK
                )
                for key, value in cognitive_metrics.items():
                    cognitive_sums[key] += float(value) * real_bs
                    cognitive_counts[key] += real_bs
                loss_values.append(float((pred_loss / real_bs).detach().cpu()))
    finally:
        restore_rng_snapshot(state)
        if was_training:
            model.train()

    metrics = {
        "correct_rate": float(np.mean(all_corrects)),
        "pred_loss": float(np.mean(loss_values)),
    }
    metrics.update(summarize_relation_corrects(relation_corrects))
    for key, value in cognitive_sums.items():
        metrics[key] = float(value / cognitive_counts[key])
    return metrics


def make_numerosity_env(numerosity_config, mode, dt):
    config = dict(numerosity_config)
    config["stimulus_mode"] = mode
    return build_numerosity_env(dt, config)


def numerosity_target_action_for_time(info, time_ms, numerosity_config, device, dtype):
    target = torch.zeros(1, 2, device=device, dtype=dtype)
    response_start = (
        numerosity_config["fixation_ms"]
        + numerosity_config["sample_ms"]
        + numerosity_config["delay_ms"]
    )
    if time_ms > response_start and info["is_match"]:
        target[:, 1] = 1.0
    else:
        target[:, 0] = 1.0
    return target


def run_closed_loop_numerosity_eval(
    model,
    sim_config,
    numerosity_config,
    mode,
    n_trials,
    eval_seed=None,
):
    # Trial count is an evaluation API input, not a reason to manufacture zero-valued metrics.
    if n_trials <= 0:
        raise ValueError("Closed-loop evaluation requires at least one trial")

    was_training = model.training
    model.eval()
    state = rng_snapshot()
    if eval_seed is not None:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)

    env = make_numerosity_env(numerosity_config, mode, sim_config["dt"])
    model_device = next(model.parameters()).device
    all_corrects = []
    relation_corrects = defaultdict(list)
    action_losses = []
    decision_accuracies = []
    decision_margins = []
    trial_match_predictions = []

    try:
        with torch.no_grad():
            for _ in range(int(n_trials)):
                obs, info = env.reset()
                trial_type = info["type"]
                noises, hiddens, current_prediction, previous_action = model.initial_state(
                    1,
                    model_device,
                    torch.float32,
                )
                reward = 0
                done = False
                max_steps = int(env.trial_end_ms / env.dt) + 5

                for _step in range(max_steps):
                    stimulus = torch.from_numpy(obs).float().reshape(1, -1).to(model_device)
                    phase_context = model.phase_context_from_time(
                        [env.t],
                        stimulus.device,
                        stimulus.dtype,
                    )
                    (
                        noises,
                        hiddens,
                        prediction,
                        _prediction_error,
                        _latent,
                    ) = model.step_once(
                        stimulus,
                        phase_context,
                        previous_action,
                        current_prediction,
                        noises,
                        hiddens,
                    )
                    current_prediction = prediction
                    action_pred = get_action_predictions(
                        prediction,
                        model.action_output_dim,
                        sim_config,
                    )
                    target_action = numerosity_target_action_for_time(
                        info,
                        env.t,
                        numerosity_config,
                        stimulus.device,
                        stimulus.dtype,
                    )
                    action_losses.append(
                        F.mse_loss(action_pred, target_action).detach().cpu().item()
                    )
                    action_probs = F.softmax(action_pred, dim=1)
                    decoded_action = int(action_probs.argmax(dim=1).item())
                    if env.t > env.response_start_ms:
                        expected_match = bool(info["is_match"])
                        predicted_match = decoded_action == 1
                        decision_accuracies.append(float(predicted_match == expected_match))
                        decision_margins.append(
                            float((action_pred[:, 1] - action_pred[:, 0]).item())
                        )
                        trial_match_predictions.append(float(predicted_match))

                    action_id = decoded_action
                    obs, reward, terminated, truncated, info = env.step(action_id)
                    previous_action_np = action_id_to_vector(
                        action_id,
                        action_dim=model.action_output_dim,
                    )
                    previous_action = (
                        torch.from_numpy(previous_action_np)
                        .float()
                        .reshape(1, -1)
                        .to(stimulus.device)
                    )
                    done = terminated or truncated
                    if done:
                        break

                # why: exhausting the loop without termination means the environment contract broke.
                assert done, "Closed-loop trial exceeded its termination bound"
                trial_correct = bool(reward == 1)
                all_corrects.append(float(trial_correct))
                relation = get_trial_relation(trial_type)
                relation_corrects[relation].append(float(trial_correct))
    finally:
        restore_rng_snapshot(state)
        if was_training:
            model.train()

    metrics = {
        "correct_rate": float(np.mean(all_corrects)),
        "pred_loss": float(np.mean(action_losses)),
        "action_mse_all": float(np.mean(action_losses)),
        "closed_loop_trials": int(n_trials),
    }
    if decision_accuracies:
        metrics["decision_accuracy_from_logits"] = float(np.mean(decision_accuracies))
        metrics["decision_margin_release_minus_hold"] = float(np.mean(decision_margins))
        metrics["trial_match_rate"] = float(np.mean(trial_match_predictions))
    metrics.update(summarize_relation_corrects(relation_corrects))
    return metrics


def run_control_eval(
    exp_config,
    model,
    batch_size,
    n_in,
    n_o,
    sim_config,
    task_metadata,
    epoch,
    global_steps,
    control_config,
):
    if not control_config["enabled"]:
        return {}

    modes = control_config["modes"]
    seed_offset = int(control_config["seed_offset"])

    results = {}
    log_info = {"global_steps": global_steps, "eval/epoch": epoch}
    base_numerosity_config = exp_config["numerosity_config"]
    for mode_index, mode in enumerate(modes):
        numerosity_config = dict(base_numerosity_config)
        numerosity_config["stimulus_mode"] = mode
        mode_seed = exp_config["seed"] + seed_offset + mode_index
        closed_loop_trials = int(control_config["closed_loop_trials"])
        metrics = run_closed_loop_numerosity_eval(
            model,
            sim_config,
            numerosity_config,
            mode,
            closed_loop_trials,
            eval_seed=mode_seed + 50000,
        )
        results[mode] = metrics
        for key, value in metrics.items():
            log_info[f"eval/{mode}/{key}"] = value

    print(
        "Control eval:",
        {
            mode: {
                key: round(value, 4)
                for key, value in metrics.items()
                if key in {"correct_rate", "balanced_correct_rate"}
            }
            for mode, metrics in results.items()
        },
    )
    if exp_config["sync2wandb"]:
        wandb.log(log_info)
    return results


def eval_model(
    exp_config,
    sim_config,
    seed,
    model=None,
    external_chkpt=None,
    rec_suffix="",
    *,
    device,
    output_dir,
):
    sync2wandb = exp_config["sync2wandb"]
    exp_config["sync2wandb"] = False  # no sync to wandb when eval
    if external_chkpt is not None:
        exp_config["chkpt"] = external_chkpt

    # generate training samples.
    dt = sim_config["dt"]
    total_t = int(exp_config["eval_total_t"])
    trial_type = exp_config["trial_type"]
    task_name = exp_config["task"]
    task_metadata = get_model_metadata(exp_config)

    cache_f = episode_cache(
        exp_config,
        trial_type,
        total_t,
        dt,
        "episodes.pt",
        seed_value=seed + 1,
        purpose="evaluation",
    )
    if cache_f.is_file():
        print(f"load from cache {cache_f}")
        res = torch.load(cache_f, weights_only=False)
        ep_r = res["ep_r"]
        cr = res["cr"]
    else:
        ep_r, cr = gen_training_samples(
            trial_type,
            total_t,
            dt=dt,
            task_name=task_name,
            numerosity_config=exp_config["numerosity_config"],
        )

        trial_counts = Counter([str(ep[1]) for ep in ep_r])
        min_trial_count = min(min(trial_counts.values()), 200)
        new_ep_r = []
        new_trial_counts = Counter()
        for ep, c_trial_type in ep_r:
            if new_trial_counts[c_trial_type] >= min_trial_count:
                continue
            new_ep_r.append((ep, c_trial_type))
            new_trial_counts[c_trial_type] += 1

        ep_r = new_ep_r
        torch.save({"ep_r": ep_r, "cr": cr}, cache_f)

    trial_counts = Counter([str(ep[1]) for ep in ep_r])
    print("trial_counts:", trial_counts)

    n_in = task_metadata["n_in"]
    n_o = task_metadata["n_o"]
    n_h = exp_config["n_hidden"]
    if model is None:
        model = build_model(n_in, n_h, n_o, sim_config)
        model.to(device)

        assert "chkpt" in exp_config and exp_config["chkpt"] is not None, (
            "without chkpt you cannot eval"
        )
        model.load_state_dict(
            torch.load(exp_config["chkpt"], map_location=device, weights_only=True)
        )
        print("loaded checkpoint from {}".format(exp_config["chkpt"]))

    batch_size = exp_config["batch_size"]
    if task_name == NUMEROSITY_TASK:
        closed_loop_trials = int(exp_config["eval_closed_loop_trials"])
        closed_loop_metrics = run_closed_loop_numerosity_eval(
            model,
            sim_config,
            exp_config["numerosity_config"],
            exp_config["numerosity_config"]["stimulus_mode"],
            closed_loop_trials,
            eval_seed=seed + 90000,
        )
        print("Closed-loop eval:", closed_loop_metrics)

    loss_list = []
    trial_correct_rates = []
    RTs = defaultdict(list)
    relation_corrects = defaultdict(list)

    global_steps = 0
    recording_all = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for b_id in range(0, len(ep_r), batch_size):
            ep_r_batched = ep_r[b_id : b_id + batch_size]
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
                RT_list,
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
                response_start_ms=task_metadata["response_start_ms"],
                numerosity_config=exp_config["numerosity_config"],
                log_to_wandb=exp_config["sync2wandb"],
            )
            # record
            for i in range(real_bs):
                recording_all[trial_types[i]]["stimuli"].append(
                    stimuli[i, : lengths[i]].data.cpu().numpy()
                )
                recording_all[trial_types[i]]["actions"].append(
                    actions[i, : lengths[i]].data.cpu().numpy()
                )
                recording_all[trial_types[i]]["outputs"].append(
                    outputs[i, : lengths[i]].data.cpu().numpy()
                )
                recording_all[trial_types[i]]["hiddens"].append(
                    hiddens[i, : lengths[i]].data.cpu().numpy()
                )
                recording_all[trial_types[i]]["xs"].append(xs[i, : lengths[i]].data.cpu().numpy())
                recording_all[trial_types[i]]["corrects"].append(trial_corrects[i])
                recording_all[trial_types[i]]["RT"].append(RT_list[i])

            trial_correct_rate = sum(trial_corrects) / real_bs
            trial_correct_rates.append(trial_correct_rate)
            relation_metrics = update_relation_corrects(
                relation_corrects, trial_types, trial_corrects, task_name
            )

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

            # Mirror the training objective during evaluation so logged losses
            # stay comparable for latent and legacy model families.
            loss = (
                sim_config["action_pred_weight"] * pred_loss
                + sim_config["latent_pred_weight"] * latent_pred_loss
                + sim_config["context_pred_weight"] * context_pred_loss
                + sim_config["l2_h"] * homeostatic_loss
                + sim_config["l2_wI"] * l2_norm_wI
                + sim_config["l2_wR"] * l2_norm_wH
                + sim_config["l2_wO"] * output_weight_penalty
            )

            loss_list.append(loss.item())

            loss_info = {
                "global_steps": global_steps,
                "loss": loss.item(),
                "pred_loss": pred_loss.item(),
                "homeostatic_loss": homeostatic_loss.item(),
                "l2_norm_wI": l2_norm_wI.item(),
                "l2_norm_wH": l2_norm_wH.item(),
                "l2_norm_wO": l2_norm_wO.item(),
                "l2_penalty_wO": output_weight_penalty.item(),
                "latent_pred_loss": latent_pred_loss.item(),
                "context_pred_loss": context_pred_loss.item(),
                "correct_rate": trial_correct_rate,
            }
            loss_info.update(cognitive_metrics)
            loss_info.update(relation_metrics)
            if task_name == NUMEROSITY_TASK:
                batch_relation_counts = Counter(
                    get_trial_relation(trial_type) for trial_type in trial_types
                )
                loss_info["batch_match_fraction"] = batch_relation_counts["match"] / real_bs

            if exp_config["sync2wandb"] is True:
                wandb.log(loss_info)

        print(
            "Correct rate: mean:",
            np.mean(trial_correct_rates),
            "std:",
            np.std(trial_correct_rates),
        )
        relation_metrics = summarize_relation_corrects(relation_corrects)
        if relation_metrics:
            print("Relation correct rates:", relation_metrics)
        # visualize one trial

    # save recordings
    for k, v in recording_all.items():
        for kk, vv in v.items():
            recording_all[k][kk] = np.stack(vv)
        recording_all[k] = dict(recording_all[k])
    recording_all = dict(recording_all)

    recording_all_f = output_dir / f"rec_{phase_name(trial_type)}.pt"
    print(f"save recording_all to {recording_all_f}")
    torch.save(recording_all, recording_all_f)

    if has_required_rt_coverage(RTs, task_name):
        print_rt_summary(RTs, task_name, "Eval |")

    plot_trial(
        hiddens,
        outputs,
        xs,
        lengths,
        output_dir / f"eval_{phase_name(trial_type)}.png",
        title="Eval",
        actions=actions,
    )

    exp_config["sync2wandb"] = sync2wandb
