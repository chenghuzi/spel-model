"""Action supervision and model regularization with explicit configuration."""

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from environments.numerosity import NUMEROSITY_TASK
from environments.trials import get_expected_joystick, get_trial_relation
from models.leaky import LeakyRNN
from models.metrics import compute_go_nogo_trial_behavior, get_action_predictions


def get_auxiliary_loss_metrics(model, device):
    # DPX has no latent objective; visual models must provide the loss from their forward pass.
    if isinstance(model, LeakyRNN):
        return torch.zeros((), device=device)
    return model.latest_latent_pred_loss


def get_recurrent_weight_norms(model, device):
    # Both supported network families expose the same recurrent and readout layers.
    l2_norm_wI = torch.norm(model.i2h.weight, p=2)
    l2_norm_wH = torch.norm(model.h2h.weight, p=2)
    l2_norm_wO = torch.norm(model.h2o.weight, p=2)
    return l2_norm_wI, l2_norm_wH, l2_norm_wO


def get_output_weight_penalty(model, device, config):
    weights = [model.h2o.weight]
    regularizer = str(config["output_weight_regularizer"]).lower()
    if regularizer == "norm":
        return sum(torch.norm(weight, p=2) for weight in weights)
    if regularizer == "squared_action":
        return 0.5 * sum(weight.square().sum() for weight in weights)
    raise ValueError(f"Unsupported output_weight_regularizer: {regularizer}")


def get_action_loss_config(config):
    # Resolve optional weight inheritance at the configuration boundary, not per trial.
    action_config = config.get("action_loss", {})
    if isinstance(action_config, str):
        action_config = {"type": action_config}
    default_config = {
        "type": "mse",
        "fixation_weight": 1.0,
        "wait_weight": 1.0,
        "pre_response_weight": None,
        "decision_weight": 1.0,
        "window_event_weight": 0.0,
        "pre_event_weight": 0.0,
        "match_event_weight": None,
        "nonmatch_event_weight": None,
        "match_weight": 1.0,
        "nonmatch_weight": 1.0,
        "relation_balance": False,
    }
    default_config.update(action_config)
    if default_config["type"] not in {"mse", "decision_ce"}:
        raise ValueError(f"Unsupported action loss: {default_config['type']}")
    if default_config["pre_response_weight"] is None:
        default_config["pre_response_weight"] = default_config["wait_weight"]
    for key in ("match_event_weight", "nonmatch_event_weight"):
        if default_config[key] is None:
            default_config[key] = default_config["window_event_weight"]
    return default_config


def get_relation_loss_weights(trial_types, task_name, action_config):
    return get_relation_balance_weights(
        trial_types,
        task_name,
        action_config["relation_balance"],
    )


def get_relation_balance_weights(trial_types, task_name, enabled):
    if task_name not in {NUMEROSITY_TASK} or enabled is not True:
        return [1.0 for _ in trial_types]

    relation_counts = Counter(get_trial_relation(trial_type) for trial_type in trial_types)
    # Weight the relations actually observed; a one-relation evaluation batch naturally has weight one.
    relation_weights = {
        relation: len(trial_types) / (len(relation_counts) * count)
        for relation, count in relation_counts.items()
    }
    return [float(relation_weights[get_trial_relation(trial_type)]) for trial_type in trial_types]


def compute_action_supervision_loss(
    act_pred,
    actions_trial,
    action_config,
    relation_weight=1.0,
):
    action_mse_all = F.mse_loss(act_pred, actions_trial)
    # DPX uses the original element-wise action MSE.
    return action_mse_all, {
        "action_mse_all": action_mse_all.detach().item(),
    }


def compute_go_nogo_action_supervision_loss(
    act_pred,
    actions_trial,
    trial_type,
    response_start_ms,
    dt,
    action_config,
    relation_weight=1.0,
):
    action_mse_all = F.mse_loss(act_pred, actions_trial)
    if action_config["type"] == "mse":
        return action_mse_all, {
            "action_mse_all": action_mse_all.detach().item(),
        }

    # This API evaluates complete supervised trials, which must include comparison activity.
    if action_config["type"] != "decision_ce":
        raise ValueError(f"Unsupported action loss: {action_config['type']}")
    comparison_start = int(response_start_ms / dt) + 1
    if not 0 < comparison_start < act_pred.shape[0]:
        raise ValueError("Action supervision requires pre-response and comparison windows")
    pre_logits = act_pred[:comparison_start]
    comparison_logits = act_pred[comparison_start:]

    pre_targets = torch.zeros(
        pre_logits.shape[0],
        device=act_pred.device,
        dtype=torch.long,
    )
    pre_response_loss = F.cross_entropy(pre_logits, pre_targets)
    pre_hold_probs = F.softmax(pre_logits, dim=-1)[:, 0]
    pre_event_penalty = -torch.log(pre_hold_probs.min().clamp(min=torch.finfo(act_pred.dtype).eps))
    pre_hold_accuracy = (pre_logits.argmax(dim=1) == 0).float().mean()

    eps = torch.finfo(act_pred.dtype).eps
    is_match = get_trial_relation(trial_type) == "match"
    match_go_prob = None
    nonmatch_nogo_prob = None

    comparison_probs = F.softmax(comparison_logits, dim=-1)
    release_probs = comparison_probs[:, 1]
    hold_probs = comparison_probs[:, 0]
    decoded_comparison = comparison_logits.argmax(dim=1)
    comparison_release_rate = (decoded_comparison == 1).float().mean()
    comparison_mean_release_prob = release_probs.mean()
    decision_margin = (comparison_logits[:, 1] - comparison_logits[:, 0]).mean()

    if is_match:
        # Do not imitate the random sampled RT. Once comparison starts, the
        # correct policy for match trials is to release at any valid step.
        go_prob = release_probs.max().clamp(min=eps)
        decision_event_penalty = -torch.log(go_prob)
        match_go_prob = go_prob
        comparison_targets = torch.ones(
            comparison_logits.shape[0],
            device=act_pred.device,
            dtype=torch.long,
        )
        comparison_decision_accuracy = (decoded_comparison == 1).any().float()
    else:
        # A no-go trial is correct only if no comparison step releases.
        nogo_prob = hold_probs.min().clamp(min=eps)
        decision_event_penalty = -torch.log(nogo_prob)
        nonmatch_nogo_prob = nogo_prob
        comparison_targets = torch.zeros(
            comparison_logits.shape[0],
            device=act_pred.device,
            dtype=torch.long,
        )
        comparison_decision_accuracy = (decoded_comparison == 0).all().float()

    comparison_ce_loss = F.cross_entropy(
        comparison_logits,
        comparison_targets,
    )
    class_weight = float(action_config["match_weight" if is_match else "nonmatch_weight"])
    event_weight_key = "match_event_weight" if is_match else "nonmatch_event_weight"
    event_weight = float(action_config[event_weight_key])
    weighted_comparison_ce_loss = class_weight * comparison_ce_loss
    decision_event_loss = class_weight * event_weight * decision_event_penalty
    decision_window_loss = weighted_comparison_ce_loss + decision_event_loss
    action_mse_decision = F.mse_loss(
        comparison_probs,
        F.one_hot(comparison_targets, num_classes=2).to(
            device=act_pred.device,
            dtype=act_pred.dtype,
        ),
    )

    pre_response_weight = action_config["pre_response_weight"]
    pre_event_weight = float(action_config["pre_event_weight"])
    pre_window_loss = pre_response_loss + pre_event_weight * pre_event_penalty
    action_loss = (
        float(pre_response_weight) * pre_window_loss
        + float(action_config["decision_weight"]) * float(relation_weight) * decision_window_loss
    )
    metrics = {
        "decision_event_loss": decision_event_loss.detach().item(),
        "decision_event_penalty": decision_event_penalty.detach().item(),
        "decision_window_loss": decision_window_loss.detach().item(),
        "decision_ce_loss": weighted_comparison_ce_loss.detach().item(),
        "comparison_ce_loss": comparison_ce_loss.detach().item(),
        "weighted_comparison_ce_loss": weighted_comparison_ce_loss.detach().item(),
        "wait_loss": pre_response_loss.detach().item(),
        "pre_response_loss": pre_response_loss.detach().item(),
        "pre_event_penalty": pre_event_penalty.detach().item(),
        "pre_event_loss": (pre_event_weight * pre_event_penalty).detach().item(),
        "pre_window_loss": pre_window_loss.detach().item(),
        "pre_response_hold_accuracy": pre_hold_accuracy.detach().item(),
        "fixation_loss": 0.0,
        "action_mse_all": action_mse_all.detach().item(),
        "action_mse_decision": action_mse_decision.detach().item(),
        "decision_accuracy_from_logits": comparison_decision_accuracy.detach().item(),
        "comparison_decision_accuracy_from_logits": (comparison_decision_accuracy.detach().item()),
        "comparison_release_rate_from_logits": (comparison_release_rate.detach().item()),
        "comparison_mean_release_prob": comparison_mean_release_prob.detach().item(),
        "decision_margin_left_minus_right": decision_margin.detach().item(),
        "decision_margin_release_minus_hold": decision_margin.detach().item(),
        "relation_loss_weight": float(relation_weight),
        "decision_class_weight": class_weight,
        "decision_event_weight": float(action_config["window_event_weight"]),
        "active_decision_event_weight": event_weight,
        "comparison_start_index": float(comparison_start),
        "comparison_steps": float(comparison_logits.shape[0]),
        "pre_response_weight": float(pre_response_weight),
        "pre_event_weight": pre_event_weight,
        "comparison_decision_weight": float(action_config["decision_weight"]),
    }
    if match_go_prob is not None:
        metrics["match_go_probability"] = match_go_prob.detach().item()
    if nonmatch_nogo_prob is not None:
        metrics["nonmatch_nogo_probability"] = nonmatch_nogo_prob.detach().item()
    return action_loss, metrics


def get_loss_metrics(
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
    predict_all,
    dt,
    task_name="axby",
    response_start_ms=2500,
    numerosity_config=None,
    log_to_wandb=False,
):
    RT_list = []
    pred_loss = 0
    homeostatic_loss = 0
    trial_corrects = []
    cognitive_metric_values = defaultdict(list)
    action_config = sim_config["action_loss"]
    relation_loss_weights = get_relation_loss_weights(
        trial_types,
        task_name,
        action_config,
    )
    for i in range(real_bs):
        # here we can tell if the prediction is correct.
        stimuli_trial = stimuli[i, 0 : lengths[i]]
        outputs_trial = outputs[i, 0 : lengths[i]]
        actions_trial = actions[i, 0 : lengths[i]]
        hiddens_trial = hiddens[i, 0 : lengths[i]]
        action_dim = actions_trial.shape[1]
        readout_dim = action_dim
        if predict_all > 0:
            assert outputs_trial.shape[1] == readout_dim + stimuli_trial.shape[1]
            readout_pred = outputs_trial[:, :readout_dim]
            stimulus_pred = outputs_trial[:, readout_dim:]
            single_pred_loss = predict_all * F.mse_loss(stimulus_pred[:-1], stimuli_trial[1:])
        else:
            assert outputs_trial.shape[1] == readout_dim
            readout_pred = outputs_trial
            single_pred_loss = torch.zeros((), device=outputs_trial.device)

        act_pred = get_action_predictions(readout_pred, action_dim, sim_config)
        if task_name == NUMEROSITY_TASK and action_dim == 2:
            action_loss, action_metrics = compute_go_nogo_action_supervision_loss(
                act_pred,
                actions_trial,
                trial_types[i],
                response_start_ms,
                dt,
                action_config,
                relation_weight=relation_loss_weights[i],
            )
        else:
            action_loss, action_metrics = compute_action_supervision_loss(
                act_pred,
                actions_trial,
                action_config,
                relation_weight=relation_loss_weights[i],
            )
        single_pred_loss = single_pred_loss + action_loss

        for key, value in action_metrics.items():
            cognitive_metric_values[key].append(value)

        pred_loss = pred_loss + single_pred_loss  # F.mse_loss(outputs_trial, actions_trial)
        if sim_config["l2_h"] > 0:
            homeostatic_loss = homeostatic_loss + torch.pow(hiddens_trial, 2).mean()
            # homeostatic_loss = homeostatic_loss + torch.abs(hiddens_trial).mean()

        curr_trial_type = trial_types[i]

        first_choice_ms = None
        if task_name == NUMEROSITY_TASK and act_pred.shape[1] == 2:
            trial_correct, first_choice_ms, behavior_metrics = compute_go_nogo_trial_behavior(
                act_pred,
                curr_trial_type,
                response_start_ms,
                dt,
            )
            for key, value in behavior_metrics.items():
                cognitive_metric_values[key].append(value)
            RT_list.append(first_choice_ms if trial_correct and first_choice_ms is not None else -1)
        else:
            response_T = int(response_start_ms / dt)
            if (act_pred.data[:, 0] > 0.5).float().mean() > 0.95:
                joysticks_trial = torch.max(act_pred.data[:, 1:], dim=1).indices
                act_v = act_pred.data[:, 1:][
                    torch.arange(len(joysticks_trial)).to(act_pred.device),
                    joysticks_trial,
                ]
                joystick_threshold = (((act_v > 0.5).float() - 0.5) * 2).int()
                joysticks_trial = joysticks_trial * joystick_threshold

                early_joysticks_trial = joysticks_trial[:response_T]
                late_joysticks_trial = joysticks_trial[response_T:]

                # Make sure there are no joystick movements before the task response epoch.
                if (early_joysticks_trial == 0).all().item() is True:
                    choose_L = torch.argwhere(late_joysticks_trial == 1)
                    choose_R = torch.argwhere(late_joysticks_trial == 2)
                    expected_joystick = get_expected_joystick(curr_trial_type, task_name)
                    if expected_joystick == 1:
                        # Target trials choose left.
                        if len(choose_L) > 0:
                            if len(choose_R) == 0:
                                trial_correct = True
                            else:
                                trial_correct = choose_L[0][0].item() < choose_R[0][0].item()
                        else:
                            trial_correct = False
                    else:
                        # Nontarget trials choose right.
                        if len(choose_R) > 0:
                            if len(choose_L) == 0:
                                trial_correct = True
                            else:
                                trial_correct = choose_R[0][0].item() < choose_L[0][0].item()
                        else:
                            trial_correct = False
                else:
                    trial_correct = False
            else:
                trial_correct = False

            if trial_correct is True:
                expected_joystick = get_expected_joystick(curr_trial_type, task_name)
                if expected_joystick == 1:
                    rt = choose_L[0][0].item() * dt
                else:
                    rt = choose_R[0][0].item() * dt
                first_choice_ms = rt
                RT_list.append(rt)
            else:
                RT_list.append(-1)

        trial_corrects.append(trial_correct)
        if trial_correct is True and first_choice_ms is not None:
            rt = float(first_choice_ms)
            RTs[curr_trial_type].append(rt)
            if log_to_wandb:
                wandb.log(
                    {
                        "global_steps": global_steps,
                        "RT/{}".format(curr_trial_type): RTs[curr_trial_type][-1],
                    }
                )
    cognitive_metrics = {
        key: float(np.mean(values))
        for key, values in cognitive_metric_values.items()
        if len(values) > 0
    }
    return trial_corrects, pred_loss, homeostatic_loss, RTs, RT_list, cognitive_metrics
