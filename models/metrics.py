"""Behavioral accuracy, reaction times, and prediction-error summaries."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from environments.numerosity import NUMEROSITY_TASK
from environments.trials import get_trial_relation
from models.leaky import LeakyRNN


def get_prediction_error_metrics(model, prediction_errors, mask):
    # Only the visual network has phase-separated latent prediction errors.
    if isinstance(model, LeakyRNN):
        return {}

    errors = prediction_errors.detach()
    valid = mask.to(device=errors.device, dtype=torch.bool)
    phase_context = model.make_phase_context(
        errors.shape[0],
        errors.shape[1],
        errors.device,
        errors.dtype,
    )
    phase_ids = phase_context.argmax(dim=-1)
    latent_slice, context_slice, action_slice = model.input_slices()

    def masked_mean(values, selected):
        selected_values = values[selected]
        if selected_values.numel() == 0:
            return None
        return float(selected_values.mean().cpu().item())

    metrics = {}
    components = {
        "visual": errors[..., latent_slice],
        "phase": errors[..., context_slice],
        "action": errors[..., action_slice],
    }
    for name, component in components.items():
        component_mae = component.abs().mean(dim=-1)
        component_mse = component.square().mean(dim=-1)
        metrics[f"prediction_error_{name}_mae"] = masked_mean(
            component_mae,
            valid,
        )
        metrics[f"prediction_error_{name}_mse"] = masked_mean(
            component_mse,
            valid,
        )

    visual_mae = components["visual"].abs().mean(dim=-1)
    for phase_id, phase_name in enumerate(("fixation", "sample", "delay", "test")):
        value = masked_mean(visual_mae, valid & (phase_ids == phase_id))
        if value is not None:
            metrics[f"prediction_error_visual_mae_{phase_name}"] = value

    previous_phase = torch.cat([phase_ids[:, :1] - 1, phase_ids[:, :-1]], dim=1)
    visible = (phase_ids == 1) | (phase_ids == 3)
    onset = valid & visible & (phase_ids != previous_phase)
    steady = valid & visible & (phase_ids == previous_phase)
    onset_value = masked_mean(visual_mae, onset)
    steady_value = masked_mean(visual_mae, steady)
    if onset_value is not None:
        metrics["prediction_error_visual_mae_onset"] = onset_value
    if steady_value is not None:
        metrics["prediction_error_visual_mae_steady"] = steady_value
    return metrics


def update_relation_corrects(relation_corrects, trial_types, trial_corrects, task_name):
    if task_name not in {NUMEROSITY_TASK}:
        return {}

    for trial_type, trial_correct in zip(trial_types, trial_corrects):
        relation_corrects[get_trial_relation(trial_type)].append(float(trial_correct))
    return get_relation_correct_metrics(trial_types, trial_corrects, task_name)


def get_relation_correct_metrics(trial_types, trial_corrects, task_name):
    if task_name not in {NUMEROSITY_TASK}:
        return {}

    relation_corrects = defaultdict(list)
    for trial_type, trial_correct in zip(trial_types, trial_corrects):
        relation_corrects[get_trial_relation(trial_type)].append(float(trial_correct))
    return summarize_relation_corrects(relation_corrects)


def summarize_relation_corrects(relation_corrects):
    metrics = {}
    for relation in sorted(relation_corrects):
        values = relation_corrects[relation]
        if len(values) > 0:
            metrics[f"correct_rate_{relation}"] = float(np.mean(values))
    relation_rates = [value for key, value in metrics.items() if key.startswith("correct_rate_")]
    if len(relation_rates) >= 2:
        metrics["balanced_correct_rate"] = float(np.mean(relation_rates))
    return metrics


def has_required_rt_coverage(RTs, task_name):
    # AXBY keeps the original four-condition gate; numerosity has many pair labels.
    if task_name in {NUMEROSITY_TASK}:
        return len(RTs) > 0
    return len(RTs) == 4


def print_rt_summary(RTs, task_name, prefix):
    # Numerosity prints aggregate match/nonmatch RTs to avoid flooding logs with pair labels.
    if task_name in {NUMEROSITY_TASK}:
        grouped_RTs = defaultdict(list)
        for trial_type, values in RTs.items():
            grouped_RTs[trial_type.split("_")[-1]].extend(values)
        for relation in sorted(grouped_RTs.keys()):
            values = grouped_RTs[relation]
            print(
                f"{prefix} RT/{relation}(n={len(values)}): "
                f"{np.mean(values):.3f}, {np.std(values):.3f}"
            )
        return

    ks = sorted(list(RTs.keys()))
    for k in ks:
        v = RTs[k]
        print(f"{prefix} RT/{k}(n={len(v)}): {np.mean(v):.3f}, {np.std(v):.3f}")


def model_action_slice(config, outputs_dim, action_dim):
    return slice(0, action_dim)


def get_action_predictions(outputs_trial, action_dim, config):
    return outputs_trial[
        ...,
        model_action_slice(config, outputs_trial.shape[-1], action_dim),
    ]


def decode_action_logits(action_pred):
    return F.softmax(action_pred, dim=-1).argmax(dim=-1)


def compute_go_nogo_trial_behavior(
    act_pred,
    trial_type,
    response_start_ms,
    dt,
):
    decoded = decode_action_logits(act_pred.detach())
    # A complete supervised trial must reach comparison before behavior is scored.
    response_T = int(response_start_ms / dt)
    if not 0 <= response_T < decoded.shape[0] - 1:
        raise ValueError("Behavior scoring requires a comparison window")
    early_actions = decoded[: response_T + 1]
    comparison_actions = decoded[response_T + 1 :]
    premature = bool((early_actions != 0).any().item())
    is_match = get_trial_relation(trial_type) == "match"
    first_release_ms = None

    release_indices = torch.argwhere(comparison_actions == 1).flatten()
    released = release_indices.numel() > 0
    if released:
        first_release_index = int(release_indices[0].item())
        first_release_ms = (first_release_index + 1) * dt

    no_response = (not premature) and is_match and not released
    false_alarm = (not premature) and (not is_match) and released
    trial_correct = (not premature) and (
        (is_match and released) or ((not is_match) and not released)
    )

    metrics = {
        "teacher_forced_trial_correct_rate": float(trial_correct),
        "teacher_forced_premature_choice_rate": float(premature),
        "teacher_forced_no_response_rate": float(no_response),
        "teacher_forced_false_alarm_rate": float(false_alarm),
    }
    if first_release_ms is not None:
        metrics["teacher_forced_first_choice_ms"] = float(first_release_ms)
    return trial_correct, first_release_ms, metrics
