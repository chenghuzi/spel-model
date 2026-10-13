"""Population tuning summaries, independent of plotting."""

import numpy as np

from analysis.numerosity import analyze_units


def selected_units(unit_stats, alpha, threshold_key):
    p_key = "p_fdr" if threshold_key == "fdr" else "p_unc"
    selected = np.where(unit_stats[p_key] < alpha)[0]
    return np.asarray(selected, dtype=int)


def normalize_response(response):
    response = np.asarray(response, dtype=float)
    low = float(np.min(response))
    high = float(np.max(response))
    return 100.0 * (response - low) / (high - low)


def phase_population(features, labels, numerosities, alpha, threshold_key):
    unit_stats = analyze_units(features, labels, alpha)
    units = selected_units(unit_stats, alpha, threshold_key)
    # A population figure requires significant units; never substitute a different selection.
    if len(units) == 0:
        raise ValueError(f"No units pass the {threshold_key} threshold at alpha={alpha}")
    responses = unit_stats["responses"]
    normalized = np.stack([normalize_response(responses[:, unit]) for unit in units])
    preferred_idx = np.argmax(responses[:, units], axis=0)
    preferred_numbers = numerosities[preferred_idx]

    distance_values = np.arange(
        int(numerosities.min() - numerosities.max()),
        int(numerosities.max() - numerosities.min()) + 1,
    )
    by_distance = {int(distance): [] for distance in distance_values}
    for unit_i, pref in enumerate(preferred_numbers):
        for num_i, numerosity in enumerate(numerosities):
            distance = int(numerosity - pref)
            by_distance[distance].append(float(normalized[unit_i, num_i]))

    distance_mean = np.array(
        [
            np.mean(by_distance[int(distance)]) if by_distance[int(distance)] else np.nan
            for distance in distance_values
        ]
    )
    distance_sem = np.array(
        [
            np.std(by_distance[int(distance)], ddof=1) / np.sqrt(len(by_distance[int(distance)]))
            if len(by_distance[int(distance)]) > 1
            else np.nan
            for distance in distance_values
        ]
    )

    tuning_mean = []
    tuning_sem = []
    for pref in numerosities:
        mask = preferred_numbers == pref
        if mask.any():
            curves = normalized[mask]
            tuning_mean.append(curves.mean(axis=0))
            tuning_sem.append(
                curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
                if curves.shape[0] > 1
                else np.full(curves.shape[1], np.nan)
            )
        else:
            tuning_mean.append(np.full_like(numerosities, np.nan, dtype=float))
            tuning_sem.append(np.full_like(numerosities, np.nan, dtype=float))

    return {
        "unit_stats": unit_stats,
        "units": units,
        "responses": responses,
        "normalized": normalized,
        "preferred_idx": preferred_idx,
        "preferred_numbers": preferred_numbers,
        "distance_values": distance_values,
        "by_distance": by_distance,
        "distance_mean": distance_mean,
        "distance_sem": distance_sem,
        "tuning_mean": np.stack(tuning_mean),
        "tuning_sem": np.stack(tuning_sem),
    }


def jsonable_phase(phase):
    return {
        "n_units": int(len(phase["units"])),
        "units": [int(unit) for unit in phase["units"]],
        "preferred_numbers": [int(value) for value in phase["preferred_numbers"]],
        "distance_values": [int(value) for value in phase["distance_values"]],
        "distance_mean": phase["distance_mean"].tolist(),
        "distance_sem": phase["distance_sem"].tolist(),
        "tuning_mean": phase["tuning_mean"].tolist(),
    }
