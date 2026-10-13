"""Task dimensions and event timing."""

from environments.numerosity import NUMEROSITY_TASK


def get_task_metadata(
    task_name="axby",
    dt=250,
    numerosity_config=None,
):
    # Expose only the dimensions and timing needed by the training loop.
    if task_name == NUMEROSITY_TASK:
        config = numerosity_config
        response_start_ms = config["fixation_ms"] + config["sample_ms"] + config["delay_ms"]
        trial_end_ms = response_start_ms + config["test_ms"]
        return {
            "task_name": NUMEROSITY_TASK,
            "n_in": config["grid_size"] * config["grid_size"],
            "n_o": 2,
            "response_start_ms": response_start_ms,
            "trial_end_ms": trial_end_ms,
        }

    assert task_name == "axby"
    return {
        "task_name": "axby",
        "n_in": 12,
        "n_o": 4,
        "response_start_ms": 2500,
        "trial_end_ms": 3200,
    }
