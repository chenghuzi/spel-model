"""Trial-label interpretation shared by batching and metrics."""

from environments.numerosity import NUMEROSITY_TASK


def get_expected_joystick(trial_type, task_name):
    if task_name == NUMEROSITY_TASK:
        relation = get_trial_relation(trial_type)
        return 1 if relation == "match" else 0

    return 1 if trial_type == "AX" else 2


def get_trial_relation(trial_type):
    return trial_type.split("_")[-1]


def get_balanced_relation_pair(task_name):
    if task_name == NUMEROSITY_TASK:
        return "match", "nonmatch"
    return None


def parse_numerosity_trial_type(trial_type):
    # Expected format: sample_3_test_5_nonmatch.
    parts = str(trial_type).split("_")
    if len(parts) != 5 or parts[0] != "sample" or parts[2] != "test":
        raise ValueError(f"invalid numerosity trial type: {trial_type}")
    return int(parts[1]), int(parts[3]), parts[4]
