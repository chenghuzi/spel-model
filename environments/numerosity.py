"""Numerosity delayed-match task, behavior policy, and settings."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.stimuli import DotArrayGenerator

NUMEROSITY_TASK = "numerosity"


class NumerosityEnv(gym.Env):
    def __init__(
        self,
        dt=25,
        grid_size=15,
        min_dot_distance=3.0,
        min_numerosity=1,
        max_numerosity=5,
        match_probability=0.5,
        stimulus_mode="standard",
        stimulus_family="nieder",
        position_lattice_size=5,
        aperture_radius=None,
        aperture_value=0.65,
        dot_value=0.05,
        dot_radius_values=None,
        equal_area_total_area=None,
        density_candidate_samples=256,
        nonmatch_policy="any",
        fixation_ms=500,
        sample_ms=500,
        delay_ms=1000,
        test_ms=500,
        response_ms=None,
    ):
        # Task timing is a public input; reject invalid durations rather than repairing the RT schedule.
        if dt <= 0 or test_ms <= 0:
            raise ValueError("dt and test_ms must be positive")
        for duration in (fixation_ms, sample_ms, delay_ms, test_ms):
            if duration < 0 or duration % dt != 0:
                raise ValueError("Task durations must be nonnegative multiples of dt")
        if response_ms is not None:
            assert response_ms % dt == 0
        self.dt = dt
        self.grid_size = grid_size
        self.min_dot_distance = min_dot_distance
        self.min_numerosity = min_numerosity
        self.max_numerosity = max_numerosity
        self.match_probability = match_probability
        self.stimulus_mode = stimulus_mode
        self.stimulus_family = stimulus_family
        self.nonmatch_policy = nonmatch_policy
        self.fixation_ms = fixation_ms
        self.sample_ms = sample_ms
        self.delay_ms = delay_ms
        self.test_ms = test_ms
        self.response_ms = 0 if response_ms is None else response_ms
        self.sample_start_ms = fixation_ms
        self.sample_end_ms = self.sample_start_ms + sample_ms
        self.test_start_ms = self.sample_end_ms + delay_ms
        self.test_end_ms = self.test_start_ms + test_ms
        self.response_start_ms = self.test_start_ms
        self.trial_end_ms = self.test_end_ms

        self.action_space = spaces.Discrete(2)
        self._action2command = {0: 0, 1: 1}
        self._command2action = {v: k for k, v in self._action2command.items()}

        # Build visual stimulus state once per trial while exposing only flattened images.
        self.numerosities = np.arange(min_numerosity, max_numerosity + 1)
        self.stimulus_dim = grid_size * grid_size
        self.empty_obs = np.zeros(self.stimulus_dim)
        self.dot_generator = DotArrayGenerator(
            grid_size=grid_size,
            stimulus_mode=stimulus_mode,
            min_dot_distance=min_dot_distance,
            stimulus_family=stimulus_family,
            position_lattice_size=position_lattice_size,
            aperture_radius=aperture_radius,
            aperture_value=aperture_value,
            dot_value=dot_value,
            dot_radius_values=dot_radius_values,
            equal_area_total_area=equal_area_total_area,
            density_candidate_samples=density_candidate_samples,
        )

    def reset(self):
        # Sample the hidden numerosity labels, then generate independent visual arrays.
        self.sample_numerosity = int(np.random.choice(self.numerosities))
        if np.random.random() < self.match_probability:
            self.test_numerosity = self.sample_numerosity
        else:
            candidates = self._nonmatch_candidates()
            self.test_numerosity = int(np.random.choice(candidates))

        self.linear_angle = None
        if self.stimulus_family == "nieder" and self.stimulus_mode == "linear":
            self.linear_angle = float(np.random.uniform(0.0, np.pi))
        self.sample_stimulus = self.dot_generator.render(
            self.sample_numerosity,
            linear_angle=self.linear_angle,
        )
        self.test_stimulus = self.dot_generator.render(
            self.test_numerosity,
            linear_angle=self.linear_angle,
        )
        self.is_match = self.sample_numerosity == self.test_numerosity
        self.trial_type = self._trial_type()
        self.t = 0

        observation = self.empty_obs
        info = {
            "type": self.trial_type,
            "sample_numerosity": self.sample_numerosity,
            "test_numerosity": self.test_numerosity,
            "is_match": self.is_match,
            "task": NUMEROSITY_TASK,
            "response_start_ms": self.response_start_ms,
            "linear_angle": self.linear_angle,
        }
        return observation, info

    def _nonmatch_candidates(self):
        if self.nonmatch_policy in {"adjacent", "nieder_adjacent"}:
            candidates = []
            if self.sample_numerosity > self.min_numerosity:
                candidates.append(self.sample_numerosity - 1)
            if self.sample_numerosity < self.max_numerosity:
                candidates.append(self.sample_numerosity + 1)
            return np.array(candidates, dtype=int)
        return self.numerosities[self.numerosities != self.sample_numerosity]

    def step(self, action):
        action_time = self.t
        self.t += self.dt
        joystick = self._action2command[int(action)]

        # Present the comparison stimulus throughout the response epoch.
        if self.sample_start_ms < self.t <= self.sample_end_ms:
            observation = self.sample_stimulus
        elif self.test_start_ms < self.t <= self.test_end_ms:
            observation = self.test_stimulus
        else:
            observation = self.empty_obs

        info = {
            "type": self.trial_type,
            "sample_numerosity": self.sample_numerosity,
            "test_numerosity": self.test_numerosity,
            "is_match": self.is_match,
            "task": NUMEROSITY_TASK,
            "response_start_ms": self.response_start_ms,
            "linear_angle": self.linear_angle,
        }

        # Any release before the comparison epoch is a premature response.
        if action_time <= self.response_start_ms:
            if joystick != 0:
                terminated = True
                reward = 0
            else:
                terminated = False
                reward = 0
            return observation, reward, terminated, False, info

        # During comparison, match is go/release and nonmatch is no-go/hold.
        if action_time <= self.trial_end_ms:
            if self.is_match and joystick == 1:
                terminated = True
                reward = 1
            elif (not self.is_match) and joystick == 0 and action_time >= self.trial_end_ms:
                terminated = True
                reward = 1
            elif joystick == 0:
                terminated = False
                reward = 0
            else:
                terminated = True
                reward = 0
            return observation, reward, terminated, False, info

        terminated = True
        reward = 0
        return observation, reward, terminated, False, info

    def _trial_type(self):
        # Store complete numerosity pair information for later distance and ratio analyses.
        relation = "match" if self.is_match else "nonmatch"
        return f"sample_{self.sample_numerosity}_test_{self.test_numerosity}_{relation}"


class NumerosityMatureAgent:
    def __init__(self, env, p_f, p_j, rt_exponential_scale_steps=None):
        self._env = env
        self.p_f = p_f
        self.p_j = p_j
        self.rt_exponential_scale_steps = rt_exponential_scale_steps
        self.reset()

    def _ensure_trial_schedule(self):
        trial_key = self._env.trial_type
        if trial_key == self.trial_key:
            return

        self.trial_key = trial_key
        self.release_time_ms = None
        if self._env.is_match:
            comparison_steps = int(self._env.test_ms / self._env.dt)
            scale_steps = self.rt_exponential_scale_steps
            if scale_steps is None:
                scale_steps = max(1.0, comparison_steps / 3.0)
            rt_steps = int(np.floor(np.random.exponential(float(scale_steps)))) + 1
            rt_steps = min(rt_steps, comparison_steps)
            self.release_time_ms = self._env.response_start_ms + rt_steps * self._env.dt

    def select_act(self, obs):
        self._ensure_trial_schedule()
        joystick = 0
        if self._env.is_match and self._env.t >= self.release_time_ms:
            joystick = 1

        self.internal_t += 1
        return self._env._command2action[joystick]

    def reset(self):
        self.internal_t = 0
        self.trial_key = None
        self.release_time_ms = None


def get_numerosity_config(numerosity_config=None):
    # Centralize defaults so task generation, metadata, and configs stay aligned.
    config = {
        "grid_size": 15,
        "min_dot_distance": 3.0,
        "min_numerosity": 1,
        "max_numerosity": 5,
        "match_probability": 0.5,
        "stimulus_mode": "standard",
        "stimulus_family": "nieder",
        "position_lattice_size": 5,
        "aperture_radius": None,
        "aperture_value": 0.65,
        "dot_value": 0.05,
        "dot_radius_values": None,
        "equal_area_total_area": None,
        "density_candidate_samples": 256,
        "nonmatch_policy": "any",
        "fixation_ms": 500,
        "sample_ms": 500,
        "delay_ms": 1000,
        "test_ms": 500,
        "response_ms": 0,
        "response_delay_ms": 0,
        "rt_exponential_scale_steps": None,
        "rt_exponential_scale_ms": None,
    }
    if numerosity_config is not None:
        config.update(numerosity_config)
    return config


def build_numerosity_env(dt, numerosity_config):
    # Callers pass resolved task settings; this factory only maps them to environment arguments.
    config = numerosity_config
    return NumerosityEnv(
        dt=dt,
        grid_size=config["grid_size"],
        min_dot_distance=config["min_dot_distance"],
        min_numerosity=config["min_numerosity"],
        max_numerosity=config["max_numerosity"],
        match_probability=config["match_probability"],
        stimulus_mode=config["stimulus_mode"],
        stimulus_family=config["stimulus_family"],
        position_lattice_size=config["position_lattice_size"],
        aperture_radius=config["aperture_radius"],
        aperture_value=config["aperture_value"],
        dot_value=config["dot_value"],
        dot_radius_values=config["dot_radius_values"],
        equal_area_total_area=config["equal_area_total_area"],
        density_candidate_samples=config["density_candidate_samples"],
        nonmatch_policy=config["nonmatch_policy"],
        fixation_ms=config["fixation_ms"],
        sample_ms=config["sample_ms"],
        delay_ms=config["delay_ms"],
        test_ms=config["test_ms"],
        response_ms=config["response_ms"],
    )
