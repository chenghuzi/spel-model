"""Supervised DPX and numerosity episode collection."""

import copy
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from environments.dpx import A2C, TRIAL_TYPES, AXBYEnv, MatureAgent
from environments.numerosity import (
    NUMEROSITY_TASK,
    NumerosityMatureAgent,
    build_numerosity_env,
    get_numerosity_config,
)


def action_id_to_vector(action_id, dtype=np.float32, action_dim=3):
    vector = np.zeros(int(action_dim), dtype=dtype)
    vector[int(action_id)] = 1.0
    return vector


def encode_supervised_row(obs, act, stimuli_dim, dtype=np.float16, action_dim=3):
    row = np.zeros((stimuli_dim + int(action_dim),), dtype=dtype)
    row[:stimuli_dim] = obs
    row[stimuli_dim + int(act)] = 1.0
    return row


def episodes_post_process_m(episodes, dt, stimuli_dim=12, trial_end_ms=3200):
    fixation_dim = 1
    joystick_dim = 3
    # dim = 12 + 6 + 2 + 2
    ep_reps = []
    ep_reps_statistics = defaultdict(list)
    for ep in episodes:
        ep_len = len(ep)
        # inputs = np.zeros((ep_len, stimuli_dim))
        # actions = np.zeros((ep_len, joystick_dim+fixation_dim))

        ep_rep = np.zeros(
            (ep_len, stimuli_dim + fixation_dim + joystick_dim),
            dtype=np.float32,
        )
        act_prev = 3
        acts = []
        for t in range(ep_len):
            obs, act, reward, done, info = ep[t]
            ep_rep[t, :stimuli_dim] = obs
            ep_rep[t, stimuli_dim] = A2C[act][0]
            if A2C[act][1] == 0:
                ep_rep[t, stimuli_dim + 1 :] = np.array([1, 0, 0])
            elif A2C[act][1] == 1:
                ep_rep[t, stimuli_dim + 1 :] = np.array([0, 1, 0])
            elif A2C[act][1] == 2:
                ep_rep[t, stimuli_dim + 1 :] = np.array([0, 0, 1])

            if act_prev != act:
                act_prev = act
            acts.append(act)

        # t_act = int(100/dt)
        # t_remaining = int(3500/dt - ep_len - t_act)
        # t_remaining = int(3200/dt - ep_len - t_act)
        target_len = max(int(trial_end_ms / dt), ep_len)
        t_act = target_len - ep_len

        ep_rep = np.concatenate(
            [
                ep_rep.copy(),
                np.tile(ep_rep[-1].reshape(1, -1), (t_act, 1)),
                # np.tile(np.zeros_like(ep_rep[-1].reshape(1,-1)), (t_remaining, 1))
            ]
        )

        ep_reps.append((ep_rep.copy(), reward, info["type"]))
        ep_reps_statistics[info["type"]].append(ep_len)

    ep_reps_statistics_mean = {}
    for k, v in ep_reps_statistics.items():
        ep_reps_statistics_mean[k] = {
            "mean": np.mean(v),
            "std": np.std(v),
        }
        print(f"{k}: {ep_reps_statistics_mean[k]}\n")

    return ep_reps


def gen_training_episodes(total_t, trial_type, p_f=0.999, p_j=0.995, dt=250):
    assert trial_type in TRIAL_TYPES
    env = AXBYEnv(trial_type, dt=dt)
    agent = MatureAgent(env, p_f, p_j)

    episodes = []

    obs, info = env.reset()
    act = 3
    reward = 0
    done = False
    time_statistics = defaultdict(list)
    episode = [(obs.copy(), act, reward, done, info)]

    phbar = tqdm(range(total_t))
    for t in phbar:
        act = agent.select_act(obs)
        obs_next, reward, terminated, truncated, info = env.step(act)
        done = truncated or terminated

        episode.append((obs.copy(), act, reward, done, info))
        obs = obs_next
        if done:
            time_statistics[info["type"]].append(len(episode))
            episodes.append(copy.deepcopy(episode))
            phbar.set_postfix({"episodes": len(episodes)})
            obs, info = env.reset()
            act = 3
            reward = 0
            done = False
            episode = [(obs.copy(), act, reward, done, info)]

    episode_reps = episodes_post_process_m(episodes, dt)
    for k, v in time_statistics.items():
        print(f"Length of {k}: mean={np.mean(v)}, std={np.std(v)}")
    return episode_reps


def gen_numerosity_training_episodes(
    total_t,
    p_f=0.999,
    p_j=0.995,
    dt=25,
    numerosity_config=None,
):
    config = get_numerosity_config(numerosity_config)
    env = build_numerosity_env(dt, config)
    rt_scale_steps = config["rt_exponential_scale_steps"]
    if rt_scale_steps is None and config["rt_exponential_scale_ms"] is not None:
        rt_scale_steps = float(config["rt_exponential_scale_ms"]) / float(dt)
    agent = NumerosityMatureAgent(
        env,
        p_f,
        p_j,
        rt_exponential_scale_steps=rt_scale_steps,
    )

    episode_reps = []
    obs, info = env.reset()
    agent.reset()
    act = 0
    reward = 0
    done = False
    time_statistics = defaultdict(list)
    action_dim = env.action_space.n
    episode_rows = []

    # Store each row as observation-at-current-env-time plus the action chosen
    # at that same time. This keeps row index k aligned to t = k * dt.
    phbar = tqdm(range(total_t))
    for t in phbar:
        act = agent.select_act(obs)
        episode_rows.append(
            encode_supervised_row(obs, act, env.stimulus_dim, action_dim=action_dim)
        )
        obs_next, reward, terminated, truncated, info = env.step(act)
        done = truncated or terminated
        obs = obs_next
        if done:
            ep_rep = np.stack(episode_rows).astype(np.float16, copy=False)

            time_statistics[info["type"]].append(len(episode_rows))
            episode_reps.append((ep_rep.copy(), reward, info["type"]))
            phbar.set_postfix({"episodes": len(episode_reps)})
            obs, info = env.reset()
            act = 0
            reward = 0
            done = False
            agent.reset()
            episode_rows = []

    ep_reps_statistics_mean = {}
    for k, v in time_statistics.items():
        ep_reps_statistics_mean[k] = {
            "mean": np.mean(v),
            "std": np.std(v),
        }
        print(f"{k}: {ep_reps_statistics_mean[k]}\n")
    for k, v in time_statistics.items():
        print(f"Length of {k}: mean={np.mean(v)}, std={np.std(v)}")
    return episode_reps


def gen_numerosity_training_samples(
    total_t_mature,
    dt=25,
    p_f=1.0,
    p_j=0.9999,
    numerosity_config=None,
):
    training_episode_reps = gen_numerosity_training_episodes(
        total_t_mature,
        p_f=p_f,
        p_j=p_j,
        dt=dt,
        numerosity_config=numerosity_config,
    )

    # Keep the same rewarded-episode filtering contract as the AXBY generator.
    rewarded = np.array([int(ep_rep[1]) for ep_rep in training_episode_reps])
    avg_lengths = np.array([len(ep_rep[0]) for ep_rep in training_episode_reps])
    print(f"training lengths: mean={np.mean(avg_lengths)}, std={np.std(avg_lengths)}")
    rewarded_cr = np.mean(rewarded)
    print(f"reward rate {rewarded_cr}")

    indices_rewarded = np.argwhere(rewarded == 1)[:, 0]
    episode_reps_rewarded = [
        (training_episode_reps[idx][0], training_episode_reps[idx][2]) for idx in indices_rewarded
    ]

    return episode_reps_rewarded, rewarded_cr


def gen_training_samples(
    trial_type,
    total_t_mature,
    dt=250,
    p_f=1.0,
    p_j=0.9999,
    task_name="axby",
    numerosity_config=None,
):
    if task_name == NUMEROSITY_TASK:
        return gen_numerosity_training_samples(
            total_t_mature,
            dt=dt,
            p_f=p_f,
            p_j=p_j,
            numerosity_config=numerosity_config,
        )

    training_episode_reps = gen_training_episodes(
        total_t_mature, trial_type, p_f=p_f, p_j=p_j, dt=dt
    )
    # rep = training_episode_reps[0][0]

    rewarded = np.array([int(ep_rep[1]) for ep_rep in training_episode_reps])
    avg_lengths = np.array([len(ep_rep[0]) for ep_rep in training_episode_reps])
    print(f"training lengths: mean={np.mean(avg_lengths)}, std={np.std(avg_lengths)}")
    rewarded_cr = np.mean(rewarded)
    print(f"reward rate {rewarded_cr}")

    indices_rewarded = np.argwhere(rewarded == 1)[:, 0]
    episode_reps_rewarded = [
        (training_episode_reps[idx][0], training_episode_reps[idx][2]) for idx in indices_rewarded
    ]

    return episode_reps_rewarded, rewarded_cr
