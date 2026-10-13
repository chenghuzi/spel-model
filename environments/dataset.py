"""Episode datasets, cache identities, and balanced batching."""

import json as pyjson
import random
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from environments.episodes import encode_supervised_row, gen_training_samples
from environments.numerosity import (
    NUMEROSITY_TASK,
    NumerosityMatureAgent,
    build_numerosity_env,
)
from environments.random import restore_rng_snapshot, rng_snapshot
from environments.trials import get_balanced_relation_pair, get_trial_relation
from paths import cache_file

NUMEROSITY_EPISODE_CACHE_VERSION = "epv2"


def make_sequence_mask(lengths, max_len, device):
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    steps = torch.arange(max_len, device=device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


class HDF5EpisodeDataset:
    def __init__(self, path):
        self.path = Path(path)
        self._h5 = None
        with h5py.File(self.path, "r") as handle:
            self.num_episodes = int(handle["lengths"].shape[0])
            self.max_steps = int(handle.attrs["max_steps"])
            self.dim = int(handle.attrs["dim"])
            trial_types = handle["trial_types"][:]
        self.trial_types = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in trial_types
        ]

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        length = int(self.h5["lengths"][index])
        episode = self.h5["episodes"][index, :length, :]
        return np.asarray(episode, dtype=np.float16), self.trial_types[index]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class InMemoryEpisodeDataset:
    def __init__(self, episodes, lengths, trial_types):
        self.episodes = np.asarray(episodes, dtype=np.float16)
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.trial_types = [str(trial_type) for trial_type in trial_types]
        self.num_episodes = int(self.lengths.shape[0])
        self.max_steps = int(self.episodes.shape[1])
        self.dim = int(self.episodes.shape[2])

    @classmethod
    def from_hdf5(cls, path):
        path = Path(path)
        print(f"load HDF5 cache into memory {path}")
        with h5py.File(path, "r") as handle:
            episodes = handle["episodes"][:]
            lengths = handle["lengths"][:]
            trial_types = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["trial_types"][:]
            ]
        size_gib = episodes.nbytes / (1024**3)
        print(
            f"loaded {episodes.shape[0]} episodes into RAM ({size_gib:.2f} GiB, {episodes.dtype})"
        )
        return cls(episodes, lengths, trial_types)

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        length = int(self.lengths[index])
        return self.episodes[index, :length, :], self.trial_types[index]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class HDF5EpisodeWriter:
    def __init__(self, path, max_steps, dim):
        self.path = Path(path)
        self.max_steps = int(max_steps)
        self.dim = int(dim)
        self.num_total = 0
        self.num_rewarded = 0
        self.time_statistics = defaultdict(list)
        self.handle = h5py.File(self.path, "w")
        self.handle.attrs["complete"] = False
        self.handle.attrs["cache_version"] = NUMEROSITY_EPISODE_CACHE_VERSION
        self.handle.attrs["max_steps"] = self.max_steps
        self.handle.attrs["dim"] = self.dim
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self.episodes = self.handle.create_dataset(
            "episodes",
            shape=(0, self.max_steps, self.dim),
            maxshape=(None, self.max_steps, self.dim),
            chunks=(1, self.max_steps, self.dim),
            dtype=np.float16,
            compression="lzf",
        )
        self.lengths = self.handle.create_dataset(
            "lengths",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1024,),
        )
        self.trial_types = self.handle.create_dataset(
            "trial_types",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=(1024,),
        )

    def append(self, episode, reward, trial_type):
        episode = np.asarray(episode, dtype=np.float16)
        length = int(episode.shape[0])
        if length > self.max_steps:
            raise ValueError(f"episode length {length} exceeds HDF5 max_steps {self.max_steps}")
        if int(episode.shape[1]) != self.dim:
            raise ValueError(f"episode dim {episode.shape[1]} != expected {self.dim}")

        self.num_total += 1
        trial_type = str(trial_type)
        self.time_statistics[trial_type].append(length)
        if int(reward) != 1:
            return

        index = self.num_rewarded
        self.num_rewarded += 1
        self.episodes.resize((self.num_rewarded, self.max_steps, self.dim))
        self.lengths.resize((self.num_rewarded,))
        self.trial_types.resize((self.num_rewarded,))
        padded = np.zeros((self.max_steps, self.dim), dtype=np.float16)
        padded[:length, :] = episode
        self.episodes[index, :, :] = padded
        self.lengths[index] = length
        self.trial_types[index] = trial_type

    def finish(self):
        reward_rate = float(self.num_rewarded) / float(self.num_total)
        stats = {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": int(len(values)),
            }
            for key, values in self.time_statistics.items()
        }
        self.handle.attrs["num_total"] = int(self.num_total)
        self.handle.attrs["num_rewarded"] = int(self.num_rewarded)
        self.handle.attrs["reward_rate"] = float(reward_rate)
        self.handle.attrs["time_statistics_json"] = pyjson.dumps(stats)
        self.handle.attrs["complete"] = True
        self.handle.flush()
        self.handle.close()
        return stats, reward_rate


def numerosity_max_steps(numerosity_config, dt):
    config = numerosity_config
    total_ms = (
        float(config["fixation_ms"])
        + float(config["sample_ms"])
        + float(config["delay_ms"])
        + float(config["test_ms"])
        + float(config["response_ms"])
        + float(config["response_delay_ms"])
    )
    return int(np.ceil(total_ms / float(dt))) + 1


def hdf5_episode_cache_is_complete(path):
    # Only an absent or explicitly incomplete file may be regenerated; corrupt HDF5 must fail.
    path = Path(path)
    if not path.exists():
        return False
    with h5py.File(path, "r") as handle:
        if not handle.attrs["complete"]:
            return False
        for key in ("episodes", "lengths", "trial_types"):
            handle[key]
        return True


def generate_numerosity_hdf5_cache(cache_f, total_t, dt, numerosity_config):
    cache_f = Path(cache_f)
    tmp_f = cache_f.with_name(cache_f.name + ".tmp")
    if tmp_f.exists():
        tmp_f.unlink()

    env = build_numerosity_env(dt, numerosity_config)
    config = numerosity_config
    rt_scale_steps = config["rt_exponential_scale_steps"]
    if rt_scale_steps is None and config["rt_exponential_scale_ms"] is not None:
        rt_scale_steps = float(config["rt_exponential_scale_ms"]) / float(dt)
    agent = NumerosityMatureAgent(
        env,
        p_f=1.0,
        p_j=1.0,
        rt_exponential_scale_steps=rt_scale_steps,
    )
    writer = HDF5EpisodeWriter(
        tmp_f,
        max_steps=numerosity_max_steps(config, dt),
        dim=env.stimulus_dim + env.action_space.n,
    )

    obs, info = env.reset()
    agent.reset()
    episode_rows = []
    phbar = tqdm(range(total_t))
    try:
        for _ in phbar:
            act = agent.select_act(obs)
            episode_rows.append(
                encode_supervised_row(
                    obs,
                    act,
                    env.stimulus_dim,
                    dtype=np.float16,
                    action_dim=env.action_space.n,
                )
            )
            obs, reward, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                episode = np.stack(episode_rows).astype(np.float16, copy=False)
                writer.append(episode, reward, info["type"])
                phbar.set_postfix({"episodes": writer.num_total})
                obs, info = env.reset()
                agent.reset()
                episode_rows = []
        stats, reward_rate = writer.finish()
    except Exception:
        writer.handle.close()
        if tmp_f.exists():
            tmp_f.unlink()
        raise

    for key, value in stats.items():
        print(f"{key}: {{'mean': {value['mean']}, 'std': {value['std']}}}\n")
    for key, value in stats.items():
        print(f"Length of {key}: mean={value['mean']}, std={value['std']}")
    lengths = [length for values in writer.time_statistics.values() for length in values]
    print(f"training lengths: mean={np.mean(lengths)}, std={np.std(lengths)}")
    print(f"reward rate {reward_rate}")
    tmp_f.replace(cache_f)
    return HDF5EpisodeDataset(cache_f)


def read_numerosity_hdf5_data(exp_config, trial_type, total_t, dt, run_name):
    cache_f = episode_cache(exp_config, trial_type, total_t, dt, "episodes.h5")
    if hdf5_episode_cache_is_complete(cache_f):
        print(f"load from HDF5 cache {cache_f}")
        if bool(exp_config["load_training_cache_to_memory"]):
            return InMemoryEpisodeDataset.from_hdf5(cache_f)
        return HDF5EpisodeDataset(cache_f)
    if cache_f.exists():
        print(f"replace incomplete HDF5 cache {cache_f}")
        cache_f.unlink()
    print(f"write HDF5 cache {cache_f}")
    dataset = generate_numerosity_hdf5_cache(
        cache_f,
        total_t=total_t,
        dt=dt,
        numerosity_config=exp_config["numerosity_config"],
    )
    if bool(exp_config["load_training_cache_to_memory"]):
        dataset.close()
        return InMemoryEpisodeDataset.from_hdf5(cache_f)
    return dataset


def balanced_eval_subset(ep_r, max_trials_per_type):
    if max_trials_per_type <= 0:
        return ep_r

    trial_counts = Counter([str(ep[1]) for ep in ep_r])
    target_count = min(min(trial_counts.values()), max_trials_per_type)
    subset = []
    new_trial_counts = Counter()
    for episode, trial_type in ep_r:
        trial_type = str(trial_type)
        if new_trial_counts[trial_type] >= target_count:
            continue
        subset.append((episode, trial_type))
        new_trial_counts[trial_type] += 1
    return subset


def get_control_eval_samples(
    exp_config,
    trial_type,
    total_t,
    dt,
    mode,
    seed_value,
    max_trials_per_type,
):
    numerosity_config = dict(exp_config["numerosity_config"])
    numerosity_config["stimulus_mode"] = mode
    cache_f = cache_file(
        {
            "kind": "control",
            "version": NUMEROSITY_EPISODE_CACHE_VERSION,
            "task": NUMEROSITY_TASK,
            "numerosity": numerosity_config,
            "total_t": total_t,
            "dt": dt,
            "seed": seed_value,
            "cap": max_trials_per_type,
        },
        "episodes.pt",
    )
    if cache_f.is_file():
        res = torch.load(cache_f, weights_only=False)
        return res["ep_r"]

    state = rng_snapshot()
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        ep_r, cr = gen_training_samples(
            trial_type,
            total_t,
            dt=dt,
            p_j=1.0,
            task_name=NUMEROSITY_TASK,
            numerosity_config=numerosity_config,
        )
    finally:
        restore_rng_snapshot(state)

    ep_r = balanced_eval_subset(ep_r, max_trials_per_type)
    torch.save({"ep_r": ep_r, "cr": cr}, cache_f)
    return ep_r


def read_data(exp_config, trial_type, total_t, dt):
    task_name = exp_config["task"]
    run_name = trial_type
    if task_name == NUMEROSITY_TASK and str(exp_config["training_cache_format"]).lower() == "hdf5":
        return read_numerosity_hdf5_data(exp_config, trial_type, total_t, dt, run_name), run_name

    cache_f = episode_cache(exp_config, trial_type, total_t, dt, "episodes.pt")
    cache_training_data = bool(exp_config["cache_training_data"])
    if cache_training_data and cache_f.is_file():
        print(f"load from cache {cache_f}")
        res = torch.load(cache_f, weights_only=False)
        ep_r = res["ep_r"]
        cr = res["cr"]
    else:
        ep_r, cr = gen_training_samples(
            trial_type,
            total_t,
            dt=dt,
            p_j=1.0,
            task_name=task_name,
            numerosity_config=exp_config["numerosity_config"],
        )
        if cache_training_data:
            torch.save({"ep_r": ep_r, "cr": cr}, cache_f)
        else:
            print(f"skip training data cache for {run_name}")
    pass
    return ep_r, run_name


def iter_epoch_batches(ep_r, batch_size, task_name, sim_config):
    # Use indices for both on-disk and in-memory episodes, preserving the same shuffle order.
    relation_pair = get_balanced_relation_pair(task_name)
    indices = list(range(len(ep_r)))
    if relation_pair is not None and sim_config["balanced_relation_batches"]:
        if batch_size < 2:
            raise ValueError("Balanced relation batches require batch_size >= 2")
        trial_types = (
            ep_r.trial_types
            if isinstance(ep_r, (HDF5EpisodeDataset, InMemoryEpisodeDataset))
            else [episode[1] for episode in ep_r]
        )
        by_relation = {relation: [] for relation in relation_pair}
        for index, trial_type in enumerate(trial_types):
            by_relation[get_trial_relation(str(trial_type))].append(index)
        first_indices, second_indices = [by_relation[relation] for relation in relation_pair]
        first_per_batch = batch_size // 2
        second_per_batch = batch_size - first_per_batch
        num_batches = min(
            len(first_indices) // first_per_batch,
            len(second_indices) // second_per_batch,
        )
        if num_batches == 0:
            raise ValueError("Balanced batches require enough episodes from both relations")
        random.shuffle(first_indices)
        random.shuffle(second_indices)
        for b_id in range(num_batches):
            batch_indices = (
                first_indices[b_id * first_per_batch : (b_id + 1) * first_per_batch]
                + second_indices[b_id * second_per_batch : (b_id + 1) * second_per_batch]
            )
            random.shuffle(batch_indices)
            yield [ep_r[index] for index in batch_indices]
        return

    # Preserve in-place shuffling of ordinary lists across training epochs.
    if isinstance(ep_r, (HDF5EpisodeDataset, InMemoryEpisodeDataset)):
        random.shuffle(indices)
        for b_id in range(len(indices) // batch_size):
            batch_indices = indices[b_id * batch_size : (b_id + 1) * batch_size]
            yield [ep_r[index] for index in batch_indices]
    else:
        random.shuffle(ep_r)
        for b_id in range(len(ep_r) // batch_size):
            yield ep_r[b_id * batch_size : (b_id + 1) * batch_size]


def episode_cache(
    exp_config, trial_type, total_t, dt, filename, seed_value=None, purpose="training"
):
    # Include every input-generation setting in a short cache identity.
    return cache_file(
        {
            "kind": "episodes",
            "purpose": purpose,
            "p_j": 1.0 if purpose == "training" else 0.9999,
            "cap_per_type": None if purpose == "training" else 200,
            "version": NUMEROSITY_EPISODE_CACHE_VERSION,
            "task": exp_config["task"],
            "trial_type": trial_type,
            "numerosity": exp_config["numerosity_config"],
            "total_t": total_t,
            "dt": dt,
            "seed": exp_config["seed"] if seed_value is None else seed_value,
        },
        filename,
    )
