"""Construct the configured manuscript network family."""

from models.leaky import LeakyRNN
from models.visual import CNNLeakyRNN


def build_model(n_in, n_h, n_o, config):
    # DPX uses LeakyRNN; visual numerosity uses CNNLeakyRNN.
    if config["rnn_type"] == "CNNLeakyRNN":
        return CNNLeakyRNN(n_in, n_h, n_o, config)
    if config["rnn_type"] == "LeakyRNN":
        return LeakyRNN(n_in, n_h, n_o, config)
    raise ValueError(f"Unsupported rnn_type: {config['rnn_type']}")
