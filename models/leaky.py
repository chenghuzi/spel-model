"""DPX recurrent network with optional predictive-error input."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import NonNegativeLinear, SaturatedSoftplus, gen_ortho_matrix


class LeakyRNN(nn.Module):
    def __init__(self, n_in, n_h, n_o, config):
        super().__init__()
        self.config = config
        self.n_in = n_in
        self.n_h = n_h
        self.action_output_dim = n_o
        if config["predict_all"] > 0:
            n_o = n_in + n_o

        self.n_o = n_o
        self.init_hidden_val = config["init_hidden_val"]

        self.i2h = nn.Linear(n_in, n_h, bias=False)
        self.h2h = nn.Linear(n_h, n_h, bias=True)
        # self.h2o = nn.Linear(n_h, n_o, bias=False)
        self.h2o = NonNegativeLinear(n_h, n_o)  # nn.Linear(n_h, n_o)
        self.actf = SaturatedSoftplus(peak=20.0)

        # re-init the model weights
        config["sparsity"] = 0.8
        with torch.no_grad():
            self.i2h.weight.data = (
                torch.normal(torch.zeros(n_h, n_in), torch.ones(n_h, n_in)) / np.sqrt(n_in) * 1.0
            )

            # self.h2h.weight.data = torch.from_numpy(gen_ortho_matrix(n_h)).float() * 1.5 # this one is faster
            self.h2h.weight.data = torch.from_numpy(gen_ortho_matrix(n_h)).float() * 1.0
            # self.h2o.weight.data = torch.zeros(n_o, n_h)  # torch.from_numpy(gen_ortho_matrix(n_h)).float()
            # self.h2o.weight.bias = torch.zeros(n_o)
            self.h2hmask = (torch.rand(n_h, n_h) < config["sparsity"]).float()

    def forward(self, stimuli_seqs, actions=None, teacher_forcing=True):
        # stimuli_seqs of shape (N, num_steps, n_in)
        bs, num_steps, _ = stimuli_seqs.shape
        # stimuli_seqs

        noises = torch.zeros(bs, self.n_h).to(stimuli_seqs.device)
        # hiddens = (
        #     torch.ones(bs, self.n_h).to(stimuli_seqs.device) * self.init_hidden_val
        # )
        # hiddens = 1.0 + (
        #     torch.rand(bs, self.n_h).to(stimuli_seqs.device) * self.init_hidden_val
        # )
        hiddens = torch.rand(bs, self.n_h).to(stimuli_seqs.device) * self.init_hidden_val
        noises_list = []
        hiddens_list = []
        outputs_list = []

        outputs = torch.zeros(bs, self.n_o).to(stimuli_seqs.device)
        x = torch.zeros_like(stimuli_seqs[:, 0, :])
        x_list = []
        for step in range(num_steps):
            if self.config["predict_all"] > 0:
                # x = torch.cat([stimuli_seqs[:, step, :], outputs], dim=1)
                # - outputs.data # send prediction error into it.
                x = (
                    stimuli_seqs[:, step, :] - outputs[:, -self.n_in :]
                )  # send prediction error into the RNN.

                # simulate this as a network
                # x = (1 - self.config["alpha"]) * x.data + self.config[
                #     "alpha"
                # ] * F.softplus(outputs[:, -self.n_in :] - stimuli_seqs[:, step, :])
            else:
                x = stimuli_seqs[:, step, :]
            new_noises = (1 - self.config["alpha_noise"]) * noises + self.config[
                "alpha_noise"
            ] * torch.randn_like(noises) * np.sqrt(2.0 * self.config["alpha_noise"]) * self.config[
                "sigma_rec"
            ]

            rec_input = self.h2h(hiddens)
            # rec_input = F.linear(hiddens, self.h2h.weight * self.h2hmask.data, self.h2h.bias)

            new_hiddens = (1 - self.config["alpha"]) * hiddens + self.config["alpha"] * self.actf(
                # F.dropout(self.i2h(x) + rec_input + new_noises, p=0)
                self.i2h(x) + rec_input + new_noises
            )
            new_hiddens = F.dropout(new_hiddens, p=0.01)

            outputs = self.h2o(new_hiddens)
            noises_list.append(new_noises)
            hiddens_list.append(new_hiddens)
            outputs_list.append(outputs)
            x_list.append(x)

            noises = new_noises
            hiddens = new_hiddens

        return (
            torch.stack(noises_list, dim=1),
            torch.stack(hiddens_list, dim=1),
            torch.stack(outputs_list, dim=1),
            torch.stack(x_list, dim=1),
        )
