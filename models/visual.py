"""CNN-backed recurrent network for visual numerosity."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import (
    CNNVisualEncoder,
    build_readout,
    build_recurrent_activation,
    gen_ortho_matrix,
)


class CNNLeakyRNN(nn.Module):
    def __init__(self, n_in, n_h, n_o, config):
        super().__init__()
        self.config = config
        self.n_in = n_in
        self.n_h = n_h
        self.action_output_dim = n_o
        self.n_o = n_o
        self.n_z = int(config["latent_dim"])
        self.context_dim = int(config["phase_context_dim"])
        self.init_hidden_val = config["init_hidden_val"]
        self.rnn_cell_type = str(config["rnn_cell"]).lower()
        self.actf = build_recurrent_activation(config)
        self.prediction_error_input = bool(config["prediction_error_input"])
        self.use_ema_target_encoder = bool(config["use_ema_target_encoder"])
        self.latent_ema_decay = float(config["latent_ema_decay"])
        self.latent_activation = str(config["latent_activation"]).lower()
        self.zero_delay_latent = bool(config["zero_delay_latent"])
        if not self.prediction_error_input:
            raise ValueError("CNNLeakyRNN requires prediction_error_input=true")
        if float(config["latent_pred_weight"]) <= 0.0:
            raise ValueError("prediction-error input requires latent prediction")

        self.encoder = CNNVisualEncoder(n_in, self.n_z, config)
        # Every visual model has latent prediction; only its EMA update policy is optional.
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.latent_predictor = nn.Linear(n_h, self.n_z)
        self.context_predictor = nn.Linear(n_h, self.context_dim)
        self.input_dim = self.n_z + self.context_dim + self.action_output_dim
        if self.rnn_cell_type != "leaky":
            raise ValueError(f"Unsupported CNNLeakyRNN rnn_cell: {self.rnn_cell_type}")
        self.i2h = nn.Linear(self.input_dim, n_h, bias=False)
        self.h2h = nn.Linear(n_h, n_h, bias=True)
        self.h2o = build_readout(n_h, self.n_o, config)

        self.i2h.weight.data = torch.normal(
            torch.zeros(n_h, self.input_dim),
            torch.ones(n_h, self.input_dim),
        ) / np.sqrt(self.input_dim)
        self.h2h.weight.data = torch.from_numpy(gen_ortho_matrix(n_h)).float()

    def transform_latent(self, latent):
        if self.latent_activation in {"none", "identity"}:
            return latent
        if self.latent_activation == "sigmoid":
            return torch.sigmoid(latent)
        raise ValueError(f"Unsupported CNNLeakyRNN latent_activation: {self.latent_activation}")

    def encode_visual_latent(self, flat_images, encoder=None):
        if encoder is None:
            encoder = self.encoder
        return self.transform_latent(encoder(flat_images))

    def encode_target_latent(self, flat_images):
        if not self.use_ema_target_encoder:
            return self.encode_visual_latent(flat_images).detach()
        with torch.no_grad():
            return self.encode_visual_latent(flat_images, encoder=self.target_encoder)

    def predict_visual_latent(self, hiddens):
        return torch.sigmoid(self.latent_predictor(hiddens))

    def predict_phase_context(self, hiddens):
        return F.softmax(self.context_predictor(hiddens), dim=-1)

    def predict_action_outputs(self, hiddens):
        # Decode hold/release from the recurrent state.
        return self.h2o(hiddens)

    def input_slices(self):
        latent_slice = slice(0, self.n_z)
        context_slice = slice(self.n_z, self.n_z + self.context_dim)
        action_slice = slice(self.n_z + self.context_dim, self.input_dim)
        return latent_slice, context_slice, action_slice

    def apply_visual_latent_policy(self, latent, phase_context):
        if not self.zero_delay_latent:
            return latent
        delay_mask = phase_context[..., 2:3]
        return latent * (1.0 - delay_mask)

    @torch.no_grad()
    def update_target_encoder(self):
        if not self.use_ema_target_encoder:
            return
        decay = self.latent_ema_decay
        for target_param, online_param in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters(),
        ):
            target_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
        for target_buffer, online_buffer in zip(
            self.target_encoder.buffers(),
            self.encoder.buffers(),
        ):
            target_buffer.copy_(online_buffer)

    def make_phase_context(self, batch_size, num_steps, device, dtype):
        dt = float(self.config["dt"])
        times = torch.arange(num_steps, device=device, dtype=torch.float32) * dt
        phase_indices = self._phase_indices_from_times(times)
        context = F.one_hot(phase_indices, num_classes=self.context_dim).to(dtype=dtype)
        return context.unsqueeze(0).expand(batch_size, -1, -1)

    def _phase_indices_from_times(self, times):
        fixation_ms = float(self.config["fixation_ms"])
        sample_ms = float(self.config["sample_ms"])
        delay_ms = float(self.config["delay_ms"])
        sample_start = fixation_ms
        sample_end = sample_start + sample_ms
        test_start = sample_end + delay_ms
        phase_indices = torch.zeros_like(times, dtype=torch.long)
        phase_indices[(times > sample_start) & (times <= sample_end)] = 1
        phase_indices[(times > sample_end) & (times <= test_start)] = 2
        phase_indices[times > test_start] = 3
        return phase_indices

    def phase_context_from_time(self, times_ms, device, dtype):
        times = torch.as_tensor(times_ms, device=device, dtype=torch.float32)
        phase_indices = self._phase_indices_from_times(times)
        return F.one_hot(phase_indices, num_classes=self.context_dim).to(dtype=dtype)

    def default_previous_action(self, batch_size, device, dtype):
        action = torch.zeros(
            batch_size,
            self.action_output_dim,
            device=device,
            dtype=dtype,
        )
        action[:, 0] = 1.0
        return action

    @torch.no_grad()
    def action_feedback_from_prediction(self, action_pred):
        action_id = action_pred.argmax(dim=1)
        return F.one_hot(
            action_id,
            num_classes=self.action_output_dim,
        ).to(dtype=action_pred.dtype)

    def action_prediction_for_error(self, action_pred):
        return F.softmax(action_pred, dim=1)

    def initial_state(self, batch_size, device, dtype):
        noises = torch.zeros(batch_size, self.n_h, device=device, dtype=dtype)
        hiddens = (
            torch.ones(batch_size, self.n_h, device=device, dtype=dtype) * self.init_hidden_val
        )
        current_prediction = self.predict_action_outputs(hiddens)
        previous_action = self.default_previous_action(batch_size, device, dtype)
        return noises, hiddens, current_prediction, previous_action

    def step_once(
        self,
        stimulus,
        phase_context,
        previous_action,
        _current_prediction,
        noises,
        hiddens,
    ):
        clean_z = self.encode_visual_latent(stimulus)
        clean_z = self.apply_visual_latent_policy(clean_z, phase_context)
        z = clean_z
        z_hat = self.predict_visual_latent(hiddens)
        context_hat = self.predict_phase_context(hiddens)
        action_hat_prev = self.action_prediction_for_error(_current_prediction)
        rnn_input = torch.cat(
            [
                z - z_hat,
                phase_context - context_hat,
                previous_action - action_hat_prev,
            ],
            dim=-1,
        )
        new_noises = (1 - self.config["alpha_noise"]) * noises + self.config[
            "alpha_noise"
        ] * torch.randn_like(noises) * np.sqrt(2.0 * self.config["alpha_noise"]) * self.config[
            "sigma_rec"
        ]
        rec_input = self.h2h(hiddens)
        new_hiddens = (1 - self.config["alpha"]) * hiddens + self.config["alpha"] * self.actf(
            self.i2h(rnn_input) + rec_input + new_noises
        )
        new_hiddens = F.dropout(
            new_hiddens,
            p=float(self.config["dropout"]),
            training=self.training,
        )
        outputs = self.predict_action_outputs(new_hiddens)
        return new_noises, new_hiddens, outputs, rnn_input, clean_z

    def masked_mse(self, predicted, target, mask):
        weights = mask.to(device=predicted.device, dtype=predicted.dtype)
        while weights.dim() < predicted.dim():
            weights = weights.unsqueeze(-1)
        denom = weights.sum() * predicted.shape[-1]
        # why: a supervised sequence must contain valid next-step targets, even after padding.
        assert denom.item() > 0, "Prediction loss requires at least one valid next-step target"
        return ((predicted - target).square() * weights).sum() / denom

    def latent_prediction_mask(self, phase_context, mask):
        return mask[:, 1:]

    def latent_prediction_loss(self, hidden_seq, target_latents, phase_context, mask):
        predictor_state = hidden_seq[:, :-1, :]
        predicted_next = self.predict_visual_latent(predictor_state)
        target_next = target_latents[:, 1:, :].detach()
        next_mask = self.latent_prediction_mask(phase_context, mask)
        return self.masked_mse(predicted_next, target_next, next_mask)

    def context_prediction_loss(self, hidden_seq, phase_context, mask):
        predicted_next = self.predict_phase_context(hidden_seq[:, :-1, :])
        target_next = phase_context[:, 1:, :]
        return self.masked_mse(predicted_next, target_next, mask[:, 1:])

    def forward(self, stimuli_seqs, actions=None, teacher_forcing=True, mask=None):
        bs, num_steps, _ = stimuli_seqs.shape
        if mask is None:
            mask = torch.ones(
                bs,
                num_steps,
                device=stimuli_seqs.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.to(device=stimuli_seqs.device, dtype=torch.bool)
        phase_context = self.make_phase_context(
            bs,
            num_steps,
            stimuli_seqs.device,
            stimuli_seqs.dtype,
        )
        noises, hiddens, current_prediction, previous_action = self.initial_state(
            bs,
            stimuli_seqs.device,
            stimuli_seqs.dtype,
        )
        noises_list = []
        hiddens_list = []
        outputs_list = []
        inputs_list = []

        for step in range(num_steps):
            noises, hiddens, outputs, rnn_input, clean_z = self.step_once(
                stimuli_seqs[:, step, :],
                phase_context[:, step, :],
                previous_action,
                current_prediction,
                noises,
                hiddens,
            )
            current_prediction = outputs
            noises_list.append(noises)
            hiddens_list.append(hiddens)
            outputs_list.append(outputs)
            inputs_list.append(rnn_input)

            if actions is not None and teacher_forcing:
                previous_action = actions[:, step, :].detach()
            else:
                previous_action = self.action_feedback_from_prediction(outputs)

        hiddens_seq = torch.stack(hiddens_list, dim=1)
        flat_stimuli = stimuli_seqs.reshape(bs * num_steps, self.n_in)
        target_latents = self.encode_target_latent(flat_stimuli).reshape(
            bs,
            num_steps,
            self.n_z,
        )
        target_latents = self.apply_visual_latent_policy(
            target_latents,
            phase_context,
        )
        self.latest_latent_pred_loss = self.latent_prediction_loss(
            hiddens_seq,
            target_latents,
            phase_context,
            mask,
        )
        self.latest_context_pred_loss = self.context_prediction_loss(
            hiddens_seq,
            phase_context,
            mask,
        )
        return (
            torch.stack(noises_list, dim=1),
            hiddens_seq,
            torch.stack(outputs_list, dim=1),
            torch.stack(inputs_list, dim=1),
        )
