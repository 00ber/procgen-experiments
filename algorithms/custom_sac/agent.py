import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .replay_buffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork, ValueNetwork


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], env=None, gamma=0.99, tau = 0.005,
                 n_actions=2, max_size=1000000, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        if not env:
            raise Exception("Must pass environment")
        
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.conv = nn.Sequential(*conv_seqs)

        self.actor = ActorNetwork(alpha, 256, n_actions=n_actions, max_action=1, name="actor")
        self.critic_1 = CriticNetwork(beta, 256, n_actions, name="critic_1")
        self.critic_2 = CriticNetwork(beta, 256, n_actions, name="critic_2")
        self.value = ValueNetwork(beta, 256, n_actions=n_actions, name="value")
        self.target_value = ValueNetwork(beta, 256, n_actions=n_actions, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def get_conv(self, x):
        return self.conv(x.permute((0, 3, 1, 2)) / 255.0)
    
    def choose_action(self, observation):
        state = torch.Tensor(observation)
        state = self.get_conv(state)
        _, actions, _ = self.actor(state)
        return actions.numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print(".... saving models ....")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print(".... loading models ....")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done)
        state_ = torch.tensor(new_state, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)

        state = self.get_conv(state)
        print(state.shape)
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        _, actions, log_probs = self.actor.forward(state)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.foward(state, actions)
        q2_new_policy = self.critic_2.foward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        _, actions, log_probs = self.actor.forward(state)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.foward(state, actions)
        q2_new_policy = self.critic_2.foward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
