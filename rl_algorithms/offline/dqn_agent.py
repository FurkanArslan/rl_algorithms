# -*- coding: utf-8 -*-
"""DQN agent for episodic tasks in OpenAI Gym.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
         https://arxiv.org/pdf/1509.06461.pdf (Double DQN)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1511.06581.pdf (Dueling)
         https://arxiv.org/pdf/1706.10295.pdf (NoisyNet)
         https://arxiv.org/pdf/1707.06887.pdf (C51)
         https://arxiv.org/pdf/1710.02298.pdf (Rainbow)
         https://arxiv.org/pdf/1806.06923.pdf (IQN)
"""

import argparse
import time
from typing import Tuple

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.buffer.supervise_buffer import SuperviseBuffer
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class OfflineDQNAgent(Agent):
    """DQN interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (PrioritizedReplayBuffer): replay memory
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        epsilon (float): parameter for epsilon greedy policy
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        per_beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        log_cfg: ConfigDict,
    ):
        """Initialize."""
        Agent.__init__(self, env, env_info, args, log_cfg)

        self.curr_state = np.zeros(1)
        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.learner_cfg.args = self.args
        self.learner_cfg.env_info = self.env_info
        self.learner_cfg.hyper_params = self.hyper_params
        self.learner_cfg.log_cfg = self.log_cfg
        self.learner_cfg.device = device

        self.dataset = SuperviseBuffer(
            self.hyper_params.experience_dir, self.hyper_params.from_disk
        )
        self.dataloader = DataLoader(
            self.dataset, self.hyper_params.batch_size, shuffle=True
        )
        self._initialize()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.learner = build_learner(self.learner_cfg)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state = self._preprocess_state(state)
                selected_action = self.learner.dqn(state).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    # pylint: disable=no-self-use
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = torch.FloatTensor(state).to(device)
        return state

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, avg_time_cost = log_value
        print(
            "[INFO] epoch %d, iter: %d, total iter: %d\n"
            "loss: %f, avg q-value: %f (spent %.6f sec/step)\n"
            % (i, self.episode_step, self.total_step, loss[0], loss[1], avg_time_cost,)
        )

        if self.args.log:
            wandb.log(
                {
                    "dqn loss": loss[0],
                    "avg q values": loss[1],
                    "time per each step": avg_time_cost,
                    "total_step": self.total_step,
                }
            )

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def sample_experience(self) -> Tuple[torch.Tensor, ...]:
        experiences_1 = self.memory.sample(self.per_beta)
        experiences_1 = numpy2floattensor(experiences_1[:6]) + experiences_1[6:]
        if self.use_n_step:
            indices = experiences_1[-2]
            experiences_n = self.memory_n.sample(indices)
            return experiences_1, numpy2floattensor(experiences_n)

        return experiences_1

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            self.episode_step = 0
            losses = list()
            t_begin = time.time()
            for _ in range(self.hyper_params.multiple_update):
                for batch_idx, experience in tqdm(enumerate(self.dataloader)):
                    self.total_step += 1
                    self.episode_step += 1
                    info = self.learner.update_model(experience)
                    loss = info[0:2]
                    losses.append(loss)  # for logging
                    if batch_idx >= 2000:
                        break

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, avg_time_cost)
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()
