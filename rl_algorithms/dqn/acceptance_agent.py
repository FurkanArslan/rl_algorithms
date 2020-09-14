# -*- coding: utf-8 -*-
"""DQN agent for episodic tasks in Genius Environment.

- Author: Furkan Arslan
- Contact: furkan.arslan@ozu.edu.tr
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
         https://arxiv.org/pdf/1509.06461.pdf (Double DQN)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1511.06581.pdf (Dueling)
         https://arxiv.org/pdf/1706.10295.pdf (NoisyNet)
         https://arxiv.org/pdf/1707.06887.pdf (C51)
         https://arxiv.org/pdf/1710.02298.pdf (Rainbow)
         https://arxiv.org/pdf/1806.06923.pdf (IQN)
"""
from collections import deque
import os
import shutil
import time

import numpy as np
import torch
import wandb

from rl_algorithms.registry import AGENTS

from .agent import DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class DQNAgent2(DQNAgent):
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

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""

        self.args.cfg_path = self.args.acceptance_cfg_path

        DQNAgent._initialize(self)

        # init stack
        self.stack_size = self.args.stack_size
        self.stack_buffer = deque(maxlen=self.args.stack_size)
        self.stack_buffer_2 = deque(maxlen=self.args.stack_size)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state2(
            state, self.stack_buffer, convert_to_tensor=True
        )

        # if initial random action should be conducted
        if self.episode_step == 0 or self.episode_step == 1:
            return np.array([0])

        # epsilon greedy policy
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array([self.env.action_space.sample()])
        else:
            with torch.no_grad():
                selected_action = self.learner.dqn(state).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    # pylint: disable=no-self-use
    def _preprocess_state2(
        self,
        state: np.ndarray,
        stack_buffer: deque,
        insert_stack=True,
        convert_to_tensor=False,
    ) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        if self.stack_size > 1:
            if insert_stack:
                stack_buffer.append(state)
            state = np.asarray(stack_buffer).flatten()

        if convert_to_tensor:
            state = torch.FloatTensor(state).to(device)

        return state

    def add_transition_to_memory(self, state, action, reward, next_state, done):
        state = self._preprocess_state2(state, self.stack_buffer, insert_stack=False)
        next_state = self._preprocess_state2(next_state, self.stack_buffer_2)
        action = np.asarray(action)
        transition = (state, action, reward, next_state, done)

        self._add_transition_to_memory(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        utility, loss, score, avg_time_cost = log_value

        if self.args.log:
            log = (
                "[AC-INFO] episode %d, episode step: %d, total step: %d, "
                "total score: %f utility: %.3f, "
                "epsilon: %f, loss: %f, avg q-value: %f (spent %.6f sec/step)\n"
                % (
                    self.i_episode,
                    self.episode_step,
                    self.total_step,
                    score,
                    utility,
                    self.epsilon,
                    loss[0],
                    loss[1],
                    avg_time_cost,
                )
            )

            self._write_log_file(log)

            wandb.log(
                {
                    "score": score,
                    "utility": utility,
                    "epsilon": self.epsilon,
                    "dqn loss": loss[0],
                    "avg q values": loss[1],
                    "time per each step": avg_time_cost,
                    "total_step": self.total_step,
                },
                step=self.i_episode,
            )

    def start_training(self):
        # logger
        if self.args.log:
            self.set_wandb()

    def start_episode(self, state):
        DQNAgent.start_episode(self, state)
        self.score = 0
        self.episode_step = 0
        self.loss_episode = list()

        self.t_begin = time.time()

        if self.stack_size > 1:
            for _ in range(self.stack_size):
                self.stack_buffer.append(state)
            for _ in range(self.stack_size):
                self.stack_buffer_2.append(state)

        self.i_episode += 1

    def end_episode(self, utility):
        t_end = time.time()
        avg_time_cost = (t_end - self.t_begin) / self.episode_step

        if self.loss_episode:
            avg_loss = np.vstack(self.loss_episode).mean(axis=0)
        else:
            avg_loss = [0, 0]

        log_value = (utility, avg_loss, self.score, avg_time_cost)

        self.write_log(log_value)

        if self.i_episode % self.args.save_period == 0:
            if self.total_step >= self.hyper_params.update_starts_from:
                self.learner.save_params(self.i_episode)

    def make_one_step(self, curr_state, action, reward, next_state, done):
        self.total_step += 1
        self.episode_step += 1

        self.add_transition_to_memory(curr_state, action, reward, next_state, done)

        if len(self.memory) >= self.hyper_params.update_starts_from:
            if self.total_step % self.hyper_params.train_freq == 0:
                for _ in range(self.hyper_params.multiple_update):
                    experience = self.sample_experience()
                    info = self.learner.update_model(experience)
                    loss = info[0:2]
                    indices, new_priorities = info[2:4]
                    self.loss_episode.append(loss)  # for logging
                    self.memory.update_priorities(indices, new_priorities)

            # decrease epsilon
            self.epsilon = max(
                self.epsilon
                - (self.max_epsilon - self.min_epsilon)
                * self.hyper_params.epsilon_decay,
                self.min_epsilon,
            )

            # increase priority beta
            fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
            self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        self.score += reward

    def _write_log_file(self, log):
        if self.args.log:
            with open(self.log_filename, "a") as file:
                file.write(log + "\n")

            wandb.save(self.log_filename)

    def set_wandb(self):
        wandb.config.update(vars(self.args))
        wandb.config.update(self.hyper_params, allow_val_change=True)

        shutil.copy(
            self.args.acceptance_cfg_path, os.path.join(wandb.run.dir, "ac_config.py")
        )

        self.log_filename = self._init_log_file()

    def _init_log_file(self):
        logs_name = "logs_" + self.log_cfg.curr_time

        return os.path.join(wandb.run.dir, logs_name + ".log")
