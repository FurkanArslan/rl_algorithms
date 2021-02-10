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
class DQNAgent5(DQNAgent):
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
        self.args.load_from = self.args.load_acceptance_from
        self.hyper_params.buffer_size = self.hyper_params.dqn_buffer_size
        self.hyper_params.batch_size = self.hyper_params.dqn_batch_size

        DQNAgent._initialize(self)

        del self.hyper_params.buffer_size
        del self.hyper_params.batch_size

        self.negotiation_thread = []

    def select_action(self, state: np.ndarray, next_bid: float = 1.0) -> np.ndarray:
        """Select an action from the input space."""
        threshold = 1
        self.curr_state = state

        # if initial random action should be conducted
        if self.episode_step == 0 or self.episode_step == 1:
            return np.array(0)

        if self.curr_state[2] >= self.hyper_params.time_threshold:
            threshold = max(self.negotiation_thread)

        action = (
            state[1] >= next_bid
            and state[2] >= self.hyper_params.time_threshold
            and state[1] >= threshold
        )

        return [action]

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

    def start_episode(self, state, min_threshold=None):
        self.score = 0
        self.episode_step = 0
        self.loss_episode = list()

        self.t_begin = time.time()
        self.min_threshold = min_threshold

        self.negotiation_thread = []

        self.i_episode += 1

    def end_episode(self, utility, ac_action=None):
        if not self.args.test:
            t_end = time.time()
            avg_time_cost = (t_end - self.t_begin) / self.episode_step

            if self.loss_episode:
                avg_loss = np.vstack(self.loss_episode).mean(axis=0)
            else:
                avg_loss = [0, 0]

            log_value = (utility, avg_loss, self.score, avg_time_cost)

            self.write_log(log_value)
        else:
            log = "[INFO] test %d\tstep: %d\ttotal score: %.2f" % (
                self.i_episode,
                self.episode_step,
                self.score,
            )

            self._write_log_file(log)

    def make_one_step(self, curr_state, action, reward, next_state, done):
        self.score += reward
        self.episode_step += 1
        self.total_step += 1

        self.negotiation_thread.append(curr_state[1])

        log = (
            "[AC-INFO] Step -"
            + str(self.episode_step)
            + ":"
            + " state: "
            + str(curr_state)
            + " next_state: "
            + str(next_state)
            + " score: {:.2f}".format(self.score)
        )

        self._write_log_file(log)

    def _write_log_file(self, log):
        if self.args.log:
            with open(self.log_filename, "a") as file:
                file.write(log + "\n")

    def set_wandb(self):
        wandb.config.update(vars(self.args))
        wandb.config.update(self.hyper_params, allow_val_change=True)

        shutil.copy(
            self.args.acceptance_cfg_path, os.path.join(wandb.run.dir, "ac_config.py")
        )

        self.log_filename = os.path.join(wandb.run.dir, "experiment.log")
