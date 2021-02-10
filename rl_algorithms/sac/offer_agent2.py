# -*- coding: utf-8 -*-
"""SAC agent for continuous tasks in Genius Environment.

- Author: Furkan Arslan
- Contact: furkan.arslan@ozu.edu.tr
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
"""
from collections import deque
import os
import shutil
import time
from typing import Tuple

import numpy as np
import torch
import wandb

from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS

from .agent import SACAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class SACAgent3(SACAgent):
    """SAC agent interacting with environment.

    Attrtibutes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        update_step (int): step number of updates
        i_episode (int): current episode number

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.args.cfg_path = self.args.offer_cfg_path
        self.args.load_from = self.args.load_offer_from
        self.hyper_params.buffer_size = self.hyper_params.sac_buffer_size
        self.hyper_params.batch_size = self.hyper_params.sac_batch_size

        SACAgent._initialize(self)

        del self.hyper_params.buffer_size
        del self.hyper_params.batch_size

        self.scores = list()
        self.utilities = list()
        self.rounds = list()
        self.opp_utilities = list()

    def start_training(self):
        # logger
        if self.args.log:
            self.set_wandb()

    def start_episode(self, state):
        self.score = 0
        self.episode_step = 0
        self.loss_episode = list()

        self.t_begin = time.time()

        self.i_episode += 1

        self._write_log_file("**** Starting - " + str(self.i_episode) + " ****")

    def make_one_step(self, curr_state, action, reward, next_state, done):
        self.score += reward
        self.episode_step += 1
        self.total_step += 1

    def end_episode(self, utility):
        self.scores.append(self.score)
        self.utilities.append(utility)
        self.rounds.append(self.episode_step)

        if self.i_episode % 250 == 0:
            try:
                wandb.log(
                    {
                        "mean_scores": np.vstack(self.scores).mean(axis=0),
                        "mean_utilities": np.vstack(self.utilities).mean(axis=0),
                        "mean_rounds": np.vstack(self.rounds).mean(axis=0),
                        "mean_opp_utilities": np.vstack(self.opp_utilities).mean(
                            axis=0
                        ),
                    },
                    step=self.i_episode,
                )
            except Exception:
                self._write_log_file("HATA: " + str(self.episode_step))

                wandb.log(
                    {
                        "mean_scores": np.vstack(self.scores).mean(axis=0),
                        "mean_utilities": np.vstack(self.utilities).mean(axis=0),
                        "mean_rounds": np.vstack(self.rounds).mean(axis=0),
                    },
                    step=self.i_episode,
                )

            self.scores = list()
            self.utilities = list()
            self.rounds = list()
            self.opp_utilities = list()

    def log_opponent_utility(self, utility):
        self.opp_utilities.append(utility)

        wandb.log({"opp_utility": utility}, step=self.i_episode)

    def _write_log_file(self, log):
        if self.args.log:
            with open(self.log_filename, "a") as file:
                file.write(log + "\n")

    def set_wandb(self):
        wandb.config.update(vars(self.args))
        wandb.config.update(self.hyper_params, allow_val_change=True)

        shutil.copy(
            self.args.offer_cfg_path, os.path.join(wandb.run.dir, "offer_config.py")
        )

        self.log_filename = os.path.join(wandb.run.dir, "experiment.log")
