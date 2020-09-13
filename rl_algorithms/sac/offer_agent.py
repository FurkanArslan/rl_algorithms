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
class SACAgent2(SACAgent):
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

        SACAgent._initialize(self)

        # init stack
        self.stack_size = self.args.stack_size
        self.stack_buffer = deque(maxlen=self.args.stack_size)
        self.stack_buffer_2 = deque(maxlen=self.args.stack_size)

        self.scores = list()
        self.utilities = list()
        self.rounds = list()
        self.opp_utilities = list()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state2(
            state, self.stack_buffer, convert_to_tensor=True
        )

        # if initial random action should be conducted
        if (
            self.total_step < self.hyper_params.initial_random_action
            and not self.args.test
        ):
            return np.array(self.env.action_space.sample())

        with torch.no_grad():
            if self.args.test:
                _, _, _, selected_action, _ = self.learner.actor(state)
            else:
                selected_action, _, _, _, _ = self.learner.actor(state)

        return selected_action.detach().cpu().numpy()

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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def add_transition_to_memory(self, state, action, reward, next_state, done):
        state = self._preprocess_state2(state, self.stack_buffer, insert_stack=False)
        next_state = self._preprocess_state2(next_state, self.stack_buffer_2)
        action = np.asarray(action)
        transition = (state, action, reward, next_state, done)

        self._add_transition_to_memory(transition)

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        self.memory.add(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        utility, loss, score, policy_update_freq, avg_time_cost = log_value
        total_loss = loss.sum() if loss is not None else 0

        if self.args.log:
            log = (
                "[OFFER-INFO] episode %d, episode_step %d, total step %d, total score: %.3f"
                " utility: %.3f total loss: %.3f (spent %.6f sec/step)"
                % (
                    self.i_episode,
                    self.episode_step,
                    self.total_step,
                    score,
                    utility,
                    total_loss,
                    avg_time_cost,
                )
            )

            self._write_log_file(log)

            wandb.log(
                {
                    "score": score,
                    "utility": utility,
                    "round": self.episode_step,
                    "total loss": total_loss,
                    "actor loss": loss[0] * policy_update_freq
                    if loss is not None
                    else 0,  # actor loss,
                    "qf_1 loss": loss[1] if loss is not None else 0,  # qf_1 loss
                    "qf_2 loss": loss[2] if loss is not None else 0,  # qf_2 loss
                    "vf loss": loss[3] if loss is not None else 0,  # vf loss
                    "alpha loss": loss[4] if loss is not None else 0,  # alpha loss
                    "time per each step": avg_time_cost,
                }
            )

    def start_training(self):
        # logger
        if self.args.log:
            self.set_wandb()

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                state = next_state
                score += reward

                # training
                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
                        experience = self.memory.sample()
                        experience = numpy2floattensor(experience)
                        loss = self.learner.update_model(experience)
                        loss_episode.append(loss)  # for logging

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                log_value = (
                    self.i_episode,
                    avg_loss,
                    score,
                    self.hyper_params.policy_update_freq,
                    avg_time_cost,
                )
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()

    def start_episode(self, state):
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

        self._write_log_file("**** Starting - " + str(self.i_episode) + " ****")

    def make_one_step(self, curr_state, action, reward, next_state, done):
        self.total_step += 1
        self.episode_step += 1

        log = (
            "[OFFER INFO] Step -"
            + str(self.episode_step)
            + ": state: "
            + str(curr_state)
            + " next_state: "
            + str(next_state)
            + " action: "
            + str(action[0])
            + " reward: "
            + str(reward)
            + " done:"
            + str(done)
        )

        self._write_log_file(log)

        self.add_transition_to_memory(curr_state, action, reward, next_state, done)

        self.score += reward

        if len(self.memory) >= self.hyper_params.batch_size:
            for _ in range(self.hyper_params.multiple_update):
                experience = self.memory.sample()
                experience = numpy2floattensor(experience)
                loss = self.learner.update_model(experience)
                self.loss_episode.append(loss)  # for logging

    def end_episode(self, utility):
        t_end = time.time()
        avg_time_cost = (t_end - self.t_begin) / self.episode_step

        self.scores.append(self.score)
        self.utilities.append(utility)
        self.rounds.append(self.episode_step)

        if self.loss_episode:
            avg_loss = np.vstack(self.loss_episode).mean(axis=0)
        else:
            avg_loss = None

        log_value = (
            utility,
            avg_loss,
            self.score,
            self.hyper_params.policy_update_freq,
            avg_time_cost,
        )

        self.write_log(log_value)

        if self.i_episode % self.args.save_period == 0:
            self.learner.save_params(self.i_episode)

            wandb.log(
                {
                    "mean_scores": np.vstack(self.scores).mean(axis=0),
                    "mean_utilities": np.vstack(self.utilities).mean(axis=0),
                    "mean_rounds": np.vstack(self.rounds).mean(axis=0),
                    "mean_opp_utilities": np.vstack(self.opp_utilities).mean(axis=0),
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

            wandb.save(self.log_filename)

    def set_wandb(self):
        wandb.config.update(vars(self.args))
        wandb.config.update(self.hyper_params)

        shutil.copy(
            self.args.offer_cfg_path, os.path.join(wandb.run.dir, "offer_config.py")
        )

        self.log_filename = self._init_log_file()

    def _init_log_file(self):
        logs_name = "logs_" + self.log_cfg.curr_time

        return os.path.join(wandb.run.dir, logs_name + ".txt")
