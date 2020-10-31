# -*- coding: utf-8 -*-
"""Train or test algorithms on Genius-v0.

- Author: Furkan Arslan
- Contact: furkan.arslan@ozu.edu.tr
"""

import argparse
import datetime
from os import environ, getpid

from flask import Flask, request
from flask_cors import cross_origin
import gym
import numpy as np
import wandb

from rl_algorithms import build_agent
from rl_algorithms.common.abstract.agent import Agent
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.utils import Config

# app instance
app = Flask(__name__)

offerAgent: Agent
acceptanceAgent: Agent


def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=777, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--offer-cfg-path",
        type=str,
        default="./configs/genius/sac.py",
        help="offer config path",
    )
    parser.add_argument(
        "--acceptance-cfg-path",
        type=str,
        default="./configs/genius/dqn.py",
        help="acceptance config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--load_acceptance_from",
        type=str,
        default=None,
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument(
        "--load_offer_from",
        type=str,
        default=None,
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument(
        "--off-render", dest="render", action="store_false", help="turn off rendering"
    )
    parser.add_argument(
        "--render-after",
        type=int,
        default=0,
        help="start rendering after the input number of episode",
    )
    parser.add_argument(
        "--log", dest="log", action="store_true", help="turn on logging"
    )
    parser.add_argument(
        "--save-period", type=int, default=250, help="save model period"
    )
    parser.add_argument(
        "--episode-num", type=int, default=1500, help="total episode num"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=300, help="max episode step"
    )
    parser.add_argument(
        "--interim-test-num",
        type=int,
        default=10,
        help="number of test during training",
    )
    parser.add_argument(
        "--demo-path", type=str, help="demonstration path for learning from demo",
    )
    parser.add_argument(
        "--integration-test",
        dest="integration_test",
        action="store_true",
        help="indicate integration test",
    )
    parser.add_argument(
        "--opponent", type=str, default="test", help="choose an algorithm"
    )
    parser.add_argument(
        "--stack-size", type=int, default=4, help="stack size for experience replay"
    )
    parser.add_argument("--port", type=int, default=1501, help="port")

    return parser.parse_args()


def build_env(env_name, seed):
    # env initialization
    env = gym.make(env_name)

    # set a random seed
    common_utils.set_random_seed(seed, env)

    return env


def get_config(
    cfg_path, env, env_name, is_discrete=False, integration_test=False
) -> Config:
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = Config.fromfile(cfg_path)

    # If running integration test, simplify experiment
    if integration_test:
        cfg = common_utils.set_cfg_for_intergration_test(cfg)

    cfg.agent.env_info = dict(
        name=env_name,
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_discrete=is_discrete,
    )

    cfg.agent.log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time)

    return cfg


def build_agent_from_config(cfg, env) -> Agent:
    build_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, build_args)

    return agent


def build_acceptance_agent() -> Agent:
    # env initialization
    env = build_env("Genius-v1", args.seed)

    cfg = get_config(args.acceptance_cfg_path, env, "Genius-v1", is_discrete=True)

    agent = build_agent_from_config(cfg, env)

    return agent


def build_offer_agent() -> Agent:
    # env initialization
    env = build_env("GeniusContinuous-v1", args.seed)

    cfg = get_config(args.offer_cfg_path, env, "GeniusContinuous-v1", is_discrete=True)

    agent = build_agent_from_config(cfg, env)

    return agent


def build_agents() -> (Agent, Agent):
    """Main."""

    acceptance_agent = build_acceptance_agent()
    offer_agent = build_offer_agent()

    if args.log:
        NOWTIMES = datetime.datetime.now()
        curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

        wandb.init(
            project="genius_negotiation_agent",
            name=f"SAC-DQN/{curr_time}",
            group=args.opponent,
        )

    return acceptance_agent, offer_agent


@app.route("/reset", methods=["POST"])
@cross_origin()
def reset():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get the state from the data.
    state = data["state"][0]

    offerAgent.start_episode(state)
    acceptanceAgent.start_episode(state)

    return "success"


@app.route("/getAction", methods=["POST"])
@cross_origin()
def getAction():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get the state from the data.
    state = data["state"][0]

    ac_action = acceptanceAgent.select_action(state)

    try:
        # opponent offer is not accepted and new offer will be offered
        if ac_action == np.array(0):
            action = offerAgent.select_action(state)

            return str(action[0])
    except Exception:
        print("HATA: ", ac_action)

    # opponent offer is accepted
    return "-1"


@app.route("/postToMemory", methods=["POST"])
@cross_origin()
def postToMemory():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get relevant data.
    state = data["state"][0]
    acAction = data["acAction"][0]
    offerAction = data["offerAction"]
    reward = data["reward"][0]
    next_state = data["next_state"][0]
    done = data["done"][0]

    # make one step & update agent parameters
    offerAction = (np.asarray(offerAction) * 2) - 1
    offerAgent.make_one_step(state, offerAction, reward, next_state, done)
    acceptanceAgent.make_one_step(state, acAction, reward, next_state, done)

    if done:
        offerAgent.end_episode(reward)
        acceptanceAgent.end_episode(reward)

    return "success"


@app.route("/logHistory", methods=["POST"])
@cross_origin()
def log_to_history():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get relevant data.
    reward = data["utility"][0]

    offerAgent.log_opponent_utility(reward)

    return "success"


if __name__ == "__main__":
    args = parse_args()

    acceptanceAgent, offerAgent = build_agents()

    acceptanceAgent.start_training()
    offerAgent.start_training()

    print(">>", "starting debug environment", "[%s]" % getpid())

    app.debug = False  # non refreshing
    host = environ.get("IP", "localhost")
    port = int(environ.get("PORT", args.port))
    app.run(host=host, port=port)
