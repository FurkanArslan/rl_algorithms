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

from rl_algorithms import build_agent
from rl_algorithms.common.abstract.agent import Agent
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.utils import Config

# app instance
app = Flask(__name__)

geniusAgent: Agent


def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=777, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--cfg-path", type=str, default="./configs/genius/sac.py", help="config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--load-from",
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

    return parser.parse_args()


def get_agent() -> Agent:
    """Main."""
    args = parse_args()

    # env initialization
    env_name = "Genius-v0"
    env = gym.make(env_name)

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = Config.fromfile(args.cfg_path)

    # If running integration test, simplify experiment
    if args.integration_test:
        cfg = common_utils.set_cfg_for_intergration_test(cfg)

    cfg.agent.env_info = dict(
        name=env_name,
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_discrete=False,
    )

    cfg.agent.log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time)
    build_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, build_args)
    agent.start_training()

    return agent


@app.route("/reset", methods=["POST"])
@cross_origin()
def reset():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get the state from the data.
    state = data["state"][0]

    geniusAgent.start_episode(state)

    return "success"


@app.route("/getAction", methods=["POST"])
@cross_origin()
def getAction():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get the state from the data.
    state = data["state"][0]

    action = geniusAgent.select_action(state)
    lastAction = (action + 1) / 2

    return str(lastAction[0])


@app.route("/postToMemory", methods=["POST"])
@cross_origin()
def postToMemory():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get relevant data.
    state = data["state"][0]
    action = data["action"]
    reward = data["reward"][0]
    next_state = data["next_state"][0]
    done = data["done"][0]

    print(data)

    # make one step & update agent parameters
    geniusAgent.make_one_step(state, action, reward, next_state, done)

    if done:
        geniusAgent.end_episode(reward)

    return "success"


@app.route("/logHistory", methods=["POST"])
@cross_origin()
def log_to_history():
    # Convert data to python dictionary.
    data = request.get_json()
    # Get relevant data.
    reward = data["utility"][0]

    geniusAgent.log_opponent_utility(reward)

    return "success"


if __name__ == "__main__":
    geniusAgent = get_agent()

    print(">>", "starting debug environment", "[%s]" % getpid())

    app.debug = False  # non refreshing
    host = environ.get("IP", "localhost")
    port = int(environ.get("PORT", 5001))
    app.run(host=host, port=port)
