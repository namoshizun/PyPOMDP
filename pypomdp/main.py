import argparse
import os
import json
import multiprocessing
from pomdp_runner import PomdpRunner
from util import RunnerParams


if __name__ == '__main__':
    """
    Parse generic params for the POMDP runner, and configurations for the chosen algorithm.
    Algorithm configurations the JSON files in ./configs

    Example usage:
        > python main.py pomcp --env Tiger-2D.POMDP
        > python main.py pbvi --env Tiger-2D.POMDP
    """
    parser = argparse.ArgumentParser(description='Solve pomdp')
    parser.add_argument('config', type=str, help='The file name of algorithm configuration (without JSON extension)')
    parser.add_argument('--env', type=str, default='GridWorld.POMDP', help='The name of environment\'s config file')
    parser.add_argument('--budget', type=float, default=float('inf'), help='The total action budget (defeault to inf)')
    parser.add_argument('--snapshot', type=bool, default=False, help='Whether to snapshot the belief tree after each episode')
    parser.add_argument('--logfile', type=str, default=None, help='Logfile path')
    parser.add_argument('--random_prior', type=bool, default=False,
                        help='Whether or not to use a randomly generated distribution as prior belief, default to False')
    parser.add_argument('--max_play', type=int, default=100, help='Maximum number of play steps')

    args = vars(parser.parse_args())
    params = RunnerParams(**args)

    with open(params.algo_config) as algo_config:
        algo_params = json.load(algo_config)
        runner = PomdpRunner(params)
        runner.run(**algo_params)
