import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, runner_wrapper
import torch

def train(args):
    print(f"[TRAIN] Creating environment: {args.task}")
    sys.stdout.flush()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"[TRAIN] Environment created successfully")
    sys.stdout.flush()
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    exp_name = args.experiment_name if args.experiment_name else "none"
    ppo_runner = runner_wrapper(ppo_runner,
        name='_'.join([args.task, exp_name, datetime.now().strftime("%Y/%m/%d-%H:%M")]), 
        group=args.task, 
        mode='online', 
        dir='logs/' + args.task
    )
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
