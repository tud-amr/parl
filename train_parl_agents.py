import gymnasium as gym

from src.model_based_agents.parl import Parl
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from torch import nn
import sys

import yaml
from importlib import import_module

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def wrap_env(env_name, wraps, n_envs=1, env_config={}):
    envs = []
    for i in range(n_envs):
        for wrapper_class_str in wraps:
            env = gym.make(env_name, render_mode='rgb_array', **env_config)
            module_name, class_name = wrapper_class_str.rsplit('.', 1)
            module = import_module(module_name)
            wrapper_class = getattr(module, class_name)
            env = wrapper_class(env)
        envs.append(lambda: env)
    return envs


def train_parl(game,steps, k=None, seed=None):
    if seed is not None:
        set_random_seed(seed)
    else:
        set_random_seed(0)
    config = load_config('parameters_parl.yml')
    params = config[game]['parameters']

    # Create environment
    env_name = config[game]['environment']['name']
    render_mode = config[game]['environment']['render_mode']
    if 'config' in config[game]['environment']:
        env_config = {'config': config[game]['environment']['config']}
    else:
        env_config = {}
    num_envs = config[game]['environment']['n_envs']
    if 'policy_kwargs' not in params:
        params['policy_kwargs'] = {}
    if "activation_fn" in params["policy_kwargs"]:
        params["policy_kwargs"]["activation_fn"] = eval(params["policy_kwargs"]["activation_fn"])
    if game in ["roundabout-v0", "highway-fast-v0", "DynamicObstaclesSwitch-8x8-v0", "SlipperyDistShift-v0"]:
        env = make_vec_env(game, seed=seed, n_envs=num_envs, vec_env_cls=DummyVecEnv,
                                   wrapper_class=gym.wrappers.flatten_observation.FlattenObservation,
                           env_kwargs=env_config)
    else:
    # Apply wrappers
        try:
            wrapper_classes = config[game]['wrappers']
            envs = wrap_env(env_name, wrapper_classes, num_envs)
        except KeyError:
            envs = [lambda: gym.make(env_name, render_mode=render_mode, **env_config) for i in range(num_envs)]
            pass
        env = DummyVecEnv(envs)
    if config[game]['environment']['normalize']:
        env = VecNormalize(env, norm_reward=False)

    if k is not None:
        params['cm_w'] = k

    model = Parl("MlpPolicy", env, verbose=1, tensorboard_log='./logs/model_based/' + game + '/', device="cpu", **params)
    model.learn(total_timesteps=steps)
    model.save(
        game+"_parl",
    )
    if isinstance(env, VecNormalize):
        env = model.get_env()
        env.save(game+"_env.pkl")


if __name__ == "__main__":
    # Get game from command line
    game = sys.argv[1]
    # Get k from command line
    k = None
    if len(sys.argv) > 3:
        k = float(sys.argv[3])
    seed = None
    if len(sys.argv) > 4:
        seed = int(sys.argv[4])
    steps = int(sys.argv[2])
    train_parl(game,steps, k, seed)



