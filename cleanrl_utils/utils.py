import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics, SingleRewardWrapper
from mo_gymnasium.envs.shapes_grid.shapes_grid import DIFFICULTY
import torch
import numpy as np

def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

def make_env(env_id, idx=None, seed=None, difficulty="", capture_video=False, run_name="", render=False):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        extra_kwargs = {}
        if difficulty:
            extra_kwargs["difficulty"] = DIFFICULTY[difficulty.upper()]
        if render:
            env = mo_gym.make(env_id, render_mode="human", **extra_kwargs)
        else: 
            env = mo_gym.make(env_id, **extra_kwargs)
        # env = mo_gym.wrappers.LinearReward(env, weight=np.array([0.8, 0.2]))
        # env = TimeLimit(env, max_episode_steps=100)  # ensure episodes end
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        if idx is not None:
            env = SingleRewardWrapper(env, idx)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer