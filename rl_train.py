import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData
import os
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer
from agents import Players

GEN_DATA = GenData.from_gen(9)

model_dir = "models"
log_dir = "logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

opponent = RandomPlayer()
second_opponent = Players.MaxDamagePlayer()
env_player = Players.SimpleRLPlayer0(
    opponent=second_opponent
)  # Change your player here

env_player.reset_battles()

model = A2C("MlpPolicy", env_player, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1, 300):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/{TIMESTEPS*i}")
