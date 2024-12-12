import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData
from agents import Players
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

GEN_DATA = GenData.from_gen(9)


model_dir = "models"
log_dir = "logs"

# Test your model against RandomPlayer and MaxDamagePlayer
# If you think your model is really much better, then face off against SimpleHeuristic Player
opponent = RandomPlayer()
second_opponent = Players.MaxDamagePlayer()
env_player = Players.SimpleRLPlayer0(opponent=opponent)
env_player.reset_battles()

model_path = f"{model_dir}/80000.zip"  # Change this to your best model
model = A2C.load(model_path, env=env_player, device="cpu")

TEST_EPISODES = 100
obs, _ = env_player.reset()
finished_episodes = 0
while True:
    try:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)
    except RuntimeError as e:
        done = True
    if done:
        finished_episodes += 1
        if finished_episodes >= TEST_EPISODES:
            break
        obs, _ = env_player.reset()

print("Won", env_player.n_won_battles, "battles against", env_player._opponent)
env_player.reset_battles()
env_player.set_opponent(second_opponent)
TEST_EPISODES = 100
obs, _ = env_player.reset()
finished_episodes = 0
while True:
    try:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)
    except RuntimeError as e:
        done = True
    if done:
        finished_episodes += 1
        if finished_episodes >= TEST_EPISODES:
            break
        obs, _ = env_player.reset()

print("Won", env_player.n_won_battles, "battles against", env_player._opponent)
