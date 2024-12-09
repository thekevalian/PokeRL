import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import RandomPlayer
from agents import SimpleRLPlayer, MaxDamagePlayer
import time

GEN_DATA = GenData.from_gen(9)

NB_TRAINING_STEPS = 20000
NB_EVALUATION_EPISODES = 100

np.random.seed(0)


model_store = {}


# This is the function that will be used to train the a2c
def a2c_training(player, nb_steps):
    model = A2C("MlpPolicy", player, verbose=1)
    model.learn(total_timesteps=10_000)
    model_store[player] = model


def a2c_evaluation(player, nb_episodes):
    # Reset battle statistics
    model = model_store[player]
    player.reset_battles()
    model.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "A2C Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


NB_TRAINING_STEPS = 20_000
TEST_EPISODES = 100

if __name__ == "__main__":

    print("Init Players")
    opponent = RandomPlayer()
    env_player = SimpleRLPlayer(
        opponent=opponent,
    )
    second_opponent = MaxDamagePlayer()

    print("Make Model Learn")
    model = A2C("MlpPolicy", env_player, verbose=1)
    print(model.action_space)
    print(model.observation_space)
    model.learn(total_timesteps=NB_TRAINING_STEPS)
    obs, reward, done, _, info = env_player.step(0)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

    all_battles_completed = False
    while not all_battles_completed:
        try:
            env_player.reset_battles()
            all_battles_completed = True
        except EnvironmentError as e:
            unfinished_battles = []
            for battle in list(env_player.agent._battles.values()):
                if not battle.finished:
                    unfinished_battles.append(battle)
            print(f"{unfinished_battles} are ongoing")
            time.sleep(0.5)
    obs, _ = env_player.reset()

    finished_episodes = 0
    while True:
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env_player.step(action)
        except RuntimeError as e:
            print(f"{e} at finished episodes {finished_episodes}")
            done = True
        if done:
            finished_episodes += 1
            if finished_episodes >= TEST_EPISODES:
                break
            obs, _ = env_player.reset()

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)

    finished_episodes = 0
    env_player._opponent = second_opponent

    env_player.reset_battles()
    obs, _ = env_player.reset()
    time.sleep(3)

    while True:
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env_player.step(action)
        except RuntimeError as e:
            print(f"{e} at finished episodes {finished_episodes}")
            done = True
        if done:
            finished_episodes += 1
            if finished_episodes >= TEST_EPISODES:
                break
            obs, _ = env_player.reset()

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)
