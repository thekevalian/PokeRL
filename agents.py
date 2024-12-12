import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer
from poke_env.environment import Battle
import time

GEN_DATA = GenData.from_gen(9)


class Players:
    class SimpleRLPlayer0(Gen9EnvSinglePlayer):
        # 94 x RandomPlayer (wins)
        # 58 x MaxDamagePlayer (wins)
        def embed_battle(self, battle: Battle):

            # -1 indicates that the move does not have a base power
            # or is not available
            moves_base_power = -np.ones(4)
            moves_dmg_multiplier = np.ones(4)

            for i, move in enumerate(battle.available_moves):
                battle.active_pokemon.stab_multiplier
                moves_base_power[i] = (
                    move.base_power / 100
                )  # Simple rescaling to facilitate learning
                if move.type:
                    moves_dmg_multiplier[i] = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=GEN_DATA.type_chart,
                    )

            # We count how many pokemons have not fainted in each team
            remaining_mon_team = (
                len([mon for mon in battle.team.values() if mon.fainted]) / 6
            )
            remaining_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
            )

            # Final vector with 10 components
            return np.concatenate(
                [
                    moves_base_power,
                    moves_dmg_multiplier,
                    [remaining_mon_team, remaining_mon_opponent],
                ]
            )

        def calc_reward(self, last_state, current_state) -> float:
            return self.reward_computing_helper(
                current_state, fainted_value=2, hp_value=1, victory_value=30
            )

        def describe_embedding(self):
            low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
            high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]

            return Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )

    # We define our RL player
    # It needs a state embedder and a reward computer, hence these two methods
    class SimpleRLPlayer1(Gen9EnvSinglePlayer):
        # 66 x RandomPlayer (wins)
        # 21 x MaxDamagePlayer (wins)
        def embed_battle(self, battle: Battle):

            dmg_multipliers = np.zeros(24)
            base_powers = np.zeros(24)
            for i, pokemon in enumerate(battle.team.values()):
                if not pokemon.fainted:
                    for j, move in enumerate(pokemon.moves.values()):
                        stab_mult = max(
                            (move.type in pokemon.types) * pokemon.stab_multiplier, 1
                        )
                        dmg_multipliers[4 * i + j] = (
                            move.type.damage_multiplier(
                                battle.opponent_active_pokemon.type_1,
                                battle.opponent_active_pokemon.type_2,
                                type_chart=GEN_DATA.type_chart,
                            )
                            * stab_mult
                        )
                        base_powers[4 * i + j] = move.base_power / 100
            return np.concatenate([dmg_multipliers, base_powers])

        def calc_reward(self, last_state, current_state) -> float:
            return self.reward_computing_helper(
                current_state, fainted_value=2, hp_value=1, victory_value=30
            )

        def describe_embedding(self):
            low_mult = np.zeros(24)
            high_mult = np.ones(24) * 10
            low_base = -np.ones(24)
            high_base = np.ones(24) * 4
            low = np.concatenate([low_mult, low_base])
            high = np.concatenate([high_mult, high_base])

            return Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )

    class SimpleRLPlayer2(Gen9EnvSinglePlayer):
        # 76 x RandomPlayer (wins)
        # 24 x MaxDamagePlayer (wins)
        def embed_battle(self, battle: Battle):

            dmg_multipliers = np.zeros(24)

            base_powers = np.zeros(24)
            for i, pokemon in enumerate(battle.team.values()):
                if not pokemon.fainted:
                    for j, move in enumerate(pokemon.moves.values()):
                        stab_mult = max(
                            (move.type in pokemon.types) * pokemon.stab_multiplier, 1
                        )
                        move.damage
                        dmg_multipliers[4 * i + j] = (
                            move.type.damage_multiplier(
                                battle.opponent_active_pokemon.type_1,
                                battle.opponent_active_pokemon.type_2,
                                type_chart=GEN_DATA.type_chart,
                            )
                            * stab_mult
                        )
                        base_powers[4 * i + j] = move.base_power / 100
            remaining_mon_team = (
                len([mon for mon in battle.team.values() if mon.fainted]) / 6
            )
            remaining_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
            )
            return np.concatenate(
                [
                    dmg_multipliers,
                    base_powers,
                    [remaining_mon_team, remaining_mon_opponent],
                ]
            )

        def calc_reward(self, last_state: Battle, current_state: Battle) -> float:
            return self.reward_computing_helper(
                current_state, fainted_value=2, hp_value=0, victory_value=30
            )

        def describe_embedding(self):
            low_mult = np.zeros(24)
            high_mult = np.ones(24) * 10
            low_base = -np.ones(24)
            high_base = np.ones(24) * 4
            low = np.concatenate([low_mult, low_base, [0, 0]])
            high = np.concatenate([high_mult, high_base, [1, 1]])

            return Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )

    class MaxDamagePlayer(RandomPlayer):
        def choose_move(self, battle):
            # If the player can attack, it will
            if battle.available_moves:
                # Finds the best move among available ones
                best_move = max(
                    battle.available_moves, key=lambda move: move.base_power
                )
                return self.create_order(best_move)

            # If no attack is available, a random switch will be made
            else:
                return self.choose_random_move(battle)
