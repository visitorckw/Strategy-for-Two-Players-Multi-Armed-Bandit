import game
import random

class agent:
    def play(agent, mahine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward):
        return int(random.random() * 10)

if __name__ == '__main__':
    G = game.game(10)
    G.run(agent, agent, True)