import game
import random

class agent:
    def play():
        return int(random.random() * 10)

if __name__ == '__main__':
    G = game.game(10, 10)
    G.run(agent, agent, True)