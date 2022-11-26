import game
import randomAgent
import greedyAgent
import advanceGreedyAgent

if __name__ == '__main__':
    G = game.game(50, 1000)
    G.run(advanceGreedyAgent.agent(), greedyAgent.agent(), False)