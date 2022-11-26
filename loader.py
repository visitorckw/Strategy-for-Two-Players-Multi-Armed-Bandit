import game
import randomAgent
import greedyAgent
import advanceGreedyAgent
import ucbAgent

if __name__ == '__main__':
    G = game.game(50, 1000)
    G.run(ucbAgent.agent(), greedyAgent.agent(), False)