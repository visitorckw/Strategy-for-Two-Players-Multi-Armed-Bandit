import game
import randomAgent
import greedyAgent
import advanceGreedyAgent
import ucbAgent
import epsilonDeltaAgent

if __name__ == '__main__':
    G = game.game(50, 1000)
    G.run(epsilonDeltaAgent.agent(), greedyAgent.agent(), False)