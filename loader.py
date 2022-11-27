import game
import randomAgent
import greedyAgent
import advanceGreedyAgent
import ucbAgent
import epsilonDeltaAgent
import mlAgent

if __name__ == '__main__':
    G = game.game(50, 1000)
    G.run(mlAgent.agent(), greedyAgent.agent(), False)