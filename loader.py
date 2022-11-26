import game
import randomAgent
import greedyAgent

if __name__ == '__main__':
    G = game.game()
    G.run(randomAgent.agent, greedyAgent.agent, True)