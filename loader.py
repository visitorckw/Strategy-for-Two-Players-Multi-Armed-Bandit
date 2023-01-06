import game
from agaent import advanceGreedyAgent
from agaent import advanceThompsonAgent
from agaent import advanceUcb
from agaent import epsilonDeltaAgent
from agaent import expSmoothAgent
from agaent import greedyAgent
from agaent import lightBGMAgent
from agaent import lstmAgent
from agaent import ml_advanceUcbAgent
from agaent import mlAgent
from agaent import polyfitAgent
from agaent import pureARAgent
from agaent import randomAgent
from agaent import thompsonAgent
from agaent import ucbAgent

import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

if __name__ == '__main__':
    G = game.game(50, 1000)
    print(greedyAgent.agent().__module__)
    G.run(lightBGMAgent.agent(), greedyAgent.agent(), False)
    # win = 0
    # tie = 0
    # for i in range(100):
    #     G = game.game(50, 1000)
    #     res = G.run(mlAgent.agent(pretrain_model='linearRegression_model', training_games=0), greedyAgent.agent(), False)
    #     if res == 1:
    #         win += 1
    #     elif res == 0:
    #         tie += 1
    # print(win, tie, 100 - win - tie)
