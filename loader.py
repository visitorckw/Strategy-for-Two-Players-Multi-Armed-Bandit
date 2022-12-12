import game
import randomAgent
import greedyAgent
import advanceGreedyAgent
import ucbAgent
import epsilonDeltaAgent
import mlAgent
import advanceUcb
import lstmAgent
import ml_advanceUcbAgent
import sklearn
import thompsonAgent
import advanceThompsonAgent
import polyfitAgent
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightBGMAgent
if __name__ == '__main__':
    G = game.game(50, 1000,dataCollect=True)
    G.run(lightBGMAgent.agent(), thompsonAgent.agent(), False)
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
