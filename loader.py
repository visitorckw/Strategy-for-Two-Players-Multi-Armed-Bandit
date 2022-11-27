import game
import randomAgent
import greedyAgent
import advanceGreedyAgent
import ucbAgent
import epsilonDeltaAgent
import mlAgent
import advanceUcb
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

if __name__ == '__main__':
    G = game.game(50, 1000)
    # G.run(mlAgent.agent(50, 1000, LinearRegression(), True, 100),mlAgent.agent(50, 1000, KNeighborsRegressor(), True, 100), False)
    G.run(advanceGreedyAgent.agent(), greedyAgent.agent(), True)