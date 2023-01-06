import game
from agaent import advanceGreedyAgent
from agaent import advanceThompsonAgent
from agaent import advanceUcb
from agaent import epsilonDeltaAgent
from agaent import expSmoothAgent
from agaent import greedyAgent
from agaent import lightBGMAgent
from agaent import ml_advanceUcbAgent
from agaent import mlAgent
from agaent import polyfitAgent
from agaent import pureARAgent
from agaent import randomAgent
from agaent import thompsonAgent
from agaent import ucbAgent

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    G = game.game(50, 1000)
    G.run( mlAgent.agent(DecisionTreeRegressor()), lightBGMAgent.agent(), False)
