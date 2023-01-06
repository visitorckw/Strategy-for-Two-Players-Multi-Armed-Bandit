import game
from agent import advanceGreedyAgent
from agent import advanceThompsonAgent
from agent import advanceUcb
from agent import epsilonDeltaAgent
from agent import expSmoothAgent
from agent import greedyAgent
from agent import lightBGMAgent
from agent import ml_advanceUcbAgent
from agent import mlAgent
from agent import polyfitAgent
from agent import pureARAgent
from agent import randomAgent
from agent import thompsonAgent
from agent import ucbAgent
from agent import gradiantBanditAgent
from agent import adaptiveUCBAgent
from agent import ml_ucbAgent

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    G = game.game(50, 1000)
    # G.run( mlAgent.agent(DecisionTreeRegressor()), lightBGMAgent.agent(), False)
    G.run(ml_ucbAgent.agent(), greedyAgent.agent(), True)
