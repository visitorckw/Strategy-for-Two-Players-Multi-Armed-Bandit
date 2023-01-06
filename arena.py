import game
from agent import advanceGreedyAgent
from agent import advanceThompsonAgent
from agent import advanceUcb
from agent import epsilonDeltaAgent
from agent import expSmoothAgent
from agent import greedyAgent
from agent import lightBGMAgent
from agent import lstmAgent
from agent import ml_advanceUcbAgent
from agent import mlAgent
from agent import polyfitAgent
from agent import pureARAgent
from agent import randomAgent
from agent import thompsonAgent
from agent import ucbAgent
import sklearn
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os
import pandas as pd

if __name__ == '__main__':
    fn = "result.csv"
    # G = game.game(50, 1000)
    # G.run(mlAgent.agent(50, 1000, LinearRegression(), True, 100),mlAgent.agent(50, 1000, KNeighborsRegressor(), True, 100), False)
    # G.run(advanceGreedyAgent.agent(), lstmAgent.agent('lstm_model.h5', 0), True)
    # G.run(ml_advanceUcbAgent.agent(), greedyAgent.agent(), True)
    if (os.path.isfile(fn)):
        data = pd.read_csv(fn)
    agent = [
        advanceThompsonAgent.agent(),
        epsilonDeltaAgent.agent(),
        expSmoothAgent.agent(),
        greedyAgent.agent(),
        lightBGMAgent.agent(),
        mlAgent.agent(),
        polyfitAgent.agent(),
        pureARAgent.agent(),
        randomAgent.agent(),
        thompsonAgent.agent(),
        ucbAgent.agent(),
    ]
    result = [[0 for j in range(len(agent))] for i in range(len(agent))]
    rating = [1500 for i in range(len(agent))]
    K = 16
    for i in range(len(agent)):
        for j in range(len(agent)):
            if i == j:
                continue
            G = game.game(50, 1000, dataCollect=True, dataName = "Train2Data")
            res = G.run(agent[i], agent[j])
            result[i][j] = res

            SA = 0.5
            SB = 0.5
            RA = rating[i]
            RB = rating[j]
            EA = 1 / (1 + 10 ** ((RA - RB) / 400))
            EB = 1 / (1 + 10 ** ((RB - RA) / 400))
            if res == 1:
                SA = 1
                SB = 0
            elif res == -1:
                SA = 0
                SB = 1
            newRA = RA + K * (SA - EA)
            newRB = RB + K * (SB - EB)

            rating[i] = newRA
            rating[j] = newRB
    print(result)
    print(rating)
