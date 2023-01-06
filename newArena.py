from multiprocessing import Pool, Lock
import copy
import game
from agent import advanceGreedyAgent
from agent import advanceThompsonAgent
from agent import advanceUcb
from agent import epsilonDeltaAgent
from agent import expSmoothAgent
from agent import greedyAgent
from agent import lightBGMAgent
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
import time

agent = [
    advanceThompsonAgent,
    epsilonDeltaAgent,
    expSmoothAgent,
    greedyAgent,
    lightBGMAgent,
    mlAgent,
    polyfitAgent,
    pureARAgent,
    randomAgent,
    thompsonAgent,
    ucbAgent,
]


lock = [Lock() for i in range(len(agent))]

def f(encode):
    x = encode // len(agent)
    y = encode % len(agent)
    if x == y:
        return
    agent1 = randomAgent.agent()
    agent2 = randomAgent.agent()
    agent1Name = ''
    agent2Name = ''
    lock[x].acquire()
    agent1 = copy.deepcopy(agent[x].agent())
    agent1Name = agent[x].__name__
    lock[x].release()
    lock[y].acquire()
    agent2 = copy.deepcopy(agent[y].agent())
    agent2Name = agent[y].__name__
    lock[y].release()
    G = game.game(50, 1000)
    res = G.run(agent1, agent2, False)
    folderName = 'result/'
    filename = agent1Name + '+' + agent2Name + '+' + str(os.getpid()) + '+' + str(int(time.time()))
    with open(folderName + filename, 'w') as f:
        f.write(str(res))


if __name__ == '__main__':
    while True:
        with Pool(2) as p:
            print(p.map(f, [i for i in range(len(agent) * len(agent))]))