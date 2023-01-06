import game
import random
import sklearn
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import math

class agent():
    def __init__(self, model = LinearRegression(), pretrain_model = None, training_games = 100, machine_numbers = 50, game_rounds = 1000, train = True, c = math.sqrt(2)):
        self.c = c
        self.delta = 1e-9
        self.machine_numbers = machine_numbers
        self.game_rounds = game_rounds
        self.model = model
        if pretrain_model:
            self.model = joblib.load(pretrain_model)
        if train and training_games:
            self.train(training_games)
    def train(self, training_games = 100):
        train_X = []
        train_Y = []
        for i in range(training_games):
            print('traning steps', str(i+1) + '/' + str(training_games))
            G = game.game(self.machine_numbers, self.game_rounds, i)
            round = 0
            while round < self.game_rounds:
                if round < self.machine_numbers * 2:
                    if round % 2 == 0: #agent1
                        G.agent1Play(round // 2, False)
                    else:
                        G.agent2Play(round // 2, False)
                    round += 1
                    continue
                for j in range(self.machine_numbers):
                    if round % 2 == 0: #agent1
                        train_X.append([G.agent1Push[j] / round * 2, G.agent2Push[j] / round * 2, G.agent1MachineReward[j] / G.agent1Push[j]])
                        reward = G.agent1Play(j, False)
                        train_Y.append([reward])
                        G.undoAgent1()
                    else:
                        train_X.append([G.agent2Push[j] / round * 2, G.agent1Push[j] / round * 2, G.agent2MachineReward[j] / G.agent2Push[j]])
                        reward = G.agent2Play(j, False)
                        train_Y.append([reward])
                        G.undoAgent2()

                choice = int(random.random() * self.machine_numbers)
                if round % 2 == 0: #agent1
                    G.agent1Play(choice, False)
                else:
                    G.agent2Play(choice, False)
                round += 1
        train_Y = np.array(train_Y)
        train_Y = train_Y.reshape((-1,))
        print(train_Y.shape)
        self.model.fit(train_X, train_Y)
        # joblib.dump(self.model, 'randomForest_model')
        print('counting scores')
        print(self.model.score(train_X, train_Y))
    def likely(self, prob, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        p = prob
        ans = 1
        for i in range(max(len(my_history_choice), len(opp_history_choice))):
            if agent == 1:
                if i < len(my_history_choice):
                    if my_history_choice[i] == machine:
                        if my_history_reward[i] == 1:
                            ans *= p
                        else:
                            ans *= (1 - p)
                        p *= 0.97
                if i < len(opp_history_choice):
                    if opp_history_choice[i] == machine:
                        p *= 0.97
            else:
                if i < len(opp_history_choice):
                    if opp_history_choice[i] == machine:
                        p *= 0.97
                if i < len(my_history_choice):
                    if my_history_choice[i] == machine:
                        if my_history_reward[i] == 1:
                            ans *= p
                        else:
                            ans *= (1 - p)
                        p *= 0.97
        return ans
    def dfdx(self, x, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        y = self.likely(x, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
        y1 = self.likely(x+self.delta, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
        return (y1 - y) / self.delta
    def binary_search(self, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        delta = 1e-9
        L = 0
        R = 1 - delta
        while L < R:
            dL = self.dfdx(L, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            dR = self.dfdx(R, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            if dL * dR >= 0:
                break
            M = (L + R) / 2
            dM = self.dfdx(M, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            if abs(dM) <= self.delta:
                L = R = M
                break
            if dM > 0:
                L = M + self.delta
            else:
                R = M - self.delta
        return (L + R) / 2
    def linear_search(self, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        N = 100
        maxLikely = -1
        ans = -1
        for i in range(N+1):
            p = i / N
            like = self.likely(p, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            if like > maxLikely:
                maxLikely = like
                ans = p
        return ans
    def play(self, data):
        agent = data.agent
        machine_numbers = data.machine_numbers
        total_round = data.total_round
        current_round = data.current_round
        my_total_rewards = data.my_total_rewards
        my_history_choice = data.my_history_choice 
        opp_history_choice = data.opp_history_choice 
        my_history_reward = data.my_history_reward
        my_push_distribute = data.my_push_distribute
        opp_push_distribute = data.opp_push_distribute
        my_reward_distribute = data.my_reward_distribute
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        X = []
        for i in range(machine_numbers):
            X.append([my_push_distribute[i] / current_round * 2, opp_push_distribute[i] / current_round * 2, my_reward_distribute[i] / my_push_distribute[i]])
        ml_predict_y = self.model.predict(X)
        expectProb = []
        for i in range(machine_numbers):
            prob = self.linear_search(agent, i, my_history_choice, opp_history_choice, my_history_reward)
            expectProb.append(prob * (0.97 ** (my_push_distribute[i] + opp_push_distribute[i])))
        n = [0 for i in range(machine_numbers)]
        N = len(my_history_choice)
        for i in range(len(my_history_choice)):
            n[my_history_choice[i]] += 1
        maxUCB = -1
        choice = -1
        beta = 0.1
        for i in range(machine_numbers):
            ucb = expectProb[i] + self.c * math.sqrt(math.log(N) / n[i]) + beta * ml_predict_y[i]
            if ucb > maxUCB:
                maxUCB = ucb
                choice = i
        return choice
