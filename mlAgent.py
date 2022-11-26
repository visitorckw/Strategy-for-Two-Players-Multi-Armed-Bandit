import game
import random
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

class agent():
    def __init__(self, machine_numbers = 50, game_rounds = 1000):
        self.machine_numbers = machine_numbers
        self.game_rounds = game_rounds
        self.model = DecisionTreeRegressor()
        self.train()
    def train(self, games = 100):
        for i in range(games):
            train_X = []
            train_Y = []
            G = game.game(self.machine_numbers, self.game_rounds, i)
            round = 0
            while round < self.game_rounds:
                if round < self.machine_numbers:
                    if round % 2 == 0: #agent1
                        G.agent1Play(round // 2, False)
                    else:
                        G.agent2Play(round // 2, False)
                    round += 1
                    continue
                x = []
                y = []
                if round % 2 == 0:
                    x = self.preprocess(self.machine_numbers, G.historyAgent1Choice, G.historyAgent1Reward)[0]
                else:
                    x = self.preprocess(self.machine_numbers, G.historyAgent2Choice, G.historyAgent2Reward)[0]
                for j in range(self.machine_numbers):
                    if round % 2 == 0: #agent1
                        reward = G.agent1Play(j, False)
                        y.append(reward)
                        G.undoAgent1()
                    else:
                        reward = G.agent2Play(j, False)
                        y.append(reward)
                        G.undoAgent2()
                train_X.append(x)
                train_Y.append(y)
                choice = int(random.random() * self.machine_numbers)
                G.agent1Play(choice, False)
                round += 1
            # import numpy as np
            # print('trainX shape', np.array(train_X).shape)
            # print('trainY shape', np.array(train_Y).shape)
            # print('BEFORE')
            self.model.fit(train_X, train_Y)
            # print('AFTER')
    def preprocess(self, machine_numbers, my_history_choice, my_history_reward):
        push_time = [0 for i in range(machine_numbers)]
        reward = [0 for i in range(machine_numbers)]
        for i in range(len(my_history_choice)):
            push_time[my_history_choice[i]] += 1
            reward[my_history_choice[i]] += my_history_reward[i]
        for i in range(machine_numbers):
            push_time[i] /= len(my_history_choice)
            reward[i] /= len(my_history_choice)
        X = []
        for x in push_time:
            X.append(x)
        for x in reward:
            X.append(x)
        return [X]
    def play(self, agent, machine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward):
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        X = self.preprocess(machine_numbers, my_history_choice, my_history_reward)
        y = self.model.predict(X)
        maxProfit = -1
        choice = -1
        for i in range(machine_numbers):
            if y[0][i] > maxProfit:
                maxProfit = y[0][i]
                choice = i
        return choice

if __name__ == '__main__':
    bot = agent()